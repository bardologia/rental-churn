import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x
       
        keep_prob = 1 - self.drop_prob
        shape     = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask      = x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)
        return x * mask


class FourierFeatures(nn.Module):
    def __init__(self, num_features: int, embedding_dimension: int, sigma: float = 1.0):
        super().__init__()
        self.num_features        = num_features
        self.embedding_dimension = embedding_dimension
        
        self.frequencies = nn.Parameter(torch.randn(num_features, embedding_dimension // 2) * sigma)
        self.phases      = nn.Parameter(torch.zeros(num_features, embedding_dimension // 2))
        
        self.projection  = nn.Linear(embedding_dimension, embedding_dimension)
        self.gate        = nn.Linear(embedding_dimension, embedding_dimension)
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_expanded   = input_tensor.unsqueeze(-1)
        angles           = input_expanded * self.frequencies.unsqueeze(0) * 2 * math.pi + self.phases.unsqueeze(0)
        fourier_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        projected        = self.projection(fourier_features)
        gated            = torch.sigmoid(self.gate(fourier_features))
        return projected * gated


class GRN(nn.Module):
    def __init__(
        self, 
        input_dimension: int, 
        hidden_dimension: int, 
        output_dimension: int, 
        dropout: float = 0.1,
        context_dimension: Optional[int] = None
    ):
        super().__init__()
        
        self.fully_connected_1 = nn.Linear(input_dimension, output_dimension)
        self.fully_connected_2 = nn.Linear(output_dimension, output_dimension)
        
        if context_dimension is not None:
            self.context_projection = nn.Linear(context_dimension, output_dimension, bias=False)
        
        self.context_dimension = context_dimension
        
        self.gate_layer      = nn.Linear(output_dimension, output_dimension)
        self.skip_connection = nn.Linear(input_dimension, output_dimension) if input_dimension != output_dimension else nn.Identity()
        self.layer_norm      = nn.LayerNorm(output_dimension)
        self.dropout_layer   = nn.Dropout(dropout)
        
    def forward(self, input_tensor: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = F.elu(self.fully_connected_1(input_tensor))
        
        if context is not None and self.context_dimension is not None:
            hidden = hidden + self.context_projection(context)
        
        hidden      = self.dropout_layer(F.elu(self.fully_connected_2(hidden)))
        gate_values = torch.sigmoid(self.gate_layer(hidden))
        skip_values = self.skip_connection(input_tensor)
        
        return self.layer_norm(gate_values * hidden + (1 - gate_values) * skip_values)


class SwiGLU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, output_features: int, dropout: float = 0.0):
        super().__init__()
        self.gate_projection   = nn.Linear(input_features, hidden_features, bias=False)
        self.output_projection = nn.Linear(hidden_features, output_features, bias=False)
        self.up_projection     = nn.Linear(input_features, hidden_features, bias=False)
        self.dropout_layer     = nn.Dropout(dropout)
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        gate_values    = self.gate_projection(input_tensor)        
        gate_activated = F.silu(gate_values)                    

        up_values      = self.up_projection(input_tensor)             

        fused     = gate_activated * up_values                      
        projected = self.output_projection(fused)                

        output = self.dropout_layer(projected)               
        return output


class RoPE(nn.Module):
    def __init__(self, dimension: int, max_sequence_length: int = 512, base: float = 10000.0):
        super().__init__()
        self.dimension           = dimension
        self.max_sequence_length = max_sequence_length
        inverse_frequency        = 1.0 / (base ** (torch.arange(0, dimension, 2).float() / dimension))
        
        self.register_buffer('inverse_frequency', inverse_frequency)
        self._build_cache(max_sequence_length)
        
    def _build_cache(self, sequence_length: int):
        positions = torch.arange(sequence_length, device=self.inverse_frequency.device).type_as(self.inverse_frequency)
        frequencies = torch.einsum('i,j->ij', positions, self.inverse_frequency)
        embeddings = torch.cat([frequencies, frequencies], dim=-1)
        self.register_buffer('cos_cached', embeddings.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', embeddings.sin().unsqueeze(0).unsqueeze(0))
        
    def _rotate_half(self, tensor: torch.Tensor) -> torch.Tensor:
        first_half, second_half = tensor[..., :tensor.shape[-1]//2], tensor[..., tensor.shape[-1]//2:]
        return torch.cat([-second_half, first_half], dim=-1)
            
    def forward(self, query: torch.Tensor, key: torch.Tensor, sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(sequence_length, torch.Tensor):
            sequence_length = int(sequence_length.item())
        
        actual_length = min(sequence_length, self.max_sequence_length)
        cos_values    = self.cos_cached[:, :, :actual_length, :].to(query.dtype)
        sin_values    = self.sin_cached[:, :, :actual_length, :].to(query.dtype)
        
        if sequence_length > self.max_sequence_length:
            additional_positions   = torch.arange(self.max_sequence_length, sequence_length, device=self.inverse_frequency.device).type_as(self.inverse_frequency)
            additional_frequencies = torch.einsum('i,j->ij', additional_positions, self.inverse_frequency)
            additional_embeddings  = torch.cat([additional_frequencies, additional_frequencies], dim=-1)
            additional_cos         = additional_embeddings.cos().unsqueeze(0).unsqueeze(0)
            additional_sin         = additional_embeddings.sin().unsqueeze(0).unsqueeze(0)
            cos_values             = torch.cat([cos_values, additional_cos], dim=2)
            sin_values             = torch.cat([sin_values, additional_sin], dim=2)
        
        query_embedded = (query * cos_values) + (self._rotate_half(query) * sin_values)
        key_embedded   = (key * cos_values) + (self._rotate_half(key) * sin_values)
        return query_embedded, key_embedded


class PredictionHead(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, dropout: float = 0.1, num_outputs: int = 1):
        super().__init__()
        self.gated_residual_network_1 = GRN(input_dimension, hidden_dimension * 2, hidden_dimension, dropout)
        self.gated_residual_network_2 = GRN(hidden_dimension, hidden_dimension, hidden_dimension // 2, dropout)
        self.output_layer             = nn.Linear(hidden_dimension // 2, num_outputs)
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = self.gated_residual_network_1(input_tensor)
        input_tensor = self.gated_residual_network_2(input_tensor)
        return self.output_layer(input_tensor)


class FeatureTokenizer(nn.Module):
    def __init__(
        self,
        cardinalities: list,
        num_continuous: int,
        token_dimension: int,
        periodic_sigma: float,
        embedding_dropout: float,
    ):
        super().__init__()

        self.categorical_embeddings = nn.ModuleList([nn.Embedding(cardinality + 1, token_dimension, padding_idx=0) for cardinality in cardinalities])
        self.embedding_dropout      = nn.Dropout(embedding_dropout)
        self.token_dimension        = token_dimension

        self.continuous_embedding   = FourierFeatures(num_continuous, token_dimension, sigma=periodic_sigma)
            
    def forward(self, categorical_features: torch.Tensor, continuous_features: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = categorical_features.shape
        categorical_flat               = categorical_features.view(batch_size * sequence_length, -1)
        continuous_flat                = continuous_features.view(batch_size * sequence_length, -1)
        
        categorical_tokens = []
        for index, embedding in enumerate(self.categorical_embeddings):
            categorical_tokens.append(embedding(categorical_flat[:, index]))
        
        categorical_tensor = torch.stack(categorical_tokens, dim=1)
        categorical_tensor = self.embedding_dropout(categorical_tensor)
        
        continuous_tensor = self.continuous_embedding(continuous_flat)
            
        tokens = torch.cat([categorical_tensor, continuous_tensor], dim=1)
        tokens = tokens.view(batch_size, sequence_length, -1, self.token_dimension)
            
        return tokens


class InvoiceEncoder(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for layer_index in range(num_layers):
            current_drop_path_rate = drop_path_rate * layer_index / max(num_layers - 1, 1)
            self.layers.append(TransformerBlock(model_dimension, num_heads, dropout, current_drop_path_rate, is_causal=False))
        
        self.pool = nn.Sequential(
            nn.LayerNorm(model_dimension),
            nn.Linear(model_dimension, model_dimension)
        )
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  
        batch_size, sequence_length, num_features, embedding_dimension = tokens.shape
        tokens_flat = tokens.view(batch_size * sequence_length, num_features, embedding_dimension)
          
        hidden = tokens_flat
        for layer in self.layers:
            hidden = layer(hidden)
        
        pooled = hidden.mean(dim=1)
        output = self.pool(pooled)
        
        output = output.view(batch_size, sequence_length, -1)
            
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self, 
        model_dimension: int, 
        num_heads: int, 
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        rotary_positional_embedding: Optional[RoPE] = None,
        is_causal: bool = False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dimension = model_dimension // num_heads
        self.rotary_positional_embedding = rotary_positional_embedding
        self.is_causal = is_causal
        self.dropout = dropout

        self.layer_norm_1         = nn.LayerNorm(model_dimension)
        self.query_key_value      = nn.Linear(model_dimension, 3 * model_dimension, bias=False)
        self.output_projection    = nn.Linear(model_dimension, model_dimension)
        self.drop_path_1          = StochasticDepth(drop_path_rate)

        self.layer_norm_2         = nn.LayerNorm(model_dimension)
        self.feed_forward_network = SwiGLU(model_dimension, model_dimension * 4, model_dimension, dropout)
        self.drop_path_2          = StochasticDepth(drop_path_rate)

    def forward(self, input_tensor: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = input_tensor.shape
        normalized = self.layer_norm_1(input_tensor)

        qkv = self.query_key_value(normalized).reshape(B, L, 3, self.num_heads, self.head_dimension)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]  

        if self.rotary_positional_embedding is not None:
            query, key = self.rotary_positional_embedding(query, key, L)

        attn_mask = None
        if self.is_causal:
            causal = torch.triu(torch.ones(L, L, device=input_tensor.device, dtype=torch.bool), diagonal=1)  
            attn_mask = causal[None, None, :, :]  

        if key_padding_mask is not None:
            pad = key_padding_mask[:, None, None, :]  
            attn_mask = pad if attn_mask is None else (attn_mask | pad) 

        use_causal_flag = self.is_causal and key_padding_mask is None
        
        attention_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal_flag
        )

        attention_output = attention_output.transpose(1, 2).reshape(B, L, D)
        attention_output = self.output_projection(attention_output)

        x = input_tensor + self.drop_path_1(attention_output)
        x = x + self.drop_path_2(self.feed_forward_network(self.layer_norm_2(x)))
        return x


class SequenceEncoder(nn.Module):
    def __init__(
        self, 
        model_dimension: int, 
        num_heads: int = 4, 
        num_layers: int = 3, 
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        max_sequence_length: int = 512
    ):
        super().__init__()
        
        self.model_dimension = model_dimension
        self.num_heads       = num_heads
        self.head_dimension  = model_dimension // num_heads
        
        self.rotary_positional_embedding = RoPE(self.head_dimension, max_sequence_length)
        
        self.layers = nn.ModuleList()
        for layer_index in range(num_layers):
            current_drop_path_rate = drop_path_rate * layer_index / max(num_layers - 1, 1)
            self.layers.append(
                TransformerBlock(
                    model_dimension, 
                    num_heads, 
                    dropout, 
                    current_drop_path_rate,              
                    rotary_positional_embedding=self.rotary_positional_embedding, 
                    is_causal=True
                )
            )
        
        self.layer_norm = nn.LayerNorm(model_dimension)
        
    def forward(self, sequence: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        B, L, D = sequence.shape

        key_padding_mask = None
        if lengths is not None:
            key_padding_mask = torch.arange(L, device=sequence.device).expand(B, L) >= lengths.unsqueeze(1)

        hidden = sequence
        for layer in self.layers:
            hidden = layer(hidden, key_padding_mask=key_padding_mask)

        hidden = self.layer_norm(hidden)

        if lengths is not None:
            idx = torch.arange(B, device=sequence.device)
            last = (lengths - 1).long().clamp(min=0, max=L - 1)
            context = hidden[idx, last]
        else:
            context = hidden[:, -1]

        return context, hidden


class CrossAttention(nn.Module):
    def __init__(self, model_dimension: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.attention              = nn.MultiheadAttention(model_dimension, num_heads, dropout=dropout, batch_first=True)
        self.gated_residual_network = GRN(model_dimension, model_dimension * 2, model_dimension, dropout)
        self.dropout_layer          = nn.Dropout(dropout)
        
    def forward(self, current: torch.Tensor, history: torch.Tensor, mask: Optional[torch.Tensor] = None):
        
        query = current.unsqueeze(1)
        attention_output, attention_weights = self.attention(query, history, history, key_padding_mask=mask)
        attended = self.gated_residual_network(current + self.dropout_layer(attention_output.squeeze(1)))
        
        return attended, attention_weights


class Model(nn.Module):
    def __init__(
        self,
        embedding_dimensions: list,
        num_continuous: int,
        target_scaler = None,
        feature_scaler = None,
        config=None,
    ):
        super().__init__()
        self.config = config
        self.embedding_dimensions = embedding_dimensions
        self.target_scaler        = target_scaler
        self.feature_scaler       = feature_scaler
        
        self.hidden_dimension = self.config.architecture.hidden_dim
        self.num_categorical  = len(self.embedding_dimensions)
        self.num_continuous   = num_continuous

        self.tokenizer = FeatureTokenizer(
            self.embedding_dimensions,
            num_continuous,
            self.hidden_dimension,
            periodic_sigma=self.config.architecture.periodic_sigma,
            embedding_dropout=self.config.architecture.embedding_dropout,
        )

        self.invoice_encoder = InvoiceEncoder(
            self.hidden_dimension,
            num_heads=self.config.architecture.num_attention_heads,
            num_layers=self.config.architecture.num_invoice_encoder_layers,
            dropout=self.config.training.dropout,
            drop_path_rate=self.config.architecture.drop_path_rate,
        )

        self.sequence_encoder = SequenceEncoder(
            self.hidden_dimension,
            num_heads=self.config.architecture.num_attention_heads,
            num_layers=self.config.architecture.num_sequence_encoder_layers,
            dropout=self.config.training.dropout,
            drop_path_rate=self.config.architecture.drop_path_rate,
            max_sequence_length=self.config.architecture.max_seq_len,
        )

        self.temporal_attention = CrossAttention(self.hidden_dimension, num_heads=self.config.architecture.num_attention_heads, dropout=self.config.training.dropout)

        head_input_dimension = self.hidden_dimension * 3
        self.head_days       = PredictionHead(head_input_dimension, self.hidden_dimension, dropout=self.config.training.dropout, num_outputs=1)
        
    def forward(
        self, 
        categorical_sequence: torch.Tensor, 
        continuous_sequence: torch.Tensor, 
        lengths: torch.Tensor
    ) -> torch.Tensor:
        
        batch_size, sequence_length, _ = categorical_sequence.shape
        batch_indices                  = torch.arange(batch_size, device=categorical_sequence.device)
        last_indices                   = (lengths - 1).long().clamp(min=0)
        mask                           = torch.arange(sequence_length, device=categorical_sequence.device).expand(batch_size, sequence_length) >= lengths.unsqueeze(1)

        tokens                  = self.tokenizer(categorical_sequence, continuous_sequence)
        invoice_representations = self.invoice_encoder(tokens)
        
        context, all_hidden     = self.sequence_encoder(invoice_representations, lengths)
        current_representation  = invoice_representations[batch_indices, last_indices]
        attended, _             = self.temporal_attention(current_representation, all_hidden, mask=mask)
        
        combined = torch.cat([current_representation, context, attended], dim=-1)
        output = self.head_days(combined)
        
        return output.squeeze(-1)
    
