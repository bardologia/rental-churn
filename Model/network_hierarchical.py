"""
Hierarchical Model for Payment Default Prediction

This architecture is designed specifically for sequential payment data where:
- Each user has a SEQUENCE of invoices over time
- The model should learn from the user's payment trajectory, not just aggregated features
- Temporal patterns matter (e.g., worsening behavior over time predicts default)

Architecture:
1. Feature Tokenizer: Converts categorical + continuous features to embeddings
2. Invoice Encoder: Transformer to encode each invoice's features
3. Sequence Encoder: LSTM/GRU to model the user's payment trajectory
4. Prediction Head: Combines sequence context with current invoice for prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PLREmbedding(nn.Module):
    """Piecewise Linear Encoding for continuous features"""
    def __init__(self, n_features, d_embedding, n_bins=8):
        super().__init__()
        self.n_bins = n_bins
        self.linear = nn.Linear(n_features * n_bins, n_features * d_embedding)
        self.d_embedding = d_embedding
        self.n_features = n_features
        self.bin_boundaries = nn.Parameter(torch.linspace(-3, 3, n_bins).unsqueeze(0).repeat(n_features, 1))
        
    def forward(self, x):
        B = x.shape[0]
        x_expanded = x.unsqueeze(-1)
        boundaries = self.bin_boundaries.unsqueeze(0)
        plr_encoding = F.relu(1 - torch.abs(x_expanded - boundaries))
        plr_flat = plr_encoding.view(B, -1)
        out = self.linear(plr_flat)
        return out.view(B, self.n_features, self.d_embedding)


class FeatureTokenizer(nn.Module):
    """Tokenize categorical and continuous features into embeddings"""
    def __init__(self, cardinalities, n_cont, d_token, use_plr=True, n_bins=8):
        super().__init__()
        
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(c, d_token) for c in cardinalities
        ])
        self.emb_dropout = nn.Dropout(0.1)
        self.n_cont = n_cont
        self.d_token = d_token
        self.n_cat = len(cardinalities)
        
        if n_cont > 0:
            if use_plr:
                self.cont_embedding = PLREmbedding(n_cont, d_token, n_bins=n_bins)
            else:
                self.cont_weights = nn.Parameter(torch.empty(1, n_cont, d_token))
                self.cont_bias = nn.Parameter(torch.empty(1, n_cont, d_token))
                nn.init.kaiming_uniform_(self.cont_weights, a=math.sqrt(5))
                bound = 1 / math.sqrt(self.cont_weights.size(2))
                nn.init.uniform_(self.cont_bias, -bound, bound)
            self.use_plr = use_plr
        
    def forward(self, x_cat, x_cont):
        """
        Args:
            x_cat: (B, n_cat) or (B, seq_len, n_cat)
            x_cont: (B, n_cont) or (B, seq_len, n_cont)
        Returns:
            tokens: (B, n_cat + n_cont, d_token) or (B, seq_len, n_cat + n_cont, d_token)
        """
        is_sequential = x_cat.dim() == 3
        
        if is_sequential:
            B, S, _ = x_cat.shape
            # Flatten for processing
            x_cat_flat = x_cat.view(B * S, -1)
            x_cont_flat = x_cont.view(B * S, -1)
        else:
            x_cat_flat = x_cat
            x_cont_flat = x_cont
            
        # Process categorical
        cat_tokens = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_tokens.append(emb(x_cat_flat[:, i]))
        
        if cat_tokens:
            x_cat_t = torch.stack(cat_tokens, dim=1)
            x_cat_t = self.emb_dropout(x_cat_t)
        else:
            x_cat_t = torch.empty(x_cat_flat.size(0), 0, self.d_token, device=x_cat.device)
        
        # Process continuous
        if self.n_cont > 0:
            if self.use_plr:
                x_cont_t = self.cont_embedding(x_cont_flat)
            else:
                x_cont_t = x_cont_flat.unsqueeze(-1) * self.cont_weights + self.cont_bias
        else:
            x_cont_t = torch.empty(x_cont_flat.size(0), 0, self.d_token, device=x_cont.device)
            
        # Combine
        tokens = torch.cat([x_cat_t, x_cont_t], dim=1)  # (B*S, n_features, d_token)
        
        if is_sequential:
            tokens = tokens.view(B, S, -1, self.d_token)
            
        return tokens


class InvoiceEncoder(nn.Module):
    """Encode features of a single invoice using self-attention"""
    def __init__(self, d_model, n_heads=2, n_layers=1, dropout=0.1):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, tokens):
        """
        Args:
            tokens: (B, n_features, d_model) or (B, S, n_features, d_model)
        Returns:
            invoice_repr: (B, d_model) or (B, S, d_model)
        """
        is_sequential = tokens.dim() == 4
        
        if is_sequential:
            B, S, F, D = tokens.shape
            tokens_flat = tokens.view(B * S, F, D)
        else:
            tokens_flat = tokens
            
        # Self-attention over features
        encoded = self.encoder(tokens_flat)  # (B*S, F, D)
        
        # Mean pooling over features
        pooled = encoded.mean(dim=1)  # (B*S, D)
        out = self.pool(pooled)
        
        if is_sequential:
            out = out.view(B, S, -1)
            
        return out


class SequenceEncoder(nn.Module):
    """Encode the user's payment history using LSTM"""
    def __init__(self, d_model, hidden_dim=None, n_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        
        hidden_dim = hidden_dim or d_model
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Project back to d_model if bidirectional
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.proj = nn.Linear(out_dim, d_model) if out_dim != d_model else nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, sequence, lengths=None):
        """
        Args:
            sequence: (B, max_seq_len, d_model)
            lengths: (B,) actual sequence lengths for packing
        Returns:
            context: (B, d_model) - final hidden state representing user history
            all_hidden: (B, max_seq_len, d_model) - all hidden states
        """
        if lengths is not None:
            # Pack for variable length sequences
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, (h_n, c_n) = self.lstm(packed)
            all_hidden, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            all_hidden, (h_n, c_n) = self.lstm(sequence)
        
        # Get final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final states
            context = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            context = h_n[-1]
            
        context = self.norm(self.proj(context))
        all_hidden = self.proj(all_hidden)
        
        return context, all_hidden


class TemporalAttention(nn.Module):
    """Attention over historical invoices conditioned on current invoice"""
    def __init__(self, d_model, n_heads=2, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, current, history, mask=None):
        """
        Args:
            current: (B, d_model) - current invoice representation
            history: (B, seq_len, d_model) - historical invoice representations
            mask: (B, seq_len) - True for positions to mask (padding)
        Returns:
            attended: (B, d_model) - history-aware current representation
        """
        # Expand current for attention
        query = current.unsqueeze(1)  # (B, 1, d_model)
        
        attn_out, attn_weights = self.attention(
            query, history, history, 
            key_padding_mask=mask
        )
        
        attended = self.norm(current + self.dropout(attn_out.squeeze(1)))
        return attended, attn_weights


class HierarchicalModel(nn.Module):
    """
    Hierarchical model for sequential payment prediction.
    
    This model processes each user's payment sequence to:
    1. Encode each invoice's features
    2. Model the temporal trajectory with LSTM
    3. Use attention to focus on relevant historical patterns
    4. Predict default probability for the next invoice
    """
    def __init__(
        self,
        embedding_dims,
        n_cont,
        hidden_dim=64,
        n_invoice_layers=1,
        n_sequence_layers=2,
        n_heads=2,
        dropout=0.2,
        use_temporal_attention=True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_cat = len(embedding_dims)
        self.n_cont = n_cont
        self.use_temporal_attention = use_temporal_attention
        
        cardinalities = [c[0] for c in embedding_dims]
        
        # 1. Feature tokenizer
        self.tokenizer = FeatureTokenizer(cardinalities, n_cont, hidden_dim, use_plr=True, n_bins=8)
        
        # 2. Invoice encoder (processes each invoice's features)
        self.invoice_encoder = InvoiceEncoder(
            hidden_dim, 
            n_heads=n_heads, 
            n_layers=n_invoice_layers, 
            dropout=dropout
        )
        
        # 3. Sequence encoder (models temporal trajectory)
        self.sequence_encoder = SequenceEncoder(
            hidden_dim,
            n_layers=n_sequence_layers,
            dropout=dropout,
            bidirectional=False  # Causal - only past information
        )
        
        # 4. Temporal attention (optional)
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(hidden_dim, n_heads=n_heads, dropout=dropout)
        
        # 5. Prediction head
        head_input_dim = hidden_dim * 3 if use_temporal_attention else hidden_dim * 2
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x_cat, x_cont, lengths=None):
        """
        Forward pass for non-sequential data (original interface).
        For sequential data, use forward_sequential.
        
        Args:
            x_cat: (B, n_cat) categorical features
            x_cont: (B, n_cont) continuous features
            lengths: ignored for non-sequential
        Returns:
            logits: (B, 1) prediction logits
        """
        # Tokenize features
        tokens = self.tokenizer(x_cat, x_cont)  # (B, n_features, hidden_dim)
        
        # Encode invoice
        invoice_repr = self.invoice_encoder(tokens)  # (B, hidden_dim)
        
        # For non-sequential, we just use the invoice representation
        # duplicated to match expected head input
        if self.use_temporal_attention:
            combined = torch.cat([invoice_repr, invoice_repr, invoice_repr], dim=-1)
        else:
            combined = torch.cat([invoice_repr, invoice_repr], dim=-1)
        
        logits = self.head(combined)
        return logits
    
    def forward_sequential(self, x_cat_seq, x_cont_seq, lengths):
        """
        Forward pass for sequential data (user payment history).
        
        Args:
            x_cat_seq: (B, max_seq_len, n_cat) categorical features for each invoice
            x_cont_seq: (B, max_seq_len, n_cont) continuous features for each invoice
            lengths: (B,) actual sequence lengths
        Returns:
            logits: (B, 1) prediction for last invoice in each sequence
        """
        B, S, _ = x_cat_seq.shape
        
        # 1. Tokenize all invoices
        tokens = self.tokenizer(x_cat_seq, x_cont_seq)  # (B, S, n_features, hidden_dim)
        
        # 2. Encode each invoice
        invoice_reprs = self.invoice_encoder(tokens)  # (B, S, hidden_dim)
        
        # 3. Encode the sequence
        context, all_hidden = self.sequence_encoder(invoice_reprs, lengths)  # (B, hidden_dim)
        
        # 4. Get current (last) invoice representation
        # Use lengths to index the actual last invoice
        batch_indices = torch.arange(B, device=x_cat_seq.device)
        last_indices = (lengths - 1).long()
        current_repr = invoice_reprs[batch_indices, last_indices]  # (B, hidden_dim)
        
        # 5. Apply temporal attention if enabled
        if self.use_temporal_attention:
            # Create mask for padding positions
            mask = torch.arange(S, device=x_cat_seq.device).expand(B, S) >= lengths.unsqueeze(1)
            attended, _ = self.temporal_attention(current_repr, all_hidden, mask=mask)
            combined = torch.cat([current_repr, context, attended], dim=-1)
        else:
            combined = torch.cat([current_repr, context], dim=-1)
        
        # 6. Predict
        logits = self.head(combined)
        return logits


# Compatibility wrapper to match original Model interface
class Model(HierarchicalModel):
    """
    Drop-in replacement for original Model.
    Maintains same interface but with hierarchical architecture internally.
    """
    def __init__(
        self,
        embedding_dims,
        n_cont,
        outcome_dim=1,  # ignored, always 1
        hidden_dim=64,
        n_blocks=2,  # maps to n_sequence_layers
        dropout=0.2,
        n_heads=2,
        use_grn=True,  # ignored for now
        use_cross_attention=True  # maps to use_temporal_attention
    ):
        super().__init__(
            embedding_dims=embedding_dims,
            n_cont=n_cont,
            hidden_dim=hidden_dim,
            n_invoice_layers=1,
            n_sequence_layers=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
            use_temporal_attention=use_cross_attention
        )
