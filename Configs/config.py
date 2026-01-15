from dataclasses import dataclass, field
from typing import List

@dataclass
class PathsDetails:
    raw_data: str = "Data/raw_data.csv"
    train_data: str = "Data/training_data.csv"
    model_save: str = "Model/best_model.pth"
    logs: str = "logs"

@dataclass
class DataParams:
    separator: str = ";"
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42

@dataclass
class Columns:
    
    drop_cols: List[str] = field(default_factory=lambda: [
        'emissaoNotaFiscalData', 'emissaoNotaFiscalDia', 'codigoOSOmie', 'numeroNFSeOmie', 
        'criacaoDataAcrescimoDesconto', 'adyenPspReferencePagamento' 
    ])
    
    date_cols: List[str] = field(default_factory=lambda: [
        'vencimentoData', 'pagamentoData', 'pagamentoData_UTC', 'criacaoData', 
        'competenciaInicioData', 'competenciaFimData', 'atualizacao_dt'
    ])
    
    cat_cols: List[str] = field(default_factory=lambda: [
        'recorrencia_pagamento', 'sexo', 'faixa_idade_resumida', 
        'veiculo_modelo', 'pacoteNome', 'formaPagamento', 
        'lugar', 'regiao', 'produto_categoria',
        'month'  # Calendar categorical feature
    ])
    
    cont_cols: List[str] = field(default_factory=lambda: [
        'invoice_sequence', 'current_streak', 'current_streak_days',
        'hist_count_delays', 'hist_sum_val_delayed', 
        'hist_total_penalty', 'hist_max_streak', 'hist_max_delay_days',
        'hist_delay_rate', 'hist_avg_val_delayed', 'hist_avg_penalty',
        'recent_delay_rate', 'recent_avg_delay_days',
        'trend_delay_rate',
        'risk_score',
        'quantidadeDiarias', 'valor_caucao_brl',
        # Calendar/seasonality features
        'day_of_week', 'is_weekend', 'is_month_end', 'is_first_invoice'
    ])
    
    target_cols: List[str] = field(default_factory=lambda: [
        'target_default_1'  
    ])
    
    static_cols: List[str] = field(default_factory=lambda: [
        'recorrencia_pagamento', 'sexo', 'faixa_idade_resumida', 
        'veiculo_modelo', 'pacoteNome', 
        'formaPagamento', 'lugar', 'regiao', 
        'produto_categoria', 'quantidadeDiarias', 'valor_caucao_brl'
    ])
    
    keep_cols_final: List[str] = field(default_factory=lambda: [
        'usuarioId', 'vencimentoData',
        'Dias_atraso', 'invoice_sequence',
        'is_delayed', 'is_default', 'current_streak', 'current_streak_days',
        # Historical
        'hist_count_delays', 'hist_sum_val_delayed', 'hist_total_penalty',
        'hist_max_streak', 'hist_max_delay_days',
        'hist_delay_rate', 'hist_avg_val_delayed', 'hist_avg_penalty',
        # Recency
        'recent_delay_rate', 'recent_avg_delay_days',
        # Trend
        'trend_delay_rate',
        # Risk
        'risk_score',
        # Calendar/seasonality
        'day_of_week', 'is_weekend', 'is_month_end', 'month', 'is_first_invoice',
        # Target
        'target_default_1'
    ])

@dataclass
class ModelParams:
    batch_size: int = 256       # Smaller batch for better generalization
    epochs: int = 150
    lr: float = 5e-4            # Slightly higher LR for new architecture
    hidden_dim: int = 32        # Reduced: 64 -> 32 (fewer params for small dataset)
    n_blocks: int = 2           # Reduced: 3 -> 2 transformer layers
    dropout: float = 0.25       # Increased: 0.15 -> 0.25 (more regularization)
    n_heads: int = 2            # Reduced: 4 -> 2 attention heads
    weight_decay: float = 1e-3  # Increased: 5e-4 -> 1e-3 (stronger L2 reg)
    outcome_dim: int = 1        # Single target: default in next invoice
    patience: int = 25          # More patience for imbalanced convergence
    mixed_precision: bool = False
    max_grad_norm: float = 1.0
    num_workers: int = 0        # Use 0 on Windows/Jupyter to prevent kernel crashes
    pin_memory: bool = False    # Disable when num_workers=0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 8   # Faster LR reduction
    min_lr: float = 1e-6
    use_grn: bool = True          # Enable Gated Residual Networks
    use_cross_attention: bool = True  # Enable cross-feature attention
    # Imbalanced data handling - tuned for ~4% positive rate
    loss_type: str = 'asymmetric'  # Better for imbalance than focal
    focal_alpha: float = 0.75     # Weight for rare positives
    focal_gamma: float = 2.0      # Focus on hard examples
    use_balanced_sampling: bool = True  # Oversample minority class
    oversample_ratio: float = 0.20 # Moderate oversampling (4% -> 20%)

@dataclass
class TensorBoardParams:
    port: str = "6006"
    start_delay: int = 3
    host: str = "localhost"

@dataclass
class Config:
    paths:       PathsDetails = field(default_factory=PathsDetails)
    data:        DataParams = field(default_factory=DataParams)
    columns:     Columns = field(default_factory=Columns)
    model:       ModelParams = field(default_factory=ModelParams)
    tensorboard: TensorBoardParams = field(default_factory=TensorBoardParams)

config = Config()
