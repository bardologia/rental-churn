from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PathsDetails:
    data_dir             : str = "data"
    raw_data             : str = "data/raw_data.parquet"
    train_data           : str = "data/training_data.parquet"
    runs_dir             : str = "runs"
    ablation_dir         : str = "ablation"


@dataclass
class SplitParams:
    test_size     : float = 0.10
    val_size      : float = 0.10


@dataclass
class LoadParams:
    use_cache            : bool  = True
    sample_fraction      : float = 1.0
    load_sample_fraction : float = 1.0
    user_sample_count    : int   = None
    random_state         : int   = 42


@dataclass
class TargetParams:
    target_threshold     : Optional[float] = 30
    rolling_window_sizes : List[int]       = field(default_factory=lambda: [3, 5])
    delay_known_value    : int             = 1
    use_log1p_transform  : bool            = True
    clip_target_min      : float           = 0.0


@dataclass
class TemporalFeatureParams:
    days_in_week      : int = 7
    months_in_year    : int = 12
    weekend_start_day : int = 5


@dataclass
class AugmentationParams:
    temporal_cutout_ratio : float = 0.20
    feature_dropout_ratio : float = 0.25
    gaussian_noise_std    : float = 0.08
    time_warp_probability : float = 0.15


@dataclass
class LayerWiseParams:
    tokenizer_lr        : float = 1e-3
    invoice_encoder_lr  : float = 5e-4
    sequence_encoder_lr : float = 3e-4
    cross_attention_lr  : float = 3e-4
    head_lr             : float = 1e-3
    
    tokenizer_name        : str = "tokenizer"
    invoice_encoder_name  : str = "invoice_encoder"
    sequence_encoder_name : str = "sequence_encoder"
    cross_attention_name  : str = "temporal_attention"
    head_name             : str = "head_days"
    
    tokenizer_weight_decay        : Optional[float] = None
    invoice_encoder_weight_decay  : Optional[float] = None
    sequence_encoder_weight_decay : Optional[float] = None
    cross_attention_weight_decay  : Optional[float] = None
    head_weight_decay             : Optional[float] = None


@dataclass
class TrainingParams:
    batch_size         : int   = 256
    epochs             : int   = 35
    warmup_enabled     : bool  = True
    warmup_steps       : int   = 100
    warmup_start_factor: float = 0.1
    grad_accum_steps   : int   = 1
    dropout            : float = 0.10
    weight_decay       : float = 2e-4
    patience           : int   = 6
    high_target_weight : float = 0.3
    mixed_precision    : bool  = True
    max_grad_norm      : float = 1.0
    num_workers        : int   = 4
    pin_memory         : bool  = True
    persistent_workers : bool  = True
    prefetch_factor    : int   = 4
    device             : str   = None


@dataclass
class LossParams:
    huber_delta         : float = 1.0
    quantile_weight     : float = 0.3
    threshold_weight    : float = 0.2
    quantiles           : List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    thresholds          : List[float] = field(default_factory=lambda: [15.0, 30.0])


@dataclass
class ArchitectureParams:
    max_seq_len                 : int   = 50
    min_seq_len                 : int   = 2
    hidden_dim                  : int   = 128
    num_attention_heads         : int   = 4
    num_invoice_encoder_layers  : int   = 1
    num_sequence_encoder_layers : int   = 3
    drop_path_rate              : float = 0.10
    use_augmentation            : bool  = True
    augment_prob                : float = 0.25
    embedding_dropout           : float = 0.05
    periodic_sigma              : float = 1.0
    rotary_embedding_base       : float = 10000.0
    categorical_padding_value   : int   = 0
    continuous_padding_value    : float = 0.0


@dataclass
class CosineSchedulerParams:
    t_max: int = None  
    eta_min: float = 1e-6


@dataclass
class EMAParams:
    use_ema                : bool  = True
    ema_decay              : float = 0.9999
    ema_warmup_steps       : int   = 2000
    ema_warmup_denominator : int   = 10


@dataclass
class EarlyStoppingParams:
    enabled  : bool  = True
    patience : int   = 6
    mode     : str   = "min"


@dataclass
class OverfitParams:
    overfit_single_batch     : bool  = False
    overfit_epochs           : int   = 100
    overfit_patience         : int   = 100
    overfit_dropout          : float = 0.0
    overfit_weight_decay     : float = 0.0
    overfit_number_of_users  : int   = 3
    overfit_min_length       : int   = 2
    overfit_test_size        : float = 0.3
    overfit_val_size         : float = 0.3
    overfit_use_ema          : bool  = True
    overfit_use_augmentation : bool  = True
    overfit_mixed_precision  : bool  = True


@dataclass
class AblationParams:
    user_sample_count: int = 300
    hidden_dim: int = 128
    num_invoice_encoder_layers: int = 1
    num_sequence_encoder_layers: int = 1
    num_attention_heads: int = 4
    training_epochs: int = 10
    patience: int = 4
    scheduler_patience: int = 2
    overfit_single_batch: bool = True
    metric: str = "rmse"


@dataclass
class Columns:
    keep_cols: List[str] = field(default_factory=lambda: [
        'usuarioId', 'locacaoId', 'ordem_parcela',
        'criacaoData', 'vencimentoData', 'pagamentoData',
        'Dias_atraso', 'categoria',
        'valor_brl', 'valor_pago_brl', 'valor_caucao_brl',
        'recorrencia_pagamento', 'sexo', 'faixa_idade_resumida',
        'veiculo_modelo', 'pacoteNome', 'formaPagamento', 'lugar', 'regiao', 'produto_categoria',
        'quantidadeDiarias', 'valor_caucao_entrada_brl', 'pacoteDuracao'
    ])
    
    date_cols: List[str] = field(default_factory=lambda: [
        'vencimentoData', 'pagamentoData', 'criacaoData', 
    ])
    
    cat_cols: List[str] = field(default_factory=lambda: [
        'recorrencia_pagamento', 'sexo', 'faixa_idade_resumida',
        'veiculo_modelo', 'pacoteNome', 'formaPagamento',
        'lugar', 'regiao', 'produto_categoria',
        'venc_dayofweek', 'venc_quarter', 'venc_is_weekend', 
        'venc_is_month_start', 'venc_is_month_end', 
        'is_first_invoice', 'is_improving', 'is_first_contract', 
        'forma_pagamento_caucao', 
    ])

    cont_cols: List[str] = field(default_factory=lambda: [
        'quantidadeDiarias', 'valor_caucao_brl', 'valor_brl',
        'venc_dayofweek_sin', 'venc_dayofweek_cos',
        'venc_day_sin', 'venc_day_cos',
        'venc_month_sin', 'venc_month_cos',
        'venc_day', 'venc_month',
        'hist_mean_delay', 'hist_std_delay', 'hist_max_delay',
        'last_delay', 'delay_trend',
        'seq_position', 'seq_position_norm', 'days_since_last_invoice',
        'rolling_mean_delay_3', 'rolling_max_delay_3',
        'rolling_mean_delay_5', 'rolling_max_delay_5',
        'parcela_position', 'parcela_position_norm', 'parcela_total',
        'num_contracts', 'contract_mean_delay',
        'value_ratio', 'hist_mean_value',
        'grace_period', 'total_billed', 'total_paid',
        'delay_clipped', 'valor_caucao_entrada_brl', 'pacoteDuracao', 'delay_is_known'
    ])

    target_cols: List[str] = field(default_factory=lambda: [
        'target_days_to_payment'
    ])
    
    user_id_col          : str = 'usuarioId'
    contract_id_col      : str = 'locacaoId'
    order_col            : str = 'ordem_parcela'
    
    creation_date_col    : str = 'criacaoData'
    due_date_col         : str = 'vencimentoData'
    payment_date_col     : str = 'pagamentoData'
    delay_col            : str = 'Dias_atraso'
    
    category_col         : str = 'categoria'
    category_filter      : str = 'aluguel'
    billed_value_col     : str = 'valor_brl'
    paid_value_col       : str = 'valor_pago_brl'
    security_deposit_col : str = 'valor_caucao_brl'
    
    sort_cols: List[str] = field(default_factory=lambda: ['usuarioId', 'vencimentoData', 'ordem_parcela'])
    group_cols: List[str] = field(default_factory=lambda: ['usuarioId'])
    
    delay_clipped_col: str = 'delay_clipped'
    delay_is_known_col: str = 'delay_is_known'
    target_col_name: str = 'target_days_to_payment'

    no_scale_cols: List[str] = field(default_factory=lambda: ['delay_is_known'])


@dataclass
class Config:
    paths:        PathsDetails = field(default_factory=PathsDetails)
    load:         LoadParams = field(default_factory=LoadParams)
    split:        SplitParams = field(default_factory=SplitParams)
    target:       TargetParams = field(default_factory=TargetParams)
    temporal:     TemporalFeatureParams = field(default_factory=TemporalFeatureParams)
    augmentation: AugmentationParams = field(default_factory=AugmentationParams)
    columns:      Columns = field(default_factory=Columns)
    training:     TrainingParams = field(default_factory=TrainingParams)
    loss:         LossParams = field(default_factory=LossParams)
    architecture: ArchitectureParams = field(default_factory=ArchitectureParams)
    layerwise:    LayerWiseParams = field(default_factory=LayerWiseParams)
    ema:          EMAParams = field(default_factory=EMAParams)
    early_stopping: EarlyStoppingParams = field(default_factory=EarlyStoppingParams)
    scheduler:    CosineSchedulerParams = field(default_factory=CosineSchedulerParams)
    overfit:      OverfitParams = field(default_factory=OverfitParams)
    ablation:     AblationParams = field(default_factory=AblationParams)


config = Config()
