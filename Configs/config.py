from dataclasses import dataclass, field
from typing import List

@dataclass
class PathsDetails:
    raw_data: str = "Data/raw_data.parquet"
    train_data: str = "Data/training_data.parquet"
    model_save: str = "Model/best_model.pth"
    logs: str = "Logs"

@dataclass
class DataParams:
    test_size: float = 0.10
    val_size: float = 0.10
    random_state: int = 42
    sample_frac: float = 0.08
    delay_threshold_1: int = 3   
    delay_threshold_2: int = 7 
    delay_threshold_3: int = 14
    
    min_sequence_length: int = 2
    
    days_since_last_default_fill: int = 999
    days_since_last_default_clip: int = 365
    days_since_last_invoice_clip: int = 365
    value_ratio_clip_min: float = 0.1
    value_ratio_clip_max: float = 10.0
    hist_mean_value_clip_min: float = 0.01

@dataclass
class TemporalFeatureParams:
    days_in_week: int = 7
    days_in_month: int = 31
    months_in_year: int = 12
    month_start_threshold: int = 5
    month_end_threshold: int = 25
    weekend_start_day: int = 5

@dataclass
class AugmentationParams:
    temporal_cutout_ratio: float = 0.15
    feature_dropout_ratio: float = 0.2
    gaussian_noise_std: float = 0.05
    mixup_alpha: float = 0.2

@dataclass
class LossParams:
    asymmetric_gamma_negative: int = 4
    asymmetric_gamma_positive: int = 1
    asymmetric_clip: float = 0.05
    label_smoothing: float = 0.1
    temperature_init: float = 1.5
    temperature_calibration_lr: float = 0.01
    temperature_calibration_max_iter: int = 50

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
        # Temporal features (categóricas)
        'venc_dayofweek', 'venc_quarter', 'venc_is_weekend', 
        'venc_is_month_start', 'venc_is_month_end', 
        # Sequence features (categóricas)
        'is_first_invoice', 'is_improving', 'is_first_contract'
    ])

    cont_cols: List[str] = field(default_factory=lambda: [
        # Original
        'quantidadeDiarias', 'valor_caucao_brl',
        # Temporal features (cíclicas)
        'venc_dayofweek_sin', 'venc_dayofweek_cos',
        'venc_day_sin', 'venc_day_cos',
        'venc_month_sin', 'venc_month_cos',
        # User history features
        'hist_mean_delay', 'hist_std_delay', 'hist_max_delay',
        'hist_default_rate', 'hist_payment_count',
        'last_delay', 'delay_trend', 'days_since_last_default',
        # Sequence features (user level)
        'seq_position', 'seq_position_norm', 'days_since_last_invoice',
        'rolling_mean_delay_3', 'rolling_max_delay_3',
        # Contract features (locacaoId level)
        'parcela_position', 'parcela_position_norm', 'n_contratos_anteriores',
        'contract_mean_delay',
        # Value features
        'value_ratio'
    ])

    target_cols: List[str] = field(default_factory=lambda: [
        'target_short',
        'target_medium',
        'target_long'
    ])

    static_cols: List[str] = field(default_factory=lambda: [
        'recorrencia_pagamento', 'sexo', 'faixa_idade_resumida',
        'veiculo_modelo', 'pacoteNome',
        'formaPagamento', 'lugar', 'regiao',
        'produto_categoria', 'quantidadeDiarias', 'valor_caucao_brl'
    ])

    keep_cols_final: List[str] = field(default_factory=lambda: [
        'usuarioId', 'vencimentoData', 'pagamentoData', 'Dias_atraso',
        'quantidadeDiarias', 'valor_caucao_brl',
        'target_short', 'target_medium', 'target_long'
    ])
    
    user_id_col: str = 'usuarioId'
    contract_id_col: str = 'locacaoId'
    order_col: str = 'ordem_parcela'
    due_date_col: str = 'vencimentoData'
    payment_date_col: str = 'pagamentoData'
    delay_col: str = 'Dias_atraso'
    category_col: str = 'categoria'
    category_filter: str = 'aluguel'
    value_col: str = 'valor_caucao_brl'
    
    sort_cols: List[str] = field(default_factory=lambda: ['usuarioId', 'locacaoId', 'ordem_parcela'])
    group_cols: List[str] = field(default_factory=lambda: ['usuarioId', 'locacaoId'])

@dataclass
class ModelParams:
    batch_size: int = 256
    max_seq_len: int = 50
    epochs: int = 50
    lr: float = 1e-3
    hidden_dim: int = 128
    n_blocks: int = 3
    dropout: float = 0.1
    n_heads: int = 4
    weight_decay: float = 1e-4
    outcome_dim: int = 1
    patience: int = 10
    mixed_precision: bool = True
    max_grad_norm: float = 1.0
    num_workers: int = 8
    pin_memory: bool = num_workers > 0
    persistent_workers: bool = num_workers > 0
    prefetch_factor: int = 0
    
    num_invoice_layers: int = 1
    
    scheduler_type: str = 'cosine'
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    min_lr: float = 1e-6
    scheduler_t0: int = 10
    scheduler_t_mult: int = 2
    warmup_epochs: int = 2
    
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 2000
    
    use_grn: bool = True
    use_cross_attention: bool = True
    use_transformer_encoder: bool = True
    use_grn_heads: bool = True
    use_temperature_scaling: bool = True
    use_temporal_attention: bool = True
    drop_path_rate: float = 0.1
    
    loss_type: str = 'asymmetric'
    
    use_augmentation: bool = True
    augment_prob: float = 0.1
    
    embedding_dropout: float = 0.1
    periodic_sigma: float = 1.0
    
    weight_init_std: float = 0.02
    
    plr_num_bins: int = 8
    plr_boundary_min: float = -3.0
    plr_boundary_max: float = 3.0
    
    head_dropout_multiplier_medium: float = 1.25
    head_dropout_multiplier_long: float = 1.5

@dataclass
class TensorBoardParams:
    port: str = "6006"
    start_delay: int = 3
    host: str = "localhost"

@dataclass
class Config:
    paths:        PathsDetails = field(default_factory=PathsDetails)
    data:         DataParams = field(default_factory=DataParams)
    temporal:     TemporalFeatureParams = field(default_factory=TemporalFeatureParams)
    augmentation: AugmentationParams = field(default_factory=AugmentationParams)
    loss:         LossParams = field(default_factory=LossParams)
    columns:      Columns = field(default_factory=Columns)
    model:        ModelParams = field(default_factory=ModelParams)
    tensorboard:  TensorBoardParams = field(default_factory=TensorBoardParams)

config = Config()
