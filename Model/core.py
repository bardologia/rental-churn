import pandas as pd
import numpy as np
from Configs.config import config

def _streak(s: pd.Series) -> pd.Series:
    groups = (s != s.shift()).cumsum()
    return s * (s.groupby(groups).cumcount() + 1)

def _streak_days(df: pd.DataFrame) -> pd.Series:
    groups = (df['is_delayed'] != df['is_delayed'].shift()).cumsum()
    return df.groupby(groups)['Dias_atraso'].cumsum() * df['is_delayed']

def _filter_until_default(user_df: pd.DataFrame) -> pd.DataFrame:
    default_idx = user_df[user_df['is_default'] == 1].index
    return user_df.loc[:default_idx[0]] if len(default_idx) else user_df

def _safe_divide(a, b, fill_value=0.0):
    """Safe division that handles zeros and infinities"""
    result = a / b.replace(0, np.nan)
    return result.fillna(fill_value).replace([np.inf, -np.inf], fill_value)

def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep=config.data.separator, low_memory=False)
    
    df = (df[df['categoria'].astype(str).str.lower().str.contains('aluguel')]
          .drop(columns=[c for c in config.columns.drop_cols if c in df.columns]))
    
    for col in config.columns.date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    df['vencimentoData'] = pd.to_datetime(df['vencimentoData'], errors='coerce', utc=True)
    now_utc = pd.Timestamp.now(tz='UTC')
    df = (df[df['vencimentoData'] <= now_utc]
          .sort_values('vencimentoData')
          .reset_index(drop=True))
    
    num_cols = [c for c in df.select_dtypes('number').columns if 'valor_' in c or c == 'Dias_atraso']
    df[num_cols] = df[num_cols].fillna(0)
    df[df.select_dtypes('object').columns] = df.select_dtypes('object').fillna('Unknown')
    
    df = df.sort_values(['usuarioId', 'vencimentoData', 'criacaoData']).copy()
    df['Dias_atraso'] = df['Dias_atraso'].clip(lower=0)
    
    df['invoice_sequence'] = df.groupby('usuarioId').cumcount() + 1
    
    # === CALENDAR/SEASONALITY FEATURES ===
    df['day_of_week'] = df['vencimentoData'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_end'] = (df['vencimentoData'].dt.day >= 25).astype(int)
    df['month'] = df['vencimentoData'].dt.month
    df['is_first_invoice'] = (df['invoice_sequence'] == 1).astype(int)
    
    days = (now_utc - df['vencimentoData']).dt.days
    df['is_default'] = (df['pagamentoData'].isna() & (days > 90)).astype(int)
    df['days_since_due'] = days
    df['is_delayed'] = (df['Dias_atraso'] > 0).astype(int)
    
    g = df.groupby('usuarioId')
    df['current_streak'] = g['is_delayed'].transform(_streak)
    df['streak_group'] = (df['is_delayed'] != df['is_delayed'].shift()).cumsum()
    df['current_streak_days'] = g.apply(_streak_days, include_groups=False).reset_index(level=0, drop=True)
    
    df['diff_pago_devido'] = df['valor_pago_brl'] - df['valor_brl']
    df['val_penalty'] = df['diff_pago_devido'].clip(lower=0)
    df['val_delayed_amount'] = df['is_delayed'] * df['valor_brl']
    
    g = df.groupby('usuarioId')
    
    # === HISTORICAL FEATURES (shifted to avoid leakage) ===
    df['hist_count_delays'] = g['is_delayed'].shift(1).cumsum().fillna(0)
    df['hist_sum_val_delayed'] = g['val_delayed_amount'].shift(1).cumsum().fillna(0)
    df['hist_total_penalty'] = g['val_penalty'].shift(1).cumsum().fillna(0)
    df['hist_total_paid'] = g['valor_pago_brl'].shift(1).cumsum().fillna(0)
    df['hist_max_streak'] = g['current_streak'].shift(1).expanding().max().reset_index(level=0, drop=True).fillna(0)
    df['hist_max_delay_days'] = g['Dias_atraso'].shift(1).expanding().max().reset_index(level=0, drop=True).fillna(0)
    
    prev_invoice_count = (df['invoice_sequence'] - 1).clip(lower=1)
    df['hist_delay_rate'] = _safe_divide(df['hist_count_delays'], prev_invoice_count, 0.0)
    df['hist_avg_val_delayed'] = _safe_divide(df['hist_sum_val_delayed'], df['hist_count_delays'], 0.0)
    df['hist_avg_penalty'] = _safe_divide(df['hist_total_penalty'], prev_invoice_count, 0.0)
    
    # === RECENCY FEATURES (últimas 3 faturas - simplified) ===
    df['recent_delay_rate'] = g['is_delayed'].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
    df['recent_avg_delay_days'] = g['Dias_atraso'].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
    
    # === TREND: is recent behavior worse than historical? ===
    df['trend_delay_rate'] = (df['recent_delay_rate'] - df['hist_delay_rate']).clip(-1, 1)  # Simple difference, not ratio
    
    # === RISK SCORE COMPOSITE ===
    df['risk_score'] = (
        df['hist_delay_rate'] * 0.3 + 
        df['recent_delay_rate'] * 0.4 + 
        (df['current_streak'] / 5).clip(upper=1) * 0.3
    ).clip(0, 1)
    
    df = (df.sort_values(['usuarioId', 'vencimentoData'])
          .reset_index(drop=True)
          .groupby('usuarioId', group_keys=False)
          .apply(_filter_until_default)
          .reset_index(drop=True))
    
    first_default_idx = df[df['is_default'] == 1].groupby('usuarioId').head(1).index
    df['is_first_default'] = 0
    df.loc[first_default_idx, 'is_first_default'] = 1
    
    df['target_default_1'] = df.groupby('usuarioId')['is_default'].shift(-1).fillna(0)
    df = df.dropna(subset=['target_default_1'])
    
    cols = list(dict.fromkeys(c for c in config.columns.keep_cols_final + config.columns.static_cols if c in df.columns))
    
    # Final cleanup - replace any remaining inf/nan
    result_df = df[cols].copy()
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_cols] = result_df[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)
    
    return result_df.sort_values('vencimentoData').pipe(lambda x: x.to_csv(output_path, index=False, sep=';') or x)


if __name__ == "__main__":
    run_pipeline(config.paths.raw_data, config.paths.train_data)
