import pandas as pd
import numpy as np
import logging
from Configs.config import config


class Preprocessor:  
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("Preprocessor")
    
    def filter_category(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        category_col = config.columns.category_col
        category_filter = config.columns.category_filter
        if category_col in dataframe.columns:
            before = len(dataframe)
            dataframe = dataframe[dataframe[category_col].astype(str).str.lower().str.contains(category_filter)]
            after = len(dataframe)
            reduction_pct = (before - after) / before * 100 if before > 0 else 0
            self.logger.info(f"[Category Filter] Filtered by '{category_filter}': {before:,} -> {after:,} rows (removed {before - after:,}, {reduction_pct:.1f}%)")
        return dataframe

    def drop_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        columns_to_drop = [column for column in config.columns.drop_cols if column in dataframe.columns]
        dataframe = dataframe.drop(columns=columns_to_drop)
        self.logger.info(f"[Column Pruning] Removed {len(columns_to_drop)} columns: {columns_to_drop[:5]}{'...' if len(columns_to_drop) > 5 else ''}")
        return dataframe

    def process_dates(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        due_date_col = config.columns.due_date_col
        date_cols_processed = [col for col in config.columns.date_cols if col in dataframe.columns]
        for column in date_cols_processed:
            dataframe[column] = pd.to_datetime(dataframe[column], errors='coerce')
        
        dataframe[due_date_col] = pd.to_datetime(dataframe[due_date_col], errors='coerce', utc=True)
        now_utc = pd.Timestamp.now(tz='UTC')

        before = len(dataframe)
        dataframe = dataframe[dataframe[due_date_col] <= now_utc]
        after = len(dataframe)
        
        self.logger.info(f"[Date Processing] Parsed {len(date_cols_processed)} date columns, filtered future dates: {before:,} -> {after:,} rows")
        
        return dataframe.sort_values(due_date_col).reset_index(drop=True)

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = self.filter_category(dataframe)
        dataframe = self.drop_columns(dataframe)
        dataframe = self.process_dates(dataframe)
        return dataframe


class FeatureEngineer:
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("FeatureEngineer")
    
    def create_temporal_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.copy()
        
        due_date_col = config.columns.due_date_col
        dataframe['venc_dayofweek'] = dataframe[due_date_col].dt.dayofweek
        dataframe['venc_day'] = dataframe[due_date_col].dt.day
        dataframe['venc_month'] = dataframe[due_date_col].dt.month
        dataframe['venc_quarter'] = dataframe[due_date_col].dt.quarter
        dataframe['venc_is_weekend'] = (dataframe['venc_dayofweek'] >= config.temporal.weekend_start_day).astype(int)
        dataframe['venc_is_month_start'] = (dataframe['venc_day'] <= config.temporal.month_start_threshold).astype(int)
        dataframe['venc_is_month_end'] = (dataframe['venc_day'] >= config.temporal.month_end_threshold).astype(int)
     
        dataframe['venc_dayofweek_sin'] = np.sin(2 * np.pi * dataframe['venc_dayofweek'] / config.temporal.days_in_week)
        dataframe['venc_dayofweek_cos'] = np.cos(2 * np.pi * dataframe['venc_dayofweek'] / config.temporal.days_in_week)
        dataframe['venc_day_sin'] = np.sin(2 * np.pi * dataframe['venc_day'] / config.temporal.days_in_month)
        dataframe['venc_day_cos'] = np.cos(2 * np.pi * dataframe['venc_day'] / config.temporal.days_in_month)
        dataframe['venc_month_sin'] = np.sin(2 * np.pi * dataframe['venc_month'] / config.temporal.months_in_year)
        dataframe['venc_month_cos'] = np.cos(2 * np.pi * dataframe['venc_month'] / config.temporal.months_in_year)
        
        self.logger.info(f"[Temporal Features] Extracted 13 features: dayofweek, day, month, quarter, weekend, month_start/end, cyclical encodings (sin/cos)")
        return dataframe

    def create_history_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        user_col = config.columns.user_id_col
        contract_col = config.columns.contract_id_col
        order_col = config.columns.order_col
        delay_col = config.columns.delay_col
        due_date_col = config.columns.due_date_col
        
        dataframe = dataframe.sort_values([user_col, contract_col, order_col]).reset_index(drop=True)
        
        dataframe['hist_mean_delay'] = dataframe.groupby(user_col)[delay_col].transform(
            lambda values: values.expanding().mean().shift(1)
        ).fillna(0)
        
        dataframe['hist_std_delay'] = dataframe.groupby(user_col)[delay_col].transform(
            lambda values: values.expanding().std().shift(1)
        ).fillna(0)
        
        dataframe['hist_max_delay'] = dataframe.groupby(user_col)[delay_col].transform(
            lambda values: values.expanding().max().shift(1)
        ).fillna(0)
        
        dataframe['hist_default_rate'] = dataframe.groupby(user_col)[delay_col].transform(
            lambda values: (values >= config.data.delay_threshold_1).expanding().mean().shift(1)
        ).fillna(0)
        
        dataframe['hist_payment_count'] = dataframe.groupby(user_col).cumcount()
        dataframe['last_delay'] = dataframe.groupby(user_col)[delay_col].shift(1).fillna(0)
        dataframe['delay_trend'] = dataframe.groupby(user_col)[delay_col].diff().shift(1).fillna(0)
    
        dataframe['had_default'] = (dataframe[delay_col] >= config.data.delay_threshold_1)
        dataframe['default_date'] = dataframe[due_date_col].where(dataframe['had_default'])
        dataframe['last_default_date'] = dataframe.groupby(user_col)['default_date'].ffill().shift(1)
        dataframe['days_since_last_default'] = (dataframe[due_date_col] - dataframe['last_default_date']).dt.days
        dataframe['days_since_last_default'] = dataframe['days_since_last_default'].fillna(config.data.days_since_last_default_fill)
        dataframe['days_since_last_default'] = dataframe['days_since_last_default'].clip(upper=config.data.days_since_last_default_clip)
        dataframe = dataframe.drop(columns=['had_default', 'default_date', 'last_default_date']) 
        
        self.logger.info(f"[History Features] Extracted 8 features: hist_mean/std/max_delay, default_rate, payment_count, last_delay, delay_trend, days_since_last_default")
        return dataframe

    def create_sequence_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        user_col = config.columns.user_id_col
        contract_col = config.columns.contract_id_col
        order_col = config.columns.order_col
        delay_col = config.columns.delay_col
        due_date_col = config.columns.due_date_col
        
        dataframe = dataframe.sort_values([user_col, contract_col, order_col]).reset_index(drop=True)
        
        dataframe['seq_position'] = dataframe.groupby(user_col).cumcount()
        dataframe['seq_total'] = dataframe.groupby(user_col)[user_col].transform('count')
        dataframe['seq_position_norm'] = dataframe['seq_position'] / dataframe['seq_total'].clip(lower=1)
        
        dataframe['days_since_last_invoice'] = dataframe.groupby(user_col)[due_date_col].diff().dt.days.fillna(0)
        dataframe['days_since_last_invoice'] = dataframe['days_since_last_invoice'].clip(lower=0, upper=config.data.days_since_last_invoice_clip)
        
        dataframe['is_first_invoice'] = (dataframe['seq_position'] == 0).astype(int)
        
        dataframe['rolling_mean_delay_3'] = dataframe.groupby(user_col)[delay_col].transform(
            lambda values: values.rolling(3, min_periods=1).mean().shift(1)
        ).fillna(0)
        
        dataframe['rolling_max_delay_3'] = dataframe.groupby(user_col)[delay_col].transform(
            lambda values: values.rolling(3, min_periods=1).max().shift(1)
        ).fillna(0)
        
        dataframe['is_improving'] = (dataframe['delay_trend'] < 0).astype(int) if 'delay_trend' in dataframe.columns else 0
        
        dataframe['parcela_position'] = dataframe.groupby([user_col, contract_col]).cumcount()
        dataframe['parcela_total'] = dataframe.groupby([user_col, contract_col])[order_col].transform('count')
        dataframe['parcela_position_norm'] = dataframe['parcela_position'] / dataframe['parcela_total'].clip(lower=1)
        
        dataframe['n_contratos_anteriores'] = dataframe.groupby(user_col)[contract_col].transform(
            lambda values: values.ne(values.shift()).cumsum() - 1
        ).clip(lower=0)
        
        dataframe['is_first_contract'] = (dataframe['n_contratos_anteriores'] == 0).astype(int)
        
        dataframe['contract_mean_delay'] = dataframe.groupby([user_col, contract_col])[delay_col].transform(
            lambda values: values.expanding().mean().shift(1)
        ).fillna(0)
        
        self.logger.info(f"[Sequence Features] Extracted 12 features: seq_position, days_since_last_invoice, rolling_stats, parcela_position, contract_features")
        return dataframe

    def create_value_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.copy()
        user_col = config.columns.user_id_col
        value_col = config.columns.value_col
        
        if value_col in dataframe.columns:
            dataframe['hist_mean_value'] = dataframe.groupby(user_col)[value_col].transform(
                lambda values: values.expanding().mean().shift(1)
            ).fillna(dataframe[value_col])
            
            dataframe['value_ratio'] = dataframe[value_col] / dataframe['hist_mean_value'].clip(lower=config.data.hist_mean_value_clip_min)
            dataframe['value_ratio'] = dataframe['value_ratio'].clip(lower=config.data.value_ratio_clip_min, upper=config.data.value_ratio_clip_max)
            
            dataframe = dataframe.drop(columns=['hist_mean_value'])
            
            self.logger.info(f"[Value Features] Extracted value_ratio feature (current/historical mean, clipped to [{config.data.value_ratio_clip_min}, {config.data.value_ratio_clip_max}])")
        
        return dataframe

    def create_targets(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        sort_cols = config.columns.sort_cols
        group_cols = config.columns.group_cols
        delay_col = config.columns.delay_col
        target_cols = config.columns.target_cols
        
        dataframe = dataframe.sort_values(sort_cols).reset_index(drop=True)
        thresholds = [config.data.delay_threshold_1, config.data.delay_threshold_2, config.data.delay_threshold_3]
        for threshold, column in zip(thresholds, target_cols):
            dataframe[column] = dataframe.groupby(group_cols)[delay_col].shift(-1).fillna(0).apply(lambda value: int(value >= threshold))
        return dataframe

    def remove_last_invoice(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        sort_cols = config.columns.sort_cols
        group_cols = config.columns.group_cols
        
        dataframe = dataframe.sort_values(sort_cols).reset_index(drop=True)
        dataframe['is_last_invoice'] = dataframe.groupby(group_cols).cumcount(ascending=False) == 0
        before = len(dataframe)
        dataframe = dataframe[~dataframe['is_last_invoice']].copy()
        after = len(dataframe)
        reduction_pct = (before - after) / before * 100 if before > 0 else 0
        self.logger.info(f"[Target Leakage Prevention] Removed last invoice per contract: {before:,} -> {after:,} rows ({reduction_pct:.1f}% removed)")
        return dataframe.drop(columns=['is_last_invoice'])

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = self.create_temporal_features(dataframe)
        dataframe = self.create_history_features(dataframe)
        dataframe = self.create_sequence_features(dataframe)
        dataframe = self.create_value_features(dataframe)
        dataframe = self.create_targets(dataframe)
        dataframe = self.remove_last_invoice(dataframe)
        return dataframe


class DataPipeline:
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("Pipeline")
        self.preprocessor = Preprocessor(self.logger)
        self.feature_engineer = FeatureEngineer(self.logger)
    
    def _load_data(self, input_path: str) -> pd.DataFrame:
        if input_path.endswith('.parquet'):
            dataframe = pd.read_parquet(input_path)
        else:
            dataframe = pd.read_csv(input_path, low_memory=False)
        self.logger.info(f"[Data Loading] Loaded {len(dataframe):,} rows from {input_path.split('/')[-1]}")
        return dataframe
    
    def _finalize_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        target_cols = config.columns.target_cols
        base_columns = [column for column in dataframe.columns if not column.startswith('target_')]
        columns = base_columns + [column for column in target_cols if column in dataframe.columns]

        result_dataframe = dataframe[columns].copy()
        numeric_columns = result_dataframe.select_dtypes(include=[np.number]).columns
        result_dataframe[numeric_columns] = result_dataframe[numeric_columns].replace([np.inf, -np.inf], 0).fillna(0)
        return result_dataframe
    
    def _save_data(self, dataframe: pd.DataFrame, output_path: str) -> None:
        dataframe = dataframe.sort_values(config.columns.sort_cols)
        dataframe.to_parquet(output_path, index=False)
        
        numeric_features = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = [col for col in numeric_features if col.startswith('target_')]
        feature_cols = [col for col in numeric_features if not col.startswith('target_')]
        
        self.logger.info(f"[Pipeline Complete] Final dataset: {len(dataframe):,} rows, {len(feature_cols)} features, {len(target_cols)} targets")
        self.logger.info(f"[Output] Saved to {output_path}")

    def run(self, input_path: str, output_path: str) -> pd.DataFrame:
        self.logger.info(">>> FEATURE ENGINEERING PIPELINE")
        
        dataframe = self._load_data(input_path)
        
        self.logger.info("  > Data Preprocessing")
        dataframe = self.preprocessor.run(dataframe)
        
        self.logger.info("  > Feature Extraction & Target Engineering")
        dataframe = self.feature_engineer.run(dataframe)
        
        self.logger.info("  > Finalization")
        dataframe = self._finalize_data(dataframe)
        self._save_data(dataframe, output_path)
        
        return dataframe


if __name__ == "__main__":
    pipeline = DataPipeline()
    raw_path = config.paths.raw_data
    train_path = config.paths.train_data
    pipeline.run(raw_path, train_path)
