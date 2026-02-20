import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from core.data import FeatureEngineer
from core.logger import Logger


class InferenceEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        target_scaler,
        feature_scalers: dict,
        config,
        logger: Logger = None,
        device: torch.device = None,
        categorical_maps: dict = None,
    ):
        self.model            = model
        self.target_scaler    = target_scaler
        self.feature_scalers  = feature_scalers or {}
        self.config           = config
        self.logger           = logger or Logger(log_dir=None, name="inference", level="INFO")
        self.device           = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categorical_maps = categorical_maps or {}
        
        self.logger.section("[Inference]")
        self.logger.subsection(f"Using device: {self.device}")

        self.model.to(self.device)
        self.model.eval()

        self.categorical_columns = list(self.config.columns.cat_cols)
        self.continuous_columns  = list(self.config.columns.cont_cols)

    def _parse_dates(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.logger.section("[Date Parsing]")

        date_cols = [col for col in self.config.columns.date_cols if col in dataframe.columns]
        for col in date_cols:
            dataframe[col] = pd.to_datetime(dataframe[col], errors="coerce", utc=True)
        
        self.logger.subsection(f"Parsed date columns: {', '.join(date_cols)} \n")
        return dataframe

    def _apply_feature_engineering(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.logger.section("[Feature Engineering]")
        
        engineer = FeatureEngineer(self.config, self.logger)
        dataframe = engineer.create_temporal_features(dataframe)
        dataframe = engineer.create_history_features(dataframe)
        dataframe = engineer.create_sequence_features(dataframe)
        dataframe = engineer.create_value_features(dataframe)

        delay_col         = self.config.columns.delay_col
        payment_date_col  = self.config.columns.payment_date_col
        delay_clipped_col = self.config.columns.delay_clipped_col
        delay_known_col   = self.config.columns.delay_is_known_col
        target_col        = self.config.columns.target_col_name

        known_mask = dataframe[payment_date_col].notna() & dataframe[delay_col].notna()
        
        dataframe[delay_clipped_col] = dataframe[delay_col].clip(lower=0).fillna(0)
        dataframe[delay_known_col]   = known_mask.astype(int)
        dataframe[target_col]        = dataframe[delay_clipped_col]

        self.logger.subsection(f"Created features: {delay_clipped_col}, {delay_known_col}, {target_col} \n")
        return dataframe

    def _normalize_continuous(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.logger.section("[Continuous Feature Normalization]")
        
        no_scale = set(self.config.columns.no_scale_cols)
        for col in self.continuous_columns:
            dataframe[col] = dataframe[col].replace([np.inf, -np.inf], 0).fillna(0.0)
            if col in no_scale:
                dataframe[col] = dataframe[col].astype(float)
                continue
            
            scaler = self.feature_scalers.get(col)
            dataframe[col] = scaler.transform(dataframe[[col]].astype(float).values)

        self.logger.subsection(f"Normalized continuous columns: {', '.join(self.continuous_columns)} \n")
        return dataframe

    def _encode_categorical(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.logger.section("[Categorical Feature Encoding]")   
        missing_map_cols = []
        for col in self.categorical_columns:
            if col not in dataframe.columns:
                dataframe[col] = 0
                continue

            if pd.api.types.is_numeric_dtype(dataframe[col]):
                dataframe[col] = dataframe[col].fillna(0).astype(int)
                continue

            mapping = self.categorical_maps.get(col)
            if mapping is not None:
                dataframe[col] = dataframe[col].map(mapping).fillna(0).astype(int)
                continue

            missing_map_cols.append(col)

            codes, _ = pd.factorize(dataframe[col].astype(str), sort=True)
            dataframe[col] = (codes + 1).astype(int)

        if not self.categorical_maps:
            self.logger.warning("No categorical maps provided. Using local factorization for non-numeric categorical columns.")
        elif missing_map_cols:
            missing_cols = ", ".join(sorted(set(missing_map_cols)))
            self.logger.warning(f"Missing categorical maps for columns: {missing_cols}. Using local factorization for these columns.")


        self.logger.subsection(f"Encoded categorical columns: {', '.join(self.categorical_columns)} \n")
        return dataframe

    def _prepare_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.logger.section("[Data Preparation]")
        
        dataframe = dataframe.copy()

        for col in dataframe.select_dtypes(include=["object"]).columns:
            try:
                dataframe[col] = pd.to_numeric(dataframe[col])
            except (ValueError, TypeError):
                pass
        
        dataframe = self._parse_dates(dataframe)

        category_col = self.config.columns.category_col
        category_filter = self.config.columns.category_filter
        if category_col in dataframe.columns:
            dataframe = dataframe[
                dataframe[category_col].astype(str).str.lower().str.contains(category_filter)
            ].copy()

        paid_value_col = self.config.columns.paid_value_col
        if paid_value_col in dataframe.columns:
            dataframe[paid_value_col] = dataframe[paid_value_col].fillna(0)

        dataframe = self._apply_feature_engineering(dataframe)
        dataframe = self._normalize_continuous(dataframe)
        dataframe = self._encode_categorical(dataframe)

        self.logger.subsection(f"Final dataframe shape after preparation: {dataframe.shape} \n")
        return dataframe

    def _build_sequences(self, dataframe: pd.DataFrame) -> tuple:
        self.logger.section("[Sequence Building]")

        group_cols       = self.config.columns.group_cols
        sort_cols        = self.config.columns.sort_cols
        payment_date_col = self.config.columns.payment_date_col
        delay_col        = self.config.columns.delay_col

        sequences = []
        meta_rows = []

        for group_id, user_df in dataframe.groupby(group_cols):
            user_df   = user_df.sort_values(sort_cols).copy()
            paid_mask = user_df[payment_date_col].notna() & user_df[delay_col].notna()
            paid_df   = user_df[paid_mask].copy()
            open_df   = user_df[~paid_mask].copy()

            if open_df.empty:
                continue

            open_row = open_df.sort_values(sort_cols).iloc[[0]].copy()
            seq_df   = pd.concat([paid_df, open_row], axis=0)

            if len(seq_df) > self.config.architecture.max_seq_len:
                seq_df = seq_df.iloc[-self.config.architecture.max_seq_len :]

            sequences.append(seq_df)
            meta_rows.append(open_row.iloc[0])

        self.logger.subsection(f"Built {len(sequences)} sequences for inference \n")
        return sequences, meta_rows

    def _mask_last_step(self, continuous_tensor: torch.Tensor) -> torch.Tensor:
        mask_cols = []
        for col in [self.config.columns.delay_clipped_col, self.config.columns.delay_is_known_col]:
            if col in self.continuous_columns:
                mask_cols.append(self.continuous_columns.index(col))

        if not mask_cols:
            return continuous_tensor

        last_t = continuous_tensor.shape[0] - 1
        if last_t >= 0:
            continuous_tensor[last_t, mask_cols] = 0.0

        return continuous_tensor

    def _decode_predictions(self, preds: np.ndarray) -> np.ndarray:
        if self.target_scaler is None:
            return np.clip(preds, 0, None)

        den = self.target_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(-1)
        if self.config.target.use_log1p_transform:
            den = np.expm1(den)
        den = np.clip(den, 0, None)
        return den

    def predict(self, dataframe: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
        self.logger.section("[Prediction]")
        dataframe            = self._prepare_dataframe(dataframe)
        sequences, meta_rows = self._build_sequences(dataframe)

        preds_all = []
        for start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[start : start + batch_size]

            categorical_list = []
            continuous_list  = []
            lengths          = []

            for seq_df in batch_seqs:
                cat  = torch.as_tensor(seq_df[self.categorical_columns].values, dtype=torch.long)
                cont = torch.as_tensor(seq_df[self.continuous_columns].values,  dtype=torch.float32)
                cont = self._mask_last_step(cont)

                categorical_list.append(cat)
                continuous_list.append(cont)
                lengths.append(len(seq_df))

            categorical_padded = pad_sequence(
                sequences     = categorical_list,
                batch_first   = True,
                padding_value = self.config.architecture.categorical_padding_value,
            )
            
            continuous_padded = pad_sequence(
                sequences     = continuous_list,
                batch_first   = True,
                padding_value = self.config.architecture.continuous_padding_value,
            )
            
            lengths_tensor = torch.tensor(lengths, dtype=torch.long)

            with torch.no_grad():
                preds = self.model(
                    categorical_padded.to(self.device),
                    continuous_padded.to(self.device),
                    lengths_tensor.to(self.device),
                )

            preds_all.append(preds.detach().cpu().numpy().reshape(-1))

        preds_all = np.concatenate(preds_all, axis=0)
        den_preds = self._decode_predictions(preds_all)

        output_rows = []
        for meta, pred, seq in zip(meta_rows, den_preds, sequences):
            output_rows.append({
                self.config.columns.user_id_col: meta[self.config.columns.user_id_col] if self.config.columns.user_id_col in meta.index else None,
                self.config.columns.contract_id_col: meta[self.config.columns.contract_id_col] if self.config.columns.contract_id_col in meta.index else None,
                self.config.columns.order_col: meta[self.config.columns.order_col] if self.config.columns.order_col in meta.index else None,
                self.config.columns.creation_date_col: meta[self.config.columns.creation_date_col] if self.config.columns.creation_date_col in meta.index else None,
                self.config.columns.due_date_col: meta[self.config.columns.due_date_col] if self.config.columns.due_date_col in meta.index else None,
                self.config.columns.payment_date_col: meta[self.config.columns.payment_date_col] if self.config.columns.payment_date_col in meta.index else None,
                "predicted_days_to_payment": float(pred),
                "sequence_length": int(len(seq)),
            })

        self.logger.subsection(f"Generated predictions for {len(output_rows)} contracts \n")
        return pd.DataFrame(output_rows)
