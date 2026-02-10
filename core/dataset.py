import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import platform


class Augmentation:
    def __init__(self, augmentation_params):
        self.params = augmentation_params

    def temporal_cutout(
        self,
        categorical_features: torch.Tensor,
        continuous_features: torch.Tensor,
        probability: float = None,
        excluded_cont_indices: list = None,
    ) -> tuple:
      
        if torch.rand(1) > probability:
            return categorical_features, continuous_features

        sequence_length = categorical_features.shape[0]
        if sequence_length <= 1:
            return categorical_features, continuous_features

        available_indices = list(range(sequence_length - 1))
        if len(available_indices) == 0:
            return categorical_features, continuous_features
        
        num_cutout = max(1, int((sequence_length - 1) * self.params.temporal_cutout_ratio))
        num_cutout = min(num_cutout, len(available_indices))
        cutout_indices = torch.tensor(available_indices, dtype=torch.long)[torch.randperm(len(available_indices))[:num_cutout]]

        categorical_augmented = categorical_features.clone()
        continuous_augmented  = continuous_features.clone()

        categorical_augmented[cutout_indices] = 0
        continuous_augmented[cutout_indices] = 0.0
        if excluded_cont_indices:
            continuous_augmented[cutout_indices][:, excluded_cont_indices] = continuous_features[cutout_indices][:, excluded_cont_indices]

        return categorical_augmented, continuous_augmented

    def feature_dropout(
        self,
        categorical_features: torch.Tensor,
        continuous_features: torch.Tensor,
        probability: float = None,
        excluded_cont_indices: list = None,
    ) -> tuple:

        if torch.rand(1) > probability:
            return categorical_features, continuous_features

        categorical_augmented = categorical_features.clone()
        continuous_augmented = continuous_features.clone()
        
        last_t = categorical_features.shape[0] - 1
        if last_t < 0:
            return categorical_augmented, continuous_augmented
        
        categorical_last = categorical_features[last_t].clone()
        continuous_last = continuous_features[last_t].clone()

        if categorical_features.shape[1] > 0:
            num_categorical_drop = max(1, int(categorical_features.shape[1] * self.params.feature_dropout_ratio))
            categorical_drop_indices = torch.randperm(categorical_features.shape[1])[:num_categorical_drop]
            categorical_augmented[:, categorical_drop_indices] = 0

        if continuous_features.shape[1] > 0:
            excluded = set(excluded_cont_indices or [])
            available_indices = [i for i in range(continuous_features.shape[1]) if i not in excluded]
            if len(available_indices) > 0:
                num_continuous_drop = max(1, int(len(available_indices) * self.params.feature_dropout_ratio))
                perm = torch.randperm(len(available_indices))
                continuous_drop_indices = torch.tensor(available_indices, device=continuous_features.device)[perm[:num_continuous_drop]]
                continuous_augmented[:, continuous_drop_indices] = 0.0
        
        categorical_augmented[last_t] = categorical_last
        continuous_augmented[last_t] = continuous_last

        return categorical_augmented, continuous_augmented

    def gaussian_noise(
        self,
        continuous_features: torch.Tensor,
        probability: float = None,
        standard_deviation: float = None,
        excluded_cont_indices: list = None,
    ) -> torch.Tensor:
        
        if torch.rand(1) > probability:
            return continuous_features

        last_t = continuous_features.shape[0] - 1
        continuous_augmented = continuous_features.clone()
        
        if last_t >= 0:
            noise = torch.randn(last_t, continuous_features.shape[1], device=continuous_features.device) * standard_deviation
            if excluded_cont_indices:
                noise[:, excluded_cont_indices] = 0.0
            continuous_augmented[:last_t] = continuous_features[:last_t] + noise
        
        return continuous_augmented

    def time_warp(self, categorical_features: torch.Tensor, continuous_features: torch.Tensor, probability: float = None) -> tuple:
        if torch.rand(1) > probability:
            return categorical_features, continuous_features

        sequence_length = categorical_features.shape[0]
        if sequence_length <= 2:
            return categorical_features, continuous_features

        available_indices = list(range(sequence_length - 1))
        
        if torch.rand(1) > 0.5:
            if len(available_indices) == 0:
                return categorical_features, continuous_features
            duplicate_index = available_indices[torch.randint(0, len(available_indices), (1,))[0]]
            categorical_augmented = torch.cat([categorical_features[:duplicate_index+1], categorical_features[duplicate_index:]], dim=0)
            continuous_augmented  = torch.cat([continuous_features[:duplicate_index+1],  continuous_features[duplicate_index:]],  dim=0)
        else:
            if len(available_indices) == 0:
                return categorical_features, continuous_features
            remove_index = available_indices[torch.randint(0, len(available_indices), (1,))[0]]
            categorical_augmented = torch.cat([categorical_features[:remove_index], categorical_features[remove_index+1:]], dim=0)
            continuous_augmented  = torch.cat([continuous_features[:remove_index],  continuous_features[remove_index+1:]],  dim=0)

        return categorical_augmented, continuous_augmented


class SequentialDataset(Dataset):
    def __init__(
        self,
        categorical_data,
        continuous_data,
        targets,
        indices,
        logger,
        augment=False,
        mask_last_cont=None,
        last_known_idx=None,
        protected_cont_indices=None,
        cfg=None,
    ):
        self.categorical_data = torch.as_tensor(categorical_data.copy(), dtype=torch.long)
        self.continuous_data = torch.as_tensor(continuous_data.copy(), dtype=torch.float32)
        self.targets = torch.as_tensor(targets.copy(), dtype=torch.float32)
        self.indices = indices
        self.config = cfg
        self.augment = (self.config.architecture.use_augmentation if (self.config is not None and self.config.architecture is not None) else False) and augment
        self.augment_probability = self.config.architecture.augment_prob if (self.config is not None and self.config.architecture is not None) else 0.0
        self.augmenter = Augmentation(self.config.augmentation) if (self.config is not None and self.config.augmentation is not None) else None
        self.logger = logger
        self.mask_last_cont = mask_last_cont or []
        self.last_known_idx = last_known_idx
        self.protected_cont_indices = protected_cont_indices or []

        self.logger.section("Data Augmentation Configuration")
        if self.augment:
            self.logger.info(f"[Augmentation] Status: ENABLED")
            self.logger.info(f"[Augmentation] Probability: {self.augment_probability:.1%}")
            self.logger.info(f"[Augmentation] Methods: Temporal Cutout, Feature Dropout, Gaussian Noise")

    def __len__(self):
        return len(self.indices)
    
    def mask_target(self, continuous_features):
        last_t = continuous_features.shape[0] - 1
        if last_t >= 0:
            for j in self.mask_last_cont:
                continuous_features[last_t, j] = 0.0

            if self.last_known_idx is not None:
                continuous_features[last_t, self.last_known_idx] = 0.0  
        
        return continuous_features

    def __getitem__(self, index):
        start_index, end_index, target_index = self.indices[index]
        
        categorical_features = self.categorical_data[start_index:end_index].clone()
        continuous_features  = self.continuous_data[start_index:end_index].clone()
        target = self.targets[target_index]
        
        self.mask_target(continuous_features)

        if self.augment and self.training_mode and self.augmenter is not None:
            categorical_features, continuous_features = self.augmenter.temporal_cutout(
                categorical_features,
                continuous_features,
                probability=self.augment_probability,
                excluded_cont_indices=self.protected_cont_indices,
            )
            
            categorical_features, continuous_features = self.augmenter.feature_dropout(
                categorical_features,
                continuous_features,
                probability=self.augment_probability,
                excluded_cont_indices=self.protected_cont_indices,
            )
            
            continuous_features = self.augmenter.gaussian_noise(
                continuous_features,
                probability=self.augment_probability,
                standard_deviation=(self.config.augmentation.gaussian_noise_std),
                excluded_cont_indices=self.protected_cont_indices,
            )
            
            categorical_features, continuous_features = self.augmenter.time_warp(
                categorical_features,
                continuous_features,
                probability=self.augment_probability,
            )
        
        self.mask_target(continuous_features)

        length = categorical_features.shape[0]
        return categorical_features, continuous_features, target, length
    
    @property
    def training_mode(self):
        return self.augment


class DatasetLoader:
    def __init__(
        self,
        cfg,
        logger,
        dataframe,
    ):
        self.logger = logger
        self.config = cfg
        
        self.dataframe = dataframe.copy()

        self.categorical_columns = list(self.config.columns.cat_cols)
        self.continuous_columns  = list(self.config.columns.cont_cols)
        self.target_columns      = list(self.config.columns.target_cols)

        self.continuous_scalers   = {}
        self.target_scalers       = {}
        self.embedding_dimensions = []
        self.categorical_maps     = {}


    def load_data(self, dataframe):
        self.logger.info(f"[Data Loading] Loaded {len(dataframe):,} rows, {len(dataframe.columns)} columns")
        
        group_col = self.config.columns.group_cols[0]
        unique_users = dataframe[group_col].unique()
        num_users = len(unique_users)

        if self.config.load.user_sample_count is not None:
            sample_size = min(self.config.load.user_sample_count, num_users)
        else:
            sample_size = int(num_users * self.config.load.load_sample_fraction)

        rng = np.random.default_rng(self.config.load.random_state)
        sampled_users = rng.choice(unique_users, size=sample_size, replace=False)
        dataframe = dataframe[dataframe[group_col].isin(sampled_users)]
        
        self.logger.info(f"[Data Loading] Sampled {sample_size:,} users (out of {num_users:,}), resulting in {len(dataframe):,} rows\n")
        return dataframe


    def clean_data(self, dataframe):
        payment_date = self.config.columns.due_date_col
        user_col = self.config.columns.user_id_col
        
        bad_users = dataframe.loc[dataframe[payment_date].isna(), user_col].unique()
        if len(bad_users) > 0:
            self.logger.warning(f"[Data Cleaning] Removing {len(bad_users):,} users with missing payment dates")
            dataframe = dataframe[~dataframe[user_col].isin(bad_users)]
    
        self.categorical_columns = [column for column in self.categorical_columns if column in dataframe.columns]
        self.continuous_columns  = [column for column in self.continuous_columns  if column in dataframe.columns]
        
        self.logger.subsection("Feature Configuration")
        self.logger.info(f"[Clean] Categorical: {len(self.categorical_columns)} features")
        self.logger.info(f"[Clean] Continuous: {len(self.continuous_columns)} features")
        self.logger.info(f"[Clean] Targets: {len(self.target_columns)} ({', '.join(self.target_columns)}) \n")
        
        dataframe = dataframe.dropna(subset=self.target_columns)
        return dataframe


    def clip_target(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        threshold = self.config.target.target_threshold
        num_clipped = (dataframe[self.target_columns] > threshold).sum().sum()
        dataframe[self.target_columns] = dataframe[self.target_columns].clip(upper=threshold)
        self.logger.info(f"[Clean] Clipped target values to threshold={threshold}: clipped {int(num_clipped):,} values\n")
        return dataframe


    def encode_categorical(self, dataframe):
        for column in self.categorical_columns:
            dataframe[column] = dataframe[column].astype(str)

        self.embedding_dimensions = []
        for column in self.categorical_columns:            
            label_encoder = LabelEncoder()
            dataframe[column] = label_encoder.fit_transform(dataframe[column]) + 1
            cardinality = len(label_encoder.classes_)
            self.embedding_dimensions.append(cardinality)
            self.categorical_maps[column] = {str(cls): int(idx) + 1 for idx, cls in enumerate(label_encoder.classes_)}
            self.logger.info(f"[Categorical Encoding] Column {column:<25} | cardinality={cardinality:6d}")

        self.logger.info(f"[Categorical Encoding] Encoded {len(self.categorical_columns)} categorical features \n")
        return dataframe
    

    def split_users(self, unique_users):
        user_col       = self.config.columns.user_id_col
        user_dataframe = pd.DataFrame({user_col: unique_users})
        
        test_size       = self.config.split.test_size
        validation_size = self.config.split.val_size
        group_shuffle_split_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state = self.config.load.random_state)
        train_validation_indices, test_indices = next(group_shuffle_split_test.split(user_dataframe, groups=user_dataframe[user_col]))
        
        train_validation_users = user_dataframe.iloc[train_validation_indices][user_col].values
        test_users = user_dataframe.iloc[test_indices][user_col].values
        
        validation_size_adjusted = validation_size / (1 - test_size)
        group_shuffle_split_validation = GroupShuffleSplit(n_splits=1, test_size=validation_size_adjusted, random_state = self.config.load.random_state)
        train_indices, validation_indices = next(group_shuffle_split_validation.split(pd.DataFrame({user_col: train_validation_users}), groups=train_validation_users))
        
        train_users      = set(train_validation_users[train_indices])
        validation_users = set(train_validation_users[validation_indices])
        test_users       = set(test_users)
        
        total_users = len(train_users) + len(validation_users) + len(test_users)
        self.logger.info(f"[Split] Train: {len(train_users):,} users ({len(train_users)/total_users:.1%})")
        self.logger.info(f"[Split] Validation: {len(validation_users):,} users ({len(validation_users)/total_users:.1%})")
        self.logger.info(f"[Split] Test: {len(test_users):,} users ({len(test_users)/total_users:.1%}) \n")
        
        return train_users, validation_users, test_users
    

    def split_dataframes(self, dataframe, train_users, validation_users, test_users):
        user_col  = self.config.columns.user_id_col
     
        train_dataframe      = dataframe[dataframe[user_col].isin(train_users)].copy()
        validation_dataframe = dataframe[dataframe[user_col].isin(validation_users)].copy()
        test_dataframe       = dataframe[dataframe[user_col].isin(test_users)].copy()

        self.logger.info(f"[Dataframe Split] Training set: {train_dataframe.shape}")
        self.logger.info(f"[Dataframe Split] Validation set: {validation_dataframe.shape}")
        self.logger.info(f"[Dataframe Split] Test set: {test_dataframe.shape} \n")
        
        train_target_avg = train_dataframe[self.target_columns[0]].mean()
        train_target_std = train_dataframe[self.target_columns[0]].std()
        train_target_p90 = train_dataframe[self.target_columns[0]].quantile(0.9)
        self.logger.info(f"[Dataframe Split] Training target '{self.target_columns[0]}' stats: mean={train_target_avg:.4f}, std={train_target_std:.4f}, 90th percentile={train_target_p90:.4f}")

        validation_target_avg = validation_dataframe[self.target_columns[0]].mean()
        validation_target_std = validation_dataframe[self.target_columns[0]].std()
        validation_target_p90 = validation_dataframe[self.target_columns[0]].quantile(0.9)
        self.logger.info(f"[Dataframe Split] Validation target '{self.target_columns[0]}' stats: mean={validation_target_avg:.4f}, std={validation_target_std:.4f}, 90th percentile={validation_target_p90:.4f}")

        test_target_avg = test_dataframe[self.target_columns[0]].mean()
        test_target_std = test_dataframe[self.target_columns[0]].std()
        test_target_p90 = test_dataframe[self.target_columns[0]].quantile(0.9)
        self.logger.info(f"[Dataframe Split] Test target '{self.target_columns[0]}' stats: mean={test_target_avg:.4f}, std={test_target_std:.4f}, 90th percentile={test_target_p90:.4f} \n")

        return train_dataframe, validation_dataframe, test_dataframe
    

    def normalize_continuous_features(self, train_dataframe, validation_dataframe, test_dataframe):
        no_scale = set(self.config.columns.no_scale_cols)
        for col in self.continuous_columns:
            if col in no_scale:
                train_dataframe[col]      = train_dataframe[col].astype(float)
                validation_dataframe[col] = validation_dataframe[col].astype(float)
                test_dataframe[col]       = test_dataframe[col].astype(float)
                continue
            
            mean_before = train_dataframe[col].mean()
            std_before  = train_dataframe[col].std()
        
            scaler = StandardScaler()
            train_values = train_dataframe[[col]].values
            scaler.fit(train_values)
            train_dataframe[col]         = scaler.transform(train_values)
            validation_dataframe[col]    = scaler.transform(validation_dataframe[[col]].values)
            test_dataframe[col]          = scaler.transform(test_dataframe[[col]].values)
            self.continuous_scalers[col] = scaler
            self.logger.info(
                f"Feature {col:<25} | mean before={mean_before:10.4f} | std before={std_before:10.4f} | mean after={train_dataframe[col].mean():10.4f} | std after={train_dataframe[col].std():10.4f}"
            )
    
        self.logger.info(f"[Continuous Normalization] Normalized {len(self.continuous_columns)} continuous features \n")
        return train_dataframe, validation_dataframe, test_dataframe


    def normalize_targets(self, train_dataframe, validation_dataframe, test_dataframe):
        for target_col in self.target_columns:
            mean_before = train_dataframe[target_col].mean()
            std_before = train_dataframe[target_col].std()
            scaler = StandardScaler()
            
            if self.config.target.use_log1p_transform:
                train_targets = np.log1p(np.maximum(train_dataframe[[target_col]].values, self.config.target.clip_target_min))
                scaler.fit(train_targets)
                train_dataframe[target_col] = scaler.transform(train_targets)
                validation_dataframe[target_col] = scaler.transform(np.log1p(np.maximum(validation_dataframe[[target_col]].values, self.config.target.clip_target_min)))
                test_dataframe[target_col] = scaler.transform(np.log1p(np.maximum(test_dataframe[[target_col]].values, self.config.target.clip_target_min)))
            else:
                train_targets = np.maximum(train_dataframe[[target_col]].values, self.config.target.clip_target_min)
                scaler.fit(train_targets)
                train_dataframe[target_col] = scaler.transform(train_targets)
                validation_dataframe[target_col] = scaler.transform(np.maximum(validation_dataframe[[target_col]].values, self.config.target.clip_target_min))
                test_dataframe[target_col] = scaler.transform(np.maximum(test_dataframe[[target_col]].values, self.config.target.clip_target_min))
            
            self.target_scalers[target_col] = scaler
            mean_after = train_dataframe[target_col].mean()
            std_after = train_dataframe[target_col].std()
            self.logger.info(
                f"[Target Normalization] Target {target_col:<30} | mean before={mean_before:8.4f} | std before={std_before:8.4f} | mean after={mean_after:8.4f} | std after={std_after:8.4f}"
            )

        self.logger.info(f"[Target Normalization] Normalized {len(self.target_columns)} target features \n")
        return train_dataframe, validation_dataframe, test_dataframe
        

    def create_expanding_indices(self, dataframe, group_column):
        indices = []
        dataframe_reset = dataframe.reset_index(drop=True)
        group_offsets = dataframe_reset.groupby(group_column).indices

        min_start = self.config.architecture.min_seq_len - 1

        for _, group_indices in group_offsets.items():
            group_indices = np.sort(group_indices)
            num_invoices = len(group_indices)

            if num_invoices < self.config.architecture.min_seq_len:
                continue
            
            group_indices_set = set(group_indices)
            first_idx = group_indices[0]
            last_idx  = group_indices[-1]
            expected_contiguous = set(range(first_idx, last_idx + 1))
            
            if group_indices_set != expected_contiguous:
                self.logger.warning(
                    f"[Index Creation] Group has non-contiguous indices. "
                    f"First: {first_idx}, Last: {last_idx}, "
                    f"Missing: {sorted(expected_contiguous - group_indices_set)[:10]}"
                )
            
            for invoice_index in range(min_start, num_invoices):
                target_index = group_indices[invoice_index]
                sequence_end = target_index + 1
          
                sequence_start_candidate = sequence_end - self.config.architecture.max_seq_len
                sequence_start = max(sequence_start_candidate, group_indices[0])
            
                if sequence_start >= sequence_end:
                    self.logger.warning(
                        f"[Index Creation] Invalid sequence bounds: start={sequence_start} >= end={sequence_end}, "
                        f"target_idx={target_index}, max_seq_len={self.config.architecture.max_seq_len}. Skipping."
                    )
                    continue
                
                if target_index < sequence_start or target_index >= sequence_end:
                    self.logger.warning(
                        f"[Index Creation] Target index {target_index} outside sequence bounds "
                        f"[{sequence_start}, {sequence_end}). This may indicate a problem. Skipping."
                    )
                    continue
                
                indices.append((sequence_start, sequence_end, target_index))

        return indices


    def create_indices(self, train_dataframe, validation_dataframe, test_dataframe):
        group_cols = self.config.columns.group_cols
        sort_cols  = self.config.columns.sort_cols
    
        train_dataframe      = train_dataframe.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        validation_dataframe = validation_dataframe.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        test_dataframe       = test_dataframe.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    
        train_indices      = self.create_expanding_indices(train_dataframe, group_cols)
        validation_indices = self.create_expanding_indices(validation_dataframe, group_cols)
        test_indices       = self.create_expanding_indices(test_dataframe, group_cols)
    
        self.logger.info(f"[Indices] Created {len(train_indices):,} training sequences")
        self.logger.info(f"[Indices] Created {len(validation_indices):,} validation sequences")    
        self.logger.info(f"[Indices] Created {len(test_indices):,} test sequences \n")

        sequence_lengths_train      = [end - start for (start, end, _) in train_indices]
        sequence_lengths_validation = [end - start for (start, end, _) in validation_indices]
        sequence_lengths_test       = [end - start for (start, end, _) in test_indices]

        avg_length_train, avg_length_val, avg_length_test = np.mean(sequence_lengths_train), np.mean(sequence_lengths_validation), np.mean(sequence_lengths_test)
        std_length_train, std_length_val, std_length_test = np.std(sequence_lengths_train),  np.std(sequence_lengths_validation),  np.std(sequence_lengths_test)
        min_length_train, min_length_val, min_length_test = np.min(sequence_lengths_train),  np.min(sequence_lengths_validation),  np.min(sequence_lengths_test)
        max_length_train, max_length_val, max_length_test = np.max(sequence_lengths_train),  np.max(sequence_lengths_validation),  np.max(sequence_lengths_test)
        
        self.logger.info(f"[Indices] Training sequence lengths: mean={avg_length_train:.2f}, std={std_length_train:.2f} timesteps")
        self.logger.info(f"[Indices] Training sequence lengths: min={min_length_train} timesteps, max={max_length_train} timesteps \n")
        
        self.logger.info(f"[Indices] Validation sequence lengths: mean={avg_length_val:.2f}, std={std_length_val:.2f} timesteps")
        self.logger.info(f"[Indices] Validation sequence lengths: min={min_length_val} timesteps, max={max_length_val} timesteps \n")

        self.logger.info(f"[Indices] Test sequence lengths: mean={avg_length_test:.2f}, std={std_length_test:.2f} timesteps")
        self.logger.info(f"[Indices] Test sequence lengths: min={min_length_test} timesteps, max={max_length_test} timesteps \n")
        
        return (train_indices, validation_indices, test_indices), (train_dataframe, validation_dataframe, test_dataframe)


    def create_arrays(self, train_dataframe, validation_dataframe, test_dataframe):   
        train_continuous       = train_dataframe[self.continuous_columns].values
        validation_continuous  = validation_dataframe[self.continuous_columns].values
        test_continuous        = test_dataframe[self.continuous_columns].values

        self.logger.info(f"[Arrays] Extracted training features: continuous={train_continuous.shape}")
        self.logger.info(f"[Arrays] Extracted validation features: continuous={validation_continuous.shape}")
        self.logger.info(f"[Arrays] Extracted test features: continuous={test_continuous.shape} \n")
    
        continuous_array = train_continuous, validation_continuous, test_continuous

        train_categorical      = train_dataframe[self.categorical_columns].values
        validation_categorical = validation_dataframe[self.categorical_columns].values
        test_categorical       = test_dataframe[self.categorical_columns].values
     
        self.logger.info(f"[Arrays] Extracted training features: categorical={train_categorical.shape}")
        self.logger.info(f"[Arrays] Extracted validation features: categorical={validation_categorical.shape}")
        self.logger.info(f"[Arrays] Extracted test features: categorical={test_categorical.shape} \n")
    
        categorical_array = train_categorical, validation_categorical, test_categorical

        train_targets      = train_dataframe[self.target_columns].values
        validation_targets = validation_dataframe[self.target_columns].values
        test_targets       = test_dataframe[self.target_columns].values

        self.logger.info(f"[Arrays] Extracted training targets: {train_targets.shape}")
        self.logger.info(f"[Arrays] Extracted validation targets: {validation_targets.shape}")
        self.logger.info(f"[Arrays] Extracted test targets: {test_targets.shape} \n")

        targets_array = train_targets, validation_targets, test_targets

        return continuous_array, categorical_array, targets_array


    def create_datasets(self, indices, continuous, categorical, targets):
        mask_idxs = []
        known_idx = None
       
        mask_idxs.append(self.continuous_columns.index(self.config.columns.delay_clipped_col))
        known_idx = self.continuous_columns.index(self.config.columns.delay_is_known_col)
        protected_cont_indices = [known_idx]

        train_indices,     validation_indices,     test_indices     = indices
        train_targets,     validation_targets,     test_targets     = targets
        train_continuous,  validation_continuous,  test_continuous  = continuous
        train_categorical, validation_categorical, test_categorical = categorical
        
        train_dataset = SequentialDataset(
            categorical_data=train_categorical,
            continuous_data=train_continuous,
            targets=train_targets,
            indices=train_indices,
            augment=True,
            mask_last_cont=mask_idxs,
            last_known_idx=known_idx,
            protected_cont_indices=protected_cont_indices,
            cfg=self.config,
            logger = self.logger
        )
        
        validation_dataset = SequentialDataset(
            categorical_data=validation_categorical,
            continuous_data=validation_continuous,
            targets=validation_targets,
            indices=validation_indices,
            augment =False,
            mask_last_cont =mask_idxs,
            last_known_idx =known_idx,
            protected_cont_indices=protected_cont_indices,
            cfg =self.config,
            logger = self.logger
        )
        
        test_dataset = SequentialDataset(
            categorical_data=test_categorical,
            continuous_data=test_continuous,
            targets=test_targets,
            indices=test_indices,
            augment=False,
            mask_last_cont=mask_idxs,
            last_known_idx=known_idx,
            protected_cont_indices=protected_cont_indices,
            cfg=self.config,
            logger = self.logger
        )
    
        self.logger.info(f"[Datasets] Created training dataset with {len(train_dataset):,} samples")
        self.logger.info(f"[Datasets] Created validation dataset with {len(validation_dataset):,} samples")
        self.logger.info(f"[Datasets] Created test dataset with {len(test_dataset):,} samples \n")
        
        return train_dataset, validation_dataset, test_dataset


    def collate_sequences(self, batch):
        categorical_list, continuous_list, target_list, lengths = zip(*batch)

        categorical_padded  = pad_sequence(categorical_list, batch_first=True, padding_value=self.config.architecture.categorical_padding_value)
        continuous_padded   = pad_sequence(continuous_list, batch_first=True, padding_value=self.config.architecture.continuous_padding_value)

        targets = torch.stack(target_list)
        lengths = torch.tensor(lengths, dtype=torch.long)

        return categorical_padded, continuous_padded, targets, lengths


    def create_dataloaders(self, train_dataset, validation_dataset, test_dataset):
        is_windows = platform.system() == 'Windows'
        num_workers = 0 if is_windows else self.config.training.num_workers
        use_persistent = num_workers > 0
        
        if is_windows and self.config.training.num_workers > 0:
            self.logger.warning("[Dataloaders] Windows detected: Setting num_workers=0 to avoid multiprocessing issues")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.config.training.pin_memory,
            persistent_workers=use_persistent,
            collate_fn=self.collate_sequences
        )

        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.config.training.pin_memory,
            persistent_workers=use_persistent,
            collate_fn=self.collate_sequences
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.config.training.pin_memory,
            persistent_workers=use_persistent,
            collate_fn=self.collate_sequences
        )

        self.logger.info(f"[Dataloaders] Created training dataloader with {len(train_dataloader):,} batches")
        self.logger.info(f"[Dataloaders] Created validation dataloader with {len(validation_dataloader):,} batches")
        self.logger.info(f"[Dataloaders] Created test dataloader with {len(test_dataloader):,} batches \n")
        return train_dataloader, validation_dataloader, test_dataloader
    

    def run(self):
        self.logger.section("Dataloader Pipeline")
         
        self.logger.subsection("Data Loading")
        dataframe = self.load_data(self.dataframe)
        
        self.logger.subsection("Data Cleaning")
        dataframe = self.clean_data(dataframe)
        
        self.logger.subsection("Target Clipping")
        dataframe = self.clip_target(dataframe)

        self.logger.subsection("Categorical Encoding")
        dataframe = self.encode_categorical(dataframe)
         
        unique_users = dataframe[self.config.columns.group_cols[0]].unique()

        self.logger.subsection("User Subsampling")
        train_users, validation_users, test_users = self.split_users(unique_users)
        
        self.logger.subsection("Dataframe Splitting")
        train_df, val_df, test_df = self.split_dataframes(dataframe, train_users, validation_users, test_users)
        
        self.logger.subsection("Feature Normalization")
        train_df, val_df, test_df = self.normalize_continuous_features(train_df, val_df, test_df)
        
        self.logger.subsection("Target Normalization")
        train_df, val_df, test_df = self.normalize_targets(train_df, val_df, test_df)
        
        self.logger.subsection("Index Creation")
        indices, (train_df, val_df, test_df) = self.create_indices(train_df, val_df, test_df)
        
        self.logger.subsection("Array Creation")
        continuous_array, categorical_array, targets_array  = self.create_arrays(train_df, val_df, test_df)
        
        self.logger.subsection("Dataset Creation")
        train_dataset, val_dataset, test_dataset = self.create_datasets(
            indices=indices,
            continuous=continuous_array,
            categorical=categorical_array,
            targets=targets_array,
        )

        self.logger.subsection("Dataloader Creation")
        train_dataloader, val_dataloader, test_dataloader = self.create_dataloaders(train_dataset, val_dataset, test_dataset)

        self.logger.section("Dataloader Pipeline Complete")
        return train_dataloader, val_dataloader, test_dataloader
 