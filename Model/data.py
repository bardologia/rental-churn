import pandas as pd
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from Configs.config import config
from Utils.logger import Logger


class TrainingDataset(Dataset):
    def __init__(self, X_cat, X_cont, y):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        # Store labels for weighted sampling
        self.labels = self.y.view(-1).numpy() if self.y.dim() > 1 else self.y.numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_cont[idx], self.y[idx]


class DataModule:
    def __init__(self, data_path, batch_size=64, num_workers=0, pin_memory=False, logger: Logger = None, use_balanced_sampling=True, oversample_ratio=0.3):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_balanced_sampling = use_balanced_sampling
        self.oversample_ratio = oversample_ratio  # Target ratio of positives in each batch
        
        if logger is None:
            self.logger = logging.getLogger("DataModule")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger
        
        self.cat_cols = config.columns.cat_cols
        
        self.cont_cols = config.columns.cont_cols
        
        self.target_cols = config.columns.target_cols
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.emb_dims = []
        self.pos_weights = None

    def prepare_data(self):
        msg = f"Loading data from {self.data_path}"
        
        self.logger.info(msg)
            
        df = pd.read_csv(self.data_path, sep=config.data.separator, low_memory=False)
        
        self.cat_cols = [c for c in self.cat_cols if c in df.columns]
        self.cont_cols = [c for c in self.cont_cols if c in df.columns]
        
        self.logger.info(f"Categorical Features: {len(self.cat_cols)}")
        self.logger.info(f"Continuous Features: {len(self.cont_cols)}")
     
        for col in self.cat_cols:
            df[col] = df[col].astype(str)

        self.emb_dims = []
        for col in self.cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            cardinality = len(le.classes_)
            emb_dim = min(50, (cardinality + 1) // 2)
            self.emb_dims.append((cardinality, emb_dim))

        df = df.dropna(subset=self.target_cols)
        
        # Group-based split strategy - prevents data leakage
        # Same user cannot appear in train AND test/val
        test_size = config.data.test_size
        val_size = config.data.val_size
        
        # Ensure we have user_id column
        user_col = 'usuarioId'
        if user_col not in df.columns:
            self.logger.warning(f"Column '{user_col}' not found. Falling back to random split.")
            user_col = None
        
        if user_col:
            # First split: separate test set by user groups
            gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=config.data.random_state)
            train_val_idx, test_idx = next(gss_test.split(df, groups=df[user_col]))
            train_val_df = df.iloc[train_val_idx].copy()
            test_df = df.iloc[test_idx].copy()
            
            # Second split: separate validation from training by user groups
            val_size_adjusted = val_size / (1 - test_size)
            gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=config.data.random_state)
            train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[user_col]))
            train_df = train_val_df.iloc[train_idx].copy()
            val_df = train_val_df.iloc[val_idx].copy()
            
            # Verify no user overlap
            train_users = set(train_df[user_col].unique())
            val_users = set(val_df[user_col].unique())
            test_users = set(test_df[user_col].unique())
            
            assert len(train_users & val_users) == 0, "User leakage between train and val!"
            assert len(train_users & test_users) == 0, "User leakage between train and test!"
            assert len(val_users & test_users) == 0, "User leakage between val and test!"
            
            self.logger.info(f"Group Split by '{user_col}' - No user leakage verified")
            self.logger.info(f"Unique users - Train: {len(train_users)}, Val: {len(val_users)}, Test: {len(test_users)}")
        else:
            # Fallback to random split (not recommended)
            from sklearn.model_selection import train_test_split
            train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=config.data.random_state)
            val_size_adjusted = val_size / (1 - test_size)
            train_df, val_df = train_test_split(train_val_df, test_size=val_size_adjusted, random_state=config.data.random_state)

        # Fit Scaler ONLY on Training Data
        if self.cont_cols:
            train_df[self.cont_cols] = self.scaler.fit_transform(train_df[self.cont_cols])
            val_df[self.cont_cols] = self.scaler.transform(val_df[self.cont_cols])
            test_df[self.cont_cols] = self.scaler.transform(test_df[self.cont_cols])
        
        self.logger.info(f"Group-based Split Strategy Implemented (seed={config.data.random_state})")


        def get_arrays(dframe):
            return (
                dframe[self.cat_cols].values,
                dframe[self.cont_cols].values,
                dframe[self.target_cols].values
            )

        X_cat_train, X_cont_train, y_train = get_arrays(train_df)

        pos_counts = np.sum(y_train, axis=0)
        total_counts = len(y_train)
        imbalance_ratio = (total_counts - pos_counts) / (pos_counts + 1e-8)
        self.logger.info(f"Class imbalance ratio (neg:pos): {imbalance_ratio[0]:.1f}:1")

        X_cat_val, X_cont_val, y_val = get_arrays(val_df)
        X_cat_test, X_cont_test, y_test = get_arrays(test_df)

        self.train_dataset = TrainingDataset(X_cat_train, X_cont_train, y_train)
        self.val_dataset = TrainingDataset(X_cat_val, X_cont_val, y_val)
        self.test_dataset = TrainingDataset(X_cat_test, X_cont_test, y_test)
        
        if self.use_balanced_sampling:
            self.train_sampler = self._create_balanced_sampler(y_train)
            self.logger.info(f"Using balanced sampling with target ratio: {self.oversample_ratio:.1%}")
        else:
            self.train_sampler = None
        
        self.logger.info(f"Train Size: {len(self.train_dataset)}")
        self.logger.info(f"Val Size: {len(self.val_dataset)}")
        self.logger.info(f"Test Size: {len(self.test_dataset)}")
        
        # Log class distribution
        train_pos_rate = y_train.mean()
        val_pos_rate = y_val.mean()
        test_pos_rate = y_test.mean()
        self.logger.info(f"Class Distribution - Train: {train_pos_rate:.2%} pos, Val: {val_pos_rate:.2%} pos, Test: {test_pos_rate:.2%} pos")

    def _create_balanced_sampler(self, y_train):
        """
        Create a weighted sampler that oversamples minority class.
        Target: ~oversample_ratio positive samples per batch.
        """
        labels = y_train.flatten() if y_train.ndim > 1 else y_train
        
        # Count class frequencies
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        pos_rate = n_pos / len(labels)
        
        # Calculate weights to achieve target ratio
        # We want: weight_pos * n_pos / (weight_pos * n_pos + weight_neg * n_neg) = oversample_ratio
        target_ratio = self.oversample_ratio
        weight_pos = target_ratio / pos_rate
        weight_neg = (1 - target_ratio) / (1 - pos_rate)
        
        # Normalize weights
        weight_pos = weight_pos / (weight_pos + weight_neg) * 2
        weight_neg = weight_neg / (weight_pos + weight_neg) * 2
        
        # Assign weight to each sample
        sample_weights = np.where(labels == 1, weight_pos, weight_neg)
        sample_weights = torch.from_numpy(sample_weights).float()
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True
        )
        
        return sampler

    def train_dataloader(self):
        if self.use_balanced_sampling and self.train_sampler is not None:
            return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                sampler=self.train_sampler,  # Use weighted sampler
                num_workers=self.num_workers, 
                pin_memory=self.pin_memory
            )
        else:
            return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,  # Shuffle when not using sampler
                num_workers=self.num_workers, 
                pin_memory=self.pin_memory
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
