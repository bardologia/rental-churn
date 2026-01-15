"""
Sequential Data Module for Hierarchical Model

This module loads and prepares data for the hierarchical model that processes
user payment sequences. Instead of treating each invoice independently, it:
1. Groups invoices by user
2. Creates sequences of invoices for each user
3. Uses the sequence to predict the next invoice's default probability
"""

import pandas as pd
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from Configs.config import config
from Utils.logger import Logger


class SequentialDataset(Dataset):
    """
    Dataset that provides sequences of invoices for each user.
    Each sample is a user's invoice sequence, with the target being
    whether the LAST invoice in the sequence leads to default.
    """
    def __init__(self, sequences, cat_cols, cont_cols, target_col='target_default_1'):
        """
        Args:
            sequences: List of DataFrames, each containing one user's invoices
            cat_cols: List of categorical column names
            cont_cols: List of continuous column names
            target_col: Name of target column
        """
        self.sequences = sequences
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.target_col = target_col
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        user_df = self.sequences[idx]
        
        # Extract features
        X_cat = torch.tensor(user_df[self.cat_cols].values, dtype=torch.long)
        X_cont = torch.tensor(user_df[self.cont_cols].values, dtype=torch.float32)
        
        # Target is the last invoice's target
        y = torch.tensor(user_df[self.target_col].iloc[-1], dtype=torch.float32)
        
        # Length of sequence
        length = len(user_df)
        
        return X_cat, X_cont, y, length


def collate_sequences(batch):
    """
    Collate function for variable-length sequences.
    Pads sequences to the max length in the batch.
    """
    X_cat_list, X_cont_list, y_list, lengths = zip(*batch)
    
    # Pad sequences
    X_cat_padded = pad_sequence(X_cat_list, batch_first=True, padding_value=0)
    X_cont_padded = pad_sequence(X_cont_list, batch_first=True, padding_value=0.0)
    
    # Stack targets and lengths
    y = torch.stack(y_list)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return X_cat_padded, X_cont_padded, y, lengths


class SequentialDataModule:
    """
    Data module for sequential (hierarchical) model.
    Groups data by user and creates sequences for temporal modeling.
    """
    def __init__(
        self, 
        data_path, 
        batch_size=32,  # Smaller batch for sequences
        num_workers=0, 
        pin_memory=False, 
        logger=None,
        max_seq_len=50,  # Maximum sequence length per user
        min_seq_len=2    # Minimum invoices per user to include
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        
        if logger is None:
            self.logger = logging.getLogger("SequentialDataModule")
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
        
    def prepare_data(self):
        self.logger.info(f"Loading sequential data from {self.data_path}")
        
        df = pd.read_csv(self.data_path, sep=config.data.separator, low_memory=False)
        
        # Filter columns that exist
        self.cat_cols = [c for c in self.cat_cols if c in df.columns]
        self.cont_cols = [c for c in self.cont_cols if c in df.columns]
        
        self.logger.info(f"Categorical Features: {len(self.cat_cols)}")
        self.logger.info(f"Continuous Features: {len(self.cont_cols)}")
        
        # Encode categoricals
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
        
        # Ensure sorted by user and time
        user_col = 'usuarioId'
        df = df.sort_values([user_col, 'vencimentoData']).reset_index(drop=True)
        
        # === GROUP-BASED SPLIT ===
        # Get unique users
        unique_users = df[user_col].unique()
        user_df = pd.DataFrame({user_col: unique_users})
        
        test_size = config.data.test_size
        val_size = config.data.val_size
        
        # Split users (not rows)
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=config.data.random_state)
        train_val_idx, test_idx = next(gss_test.split(user_df, groups=user_df[user_col]))
        
        train_val_users = user_df.iloc[train_val_idx][user_col].values
        test_users = user_df.iloc[test_idx][user_col].values
        
        val_size_adjusted = val_size / (1 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=config.data.random_state)
        train_idx, val_idx = next(gss_val.split(
            pd.DataFrame({user_col: train_val_users}), 
            groups=train_val_users
        ))
        
        train_users = set(train_val_users[train_idx])
        val_users = set(train_val_users[val_idx])
        test_users = set(test_users)
        
        self.logger.info(f"User Split - Train: {len(train_users)}, Val: {len(val_users)}, Test: {len(test_users)}")
        
        # Split data by user
        train_df = df[df[user_col].isin(train_users)].copy()
        val_df = df[df[user_col].isin(val_users)].copy()
        test_df = df[df[user_col].isin(test_users)].copy()
        
        # Scale continuous features (fit on train only)
        if self.cont_cols:
            train_df[self.cont_cols] = self.scaler.fit_transform(train_df[self.cont_cols])
            val_df[self.cont_cols] = self.scaler.transform(val_df[self.cont_cols])
            test_df[self.cont_cols] = self.scaler.transform(test_df[self.cont_cols])
        
        # Create sequences
        self.train_sequences = self._create_sequences(train_df, user_col)
        self.val_sequences = self._create_sequences(val_df, user_col)
        self.test_sequences = self._create_sequences(test_df, user_col)
        
        self.logger.info(f"Sequences - Train: {len(self.train_sequences)}, Val: {len(self.val_sequences)}, Test: {len(self.test_sequences)}")
        
        # Class distribution
        train_targets = [seq[self.target_cols[0]].iloc[-1] for seq in self.train_sequences]
        pos_rate = np.mean(train_targets)
        self.logger.info(f"Target positive rate: {pos_rate:.2%}")
        
        # Create datasets
        self.train_dataset = SequentialDataset(self.train_sequences, self.cat_cols, self.cont_cols, self.target_cols[0])
        self.val_dataset = SequentialDataset(self.val_sequences, self.cat_cols, self.cont_cols, self.target_cols[0])
        self.test_dataset = SequentialDataset(self.test_sequences, self.cat_cols, self.cont_cols, self.target_cols[0])
    
    def _create_sequences(self, df, user_col):
        """
        Create sequences for each user.
        For users with many invoices, we can create multiple overlapping sequences.
        """
        sequences = []
        
        for user_id, user_df in df.groupby(user_col):
            user_df = user_df.reset_index(drop=True)
            n_invoices = len(user_df)
            
            if n_invoices < self.min_seq_len:
                continue
            
            if n_invoices <= self.max_seq_len:
                # Single sequence for this user
                sequences.append(user_df)
            else:
                # Create sliding window sequences for long histories
                # Focus on recent history with stride
                stride = max(1, self.max_seq_len // 2)
                for start in range(0, n_invoices - self.max_seq_len + 1, stride):
                    end = start + self.max_seq_len
                    sequences.append(user_df.iloc[start:end].reset_index(drop=True))
                
                # Always include the final sequence (most recent)
                if n_invoices > self.max_seq_len:
                    sequences.append(user_df.iloc[-self.max_seq_len:].reset_index(drop=True))
        
        return sequences
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_sequences
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_sequences
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_sequences
        )
