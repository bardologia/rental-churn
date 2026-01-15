"""
Trainer for Hierarchical Sequential Model

Adapted trainer for the hierarchical model that processes user payment sequences.
Main differences from the standard trainer:
1. Handles variable-length sequences with padding
2. Uses forward_sequential for proper temporal modeling
3. Adapted metrics for sequence-level predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
import numpy as np
import time
import copy
import os
from tqdm import tqdm
from Utils.logger import Logger


class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        focal_loss = alpha_t * focal_weight * bce_loss
        return focal_loss.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss - different gamma for positives and negatives."""
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, pos_weight=10.0):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_neg = (probs + self.clip).clamp(max=1)
        
        loss_pos = targets * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))
        
        loss_pos = loss_pos * ((1 - probs) ** self.gamma_pos) * self.pos_weight
        loss_neg = loss_neg * (probs_neg ** self.gamma_neg)
        
        loss = -loss_pos - loss_neg
        return loss.mean()


class SequentialTrainer:
    """Trainer for hierarchical sequential model."""
    
    def __init__(
        self,
        model,
        data_module,
        lr=1e-3,
        epochs=50,
        patience=10,
        mixed_precision=False,
        device=None,
        weight_decay=1e-4,
        checkpoint_dir="checkpoints",
        logger=None,
        max_grad_norm=1.0,
        scheduler_factor=0.5,
        scheduler_patience=5,
        min_lr=1e-6,
        loss_type='asymmetric',
        focal_alpha=0.75,
        focal_gamma=2.0
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dm = data_module
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.loss_type = loss_type
        self.optimal_threshold = 0.3
        
        # Loss function
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            if self.logger:
                self.logger.info(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        elif loss_type == 'asymmetric':
            self.criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, pos_weight=15.0)
            if self.logger:
                self.logger.info("Using Asymmetric Loss (gamma_neg=4, pos_weight=15)")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            if self.logger:
                self.logger.info("Using standard BCE Loss")
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.patience = patience
        self.mixed_precision = mixed_precision
        self.max_grad_norm = max_grad_norm
        self.scaler = GradScaler() if mixed_precision and self.device.type == 'cuda' else None
        
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.min_lr = min_lr
        self.scheduler = None
        
    def save_checkpoint(self, epoch, metric, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': metric,
        }
        
        last_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pth')
        torch.save(state, last_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_path)
            if self.logger:
                self.logger.info(f"New best model saved with AUC: {metric:.4f}")
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        loop = tqdm(dataloader, desc=f"Train Epoch {epoch}")
        for batch_idx, (x_cat, x_cont, y, lengths) in enumerate(loop):
            x_cat = x_cat.to(self.device)
            x_cont = x_cont.to(self.device)
            y = y.to(self.device)
            lengths = lengths.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=self.device.type, enabled=self.mixed_precision and self.device.type == 'cuda'):
                # Use sequential forward for variable-length sequences
                logits = self.model.forward_sequential(x_cat, x_cont, lengths)
                target = y.view(-1, 1)
                loss = self.criterion(logits, target)
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            current_lr = self.optimizer.param_groups[0]['lr']
            loop.set_postfix(loss=loss.item(), lr=current_lr)
        
        avg_loss = total_loss / max(n_batches, 1)
        
        if self.logger:
            self.logger.log_scalar("Loss/train", avg_loss, epoch)
            self.logger.log_scalar("LR", self.optimizer.param_groups[0]['lr'], epoch)
        
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader, epoch=None, phase="val"):
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0
        n_batches = 0
        
        for x_cat, x_cont, y, lengths in dataloader:
            x_cat = x_cat.to(self.device)
            x_cont = x_cont.to(self.device)
            y = y.to(self.device)
            lengths = lengths.to(self.device)
            
            logits = self.model.forward_sequential(x_cat, x_cont, lengths)
            target = y.view(-1, 1)
            
            loss = self.criterion(logits, target)
            total_loss += loss.item()
            n_batches += 1
            
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets = target.cpu().numpy().flatten()
            
            all_preds.extend(probs)
            all_targets.extend(targets)
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        avg_loss = total_loss / max(n_batches, 1)
        
        # Calculate metrics
        try:
            auc_roc = roc_auc_score(all_targets, all_preds) if len(np.unique(all_targets)) > 1 else 0.5
            auc_pr = average_precision_score(all_targets, all_preds) if len(np.unique(all_targets)) > 1 else 0.0
        except:
            auc_roc, auc_pr = 0.5, 0.0
        
        # Find optimal threshold
        if len(np.unique(all_targets)) > 1:
            precisions, recalls, thresholds = precision_recall_curve(all_targets, all_preds)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            if best_idx < len(thresholds):
                self.optimal_threshold = thresholds[best_idx]
        
        # Calculate F1 at optimal threshold
        preds_binary = (all_preds >= self.optimal_threshold).astype(int)
        f1 = f1_score(all_targets, preds_binary, zero_division=0)
        
        if self.logger and epoch is not None:
            self.logger.log_scalar(f"Loss/{phase}", avg_loss, epoch)
            self.logger.log_scalar(f"AUC-ROC/{phase}", auc_roc, epoch)
            self.logger.log_scalar(f"AUC-PR/{phase}", auc_pr, epoch)
            self.logger.log_scalar(f"F1/{phase}", f1, epoch)
        
        return {
            'loss': avg_loss,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'f1': f1,
            'threshold': self.optimal_threshold
        }
    
    def fit(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.min_lr,
            verbose=True
        )
        
        best_auc = 0
        best_model_state = None
        patience_counter = 0
        
        train_loader = self.dm.train_dataloader()
        val_loader = self.dm.val_dataloader()
        
        for epoch in range(1, self.epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.evaluate(val_loader, epoch, phase="val")
            val_auc = val_metrics['auc_roc']
            
            # Scheduler step
            self.scheduler.step(val_auc)
            
            # Logging
            if self.logger:
                self.logger.info(
                    f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                    f"Val AUC={val_auc:.4f}, Val F1={val_metrics['f1']:.4f}, "
                    f"Threshold={val_metrics['threshold']:.3f}"
                )
            
            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                self.save_checkpoint(epoch, val_auc, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.logger:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.model
    
    def test(self, test_loader):
        if self.logger:
            self.logger.info("=== TEST SET EVALUATION ===")
        
        metrics = self.evaluate(test_loader, phase="test")
        
        if self.logger:
            self.logger.info(f"Test AUC-ROC: {metrics['auc_roc']:.4f}")
            self.logger.info(f"Test AUC-PR: {metrics['auc_pr']:.4f}")
            self.logger.info(f"Test F1: {metrics['f1']:.4f}")
            self.logger.info(f"Optimal Threshold: {metrics['threshold']:.3f}")
        
        return metrics
