import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, LambdaLR
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, brier_score_loss
import numpy as np
import time
import copy
import os
import math
from tqdm import tqdm
from Utils.logger import Logger
from Configs.config import config


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = None):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps if warmup_steps is not None else config.model.ema_warmup_steps
        self.step = 0
        
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def _get_decay(self) -> float:
        if self.step < self.warmup_steps:
            return min(self.decay, (1 + self.step) / (10 + self.step))
        return self.decay
    
    @torch.no_grad()
    def update(self):
        decay = self._get_decay()
        self.step += 1
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self):
        return {
            'shadow': self.shadow,
            'step': self.step,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.step = state_dict['step']
        self.decay = state_dict.get('decay', self.decay)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_negative=None, gamma_positive=None, clip=None, reduction='mean'):
        super().__init__()
        self.gamma_negative = gamma_negative if gamma_negative is not None else config.loss.asymmetric_gamma_negative
        self.gamma_positive = gamma_positive if gamma_positive is not None else config.loss.asymmetric_gamma_positive
        self.clip = clip if clip is not None else config.loss.asymmetric_clip
        self.reduction = reduction
        
    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        probabilities_clipped = probabilities.clamp(min=self.clip)

        positive_loss = targets * torch.log(probabilities_clipped.clamp(min=1e-8))
        if self.gamma_positive > 0:
            positive_loss = positive_loss * ((1 - probabilities) ** self.gamma_positive)
        
        negative_probabilities = (probabilities - self.clip).clamp(min=0)
        negative_loss = (1 - targets) * torch.log((1 - negative_probabilities).clamp(min=1e-8))
        if self.gamma_negative > 0:
            negative_loss = negative_loss * (negative_probabilities ** self.gamma_negative)
        
        loss = -(positive_loss + negative_loss)
        return loss.mean()
        

class Trainer: 
    def __init__(
        self,
        model,
        data_module,
        learning_rate=1e-3,
        epochs=50,
        patience=10,
        mixed_precision=False,
        device=None,
        weight_decay=1e-4,
        checkpoint_dir="checkpoints",
        logger=None,
        max_grad_norm=1.0,
        scheduler_type='cosine',
        scheduler_factor=0.5,
        scheduler_patience=5,
        min_learning_rate=1e-6,
        loss_type='asymmetric',
        warmup_epochs=5,
        use_calibration=True,
        use_ema=True,
        ema_decay=0.9999,
        use_compile=False  
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Log GPU information
        if logger:
            logger.info(f"[Device] Using: {self.device}")
            if self.device.type == 'cuda':
                logger.info(f"[GPU] Name: {torch.cuda.get_device_name(0)}")
                logger.info(f"[GPU] Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                logger.info(f"[GPU] CUDA Version: {torch.version.cuda}")
        
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True 
            torch.backends.cuda.matmul.allow_tf32 = True  
            torch.backends.cudnn.allow_tf32 = True
        
        self.model = model.to(self.device)
        
        if use_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode='reduce-overhead')
            if logger:
                logger.info("[Optimization] torch.compile enabled (reduce-overhead mode)")
        
        self.data_module = data_module
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.loss_type = loss_type
        self.optimal_thresholds = {}
        self.use_calibration = use_calibration
        self.warmup_epochs = warmup_epochs
        self.scheduler_type = scheduler_type
        self.learning_rate = learning_rate
        
        self.criterion = AsymmetricLoss()
        self.logger.info(f"[Loss] Asymmetric Loss (ASL) with γ_neg={config.loss.asymmetric_gamma_negative}, γ_pos={config.loss.asymmetric_gamma_positive}")
               
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.epochs = epochs
        self.patience = patience
        self.mixed_precision = mixed_precision
        self.max_grad_norm = max_grad_norm
        self.scaler = GradScaler() if mixed_precision and self.device.type == 'cuda' else None
        
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.min_learning_rate = min_learning_rate
        self.scheduler = None
        
        self.use_ema = use_ema
        self.ema = None
        if use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
            self.logger.info(f"[EMA] Exponential Moving Average enabled (decay={ema_decay}, warmup={config.model.ema_warmup_steps} steps)")
        
    def save_checkpoint(self, epoch, metric, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': metric,
            'optimal_thresholds': self.optimal_thresholds,
            'ema_state_dict': self.ema.state_dict() if self.ema else None,
        }
        
        last_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pth')
        torch.save(state, last_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_path)
            
            if self.ema:
                ema_path = os.path.join(self.checkpoint_dir, 'best_model_ema.pth')
                self.ema.apply_shadow()
                torch.save(self.model.state_dict(), ema_path)
                self.ema.restore()
            
            if self.logger:
                self.logger.info(f"[Checkpoint] New best model saved → AUC: {metric:.4f}")
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        running_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
        log_interval = 50  
        
        loop = tqdm(dataloader, desc=f"Train Epoch {epoch}")
        for batch_index, (categorical_features, continuous_features, targets, lengths) in enumerate(loop):
            categorical_features = categorical_features.to(self.device, non_blocking=True)
            continuous_features = continuous_features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=self.device.type, enabled=self.mixed_precision and self.device.type == 'cuda'):
                logits = self.model(categorical_features, continuous_features, lengths)
                target_tensor = targets.view(-1, len(config.columns.target_cols))
                loss = self.criterion(logits, target_tensor)
            
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
            
            if self.ema:
                self.ema.update()
            
            running_loss += loss.detach()
            num_batches += 1
           
            if num_batches % log_interval == 0:
                current_learning_rate = self.optimizer.param_groups[0]['lr']
                loop.set_postfix(loss=f"{(running_loss / num_batches).item():.4f}", lr=f"{current_learning_rate:.2e}")
            
        
        
        average_loss = (running_loss / max(num_batches, 1)).item()
        if self.logger:
            self.logger.log_scalar("Loss/train", average_loss, epoch)
            self.logger.log_scalar("LR", self.optimizer.param_groups[0]['lr'], epoch)
        return average_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader, epoch=None, phase="val", use_ema=True):
        if use_ema and self.ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        # Collect as lists of tensors, convert at end (more efficient)
        all_probabilities = []
        all_targets_list = []
        
        running_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
        
        for categorical_features, continuous_features, targets, lengths in dataloader:
            categorical_features = categorical_features.to(self.device, non_blocking=True)
            continuous_features = continuous_features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)
            
            logits = self.model(categorical_features, continuous_features, lengths)
            target_tensor = targets.view(-1, len(config.columns.target_cols))
            
            loss = self.criterion(logits, target_tensor)
            running_loss += loss.detach()
            num_batches += 1
            
            # Keep on CPU but as tensors (avoid per-element extend)
            all_probabilities.append(torch.sigmoid(logits).cpu())
            all_targets_list.append(target_tensor.cpu())
        
        if use_ema and self.ema:
            self.ema.restore()
        
        # Single .item() call at end instead of every batch
        average_loss = (running_loss / max(num_batches, 1)).item()
        
        # Concatenate all at once and convert to numpy (much faster than extend)
        all_probs_tensor = torch.cat(all_probabilities, dim=0).numpy()
        all_targs_tensor = torch.cat(all_targets_list, dim=0).numpy()
        
        metrics = {}
        target_names = config.columns.target_cols
        
        for target_index, name in enumerate(target_names):
            predictions = all_probs_tensor[:, target_index]
            targets_numpy = all_targs_tensor[:, target_index]
            
            try:
                auc_roc = roc_auc_score(targets_numpy, predictions) if len(np.unique(targets_numpy)) > 1 else 0.5
                auc_pr = average_precision_score(targets_numpy, predictions) if len(np.unique(targets_numpy)) > 1 else 0.0
            except:
                auc_roc, auc_pr = 0.5, 0.0
            
            if len(np.unique(targets_numpy)) > 1:
                precisions, recalls, thresholds = precision_recall_curve(targets_numpy, predictions)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
                best_threshold_index = np.argmax(f1_scores)
                if best_threshold_index < len(thresholds):
                    optimal_threshold = thresholds[best_threshold_index]
                else:
                    optimal_threshold = 0.5
            else:
                optimal_threshold = np.mean(targets_numpy) if len(targets_numpy) > 0 else 0.5
            
            if phase == "val":
                self.optimal_thresholds[name] = optimal_threshold
                threshold = optimal_threshold
            elif phase == "test" and name in self.optimal_thresholds:
                threshold = self.optimal_thresholds[name]
            else:
                threshold = optimal_threshold
            
            predictions_binary = (predictions >= threshold).astype(int)
            
            f1 = f1_score(targets_numpy, predictions_binary, zero_division=0)
            if self.logger and epoch is not None:
                self.logger.log_scalar(f"Loss/{phase}_{name}", average_loss, epoch)
                self.logger.log_scalar(f"AUC-ROC/{phase}_{name}", auc_roc, epoch)
                self.logger.log_scalar(f"AUC-PR/{phase}_{name}", auc_pr, epoch)
                self.logger.log_scalar(f"F1/{phase}_{name}", f1, epoch)
            
            metrics[f'{name}_auc_roc'] = auc_roc
            metrics[f'{name}_auc_pr'] = auc_pr
            metrics[f'{name}_f1'] = f1
            metrics[f'{name}_threshold'] = threshold
        
        metrics['loss'] = average_loss
        return metrics
    
    def fit(self):
        self.logger.section("Model Training")
        torch.cuda.empty_cache()
        num_parameters = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        
        self.logger.subsection("Training Configuration")
        self.logger.experiment_config({
            "Device": str(self.device),
            "Trainable Parameters": f"{num_parameters:,}",
            "Epochs": self.epochs,
            "Learning Rate": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            "Batch Size": self.data_module.batch_size,
            "Weight Decay": f"{self.optimizer.param_groups[0].get('weight_decay', 0):.2e}",
            "Mixed Precision": "Enabled" if self.mixed_precision else "Disabled",
            "EMA": f"Enabled (decay={self.ema.decay})" if self.ema else "Disabled",
            "Gradient Clipping": f"max_norm={self.max_grad_norm}",
        })
        
        self.logger.subsection("Dataset Statistics")
        self.logger.experiment_config({
            "Train Samples": f"{len(self.data_module.train_indices):,}",
            "Validation Samples": f"{len(self.data_module.validation_indices):,}",
            "Test Samples": f"{len(self.data_module.test_indices):,}",
        })

        train_loader = self.data_module.train_dataloader()
        validation_loader = self.data_module.validation_dataloader()
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.model.scheduler_t0,
            T_mult=config.model.scheduler_t_mult,
            eta_min=self.min_learning_rate
        )
        self.logger.info(f"[Scheduler] CosineAnnealingWarmRestarts (T_0={config.model.scheduler_t0}, T_mult={config.model.scheduler_t_mult}, η_min={self.min_learning_rate:.2e})")
    
        self.logger.subsection("Training Progress")
        
        best_auc = 0
        best_precision_recall = 0
        best_model_state = None
        best_ema_state = None
        patience_counter = 0
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            
            validation_metrics = self.evaluate(validation_loader, epoch, phase="val")
        
            target_names = config.columns.target_cols
            validation_auc = np.mean([validation_metrics.get(f'{target}_auc_roc', 0.5) for target in target_names])
            validation_f1  = np.mean([validation_metrics.get(f'{target}_f1', 0.0) for target in target_names])
            validation_precision_recall  = np.mean([validation_metrics.get(f'{target}_auc_pr', 0.0) for target in target_names])
            
            if self.scheduler_type == 'cosine':
                self.scheduler.step()
            else:
                self.scheduler.step(validation_precision_recall)
            
            current_learning_rate = self.optimizer.param_groups[0]['lr']
            
            if self.logger:
                self.logger.log_epoch_results(
                    epoch=epoch,
                    total_epochs=self.epochs,
                    train_loss=train_loss,
                    val_loss=validation_metrics['loss'],
                    metrics={
                        "AUC-ROC": validation_auc,
                        "AUC-PR": validation_precision_recall,
                        "F1-Score": validation_f1,
                        "LR": current_learning_rate,
                    }
                )
                for target in target_names:
                    target_auc = validation_metrics.get(f'{target}_auc_roc', 0.5)
                    target_precision_recall = validation_metrics.get(f'{target}_auc_pr', 0.0)
                    target_f1 = validation_metrics.get(f'{target}_f1', 0.0)
                    target_threshold = validation_metrics.get(f'{target}_threshold', 0.5)
                    self.logger.info(f"    {target}: AUC={target_auc:.4f} | PR={target_precision_recall:.4f} | F1={target_f1:.4f} | τ={target_threshold:.3f}")
       
            if validation_precision_recall > best_precision_recall:
                best_auc = validation_auc
                best_precision_recall = validation_precision_recall
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_thresholds = copy.deepcopy(self.optimal_thresholds)
                best_ema_state = copy.deepcopy(self.ema.state_dict())
                patience_counter = 0
                self.save_checkpoint(epoch, validation_precision_recall, is_best=True)
                self.logger.info(f"    ★ New Best Model: AUC-PR={validation_precision_recall:.4f}, AUC-ROC={validation_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.warning(f"[Early Stopping] Training halted at epoch {epoch} (patience={self.patience}). Best AUC-PR: {best_precision_recall:.4f}")
                    break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.optimal_thresholds = best_thresholds
            if best_ema_state is not None:
                self.ema.load_state_dict(best_ema_state)

        if self.use_calibration and hasattr(self.model, 'calibrate_temperature'):
            self.logger.subsection("Temperature Calibration")
            self.logger.info("[Calibration] Performing temperature scaling on validation set")
            
            self.ema.apply_shadow()
            self.model.calibrate_temperature(validation_loader, self.device)
            
            self.ema.restore()
            self.logger.info("[Calibration] Temperature scaling complete")
        
        self.logger.log_experiment_summary(
            best_metrics={
                "Best AUC-ROC": best_auc,
                "Best AUC-PR": best_precision_recall,
            },
            notes=f"Training completed successfully. Model checkpoints saved to {self.checkpoint_dir}"
        )
        
        return self.model
    
    def test(self, test_loader):
        self.logger.section("Model Evaluation on Test Set")
        
        metrics = self.evaluate(test_loader, phase="test")
        
        target_names = config.columns.target_cols
        average_auc = np.mean([metrics.get(f'{target}_auc_roc', 0.5) for target in target_names])
        average_precision_recall  = np.mean([metrics.get(f'{target}_auc_pr', 0.0) for target in target_names])
        average_f1  = np.mean([metrics.get(f'{target}_f1', 0.0) for target in target_names])
        
        metrics['avg_auc_roc'] = average_auc
        metrics['avg_auc_pr'] = average_precision_recall
        metrics['avg_f1'] = average_f1

        if self.logger:
            self.logger.log_evaluation_results(
                dataset_name="Test Set",
                metrics={
                    "Average AUC-ROC": average_auc,
                    "Average AUC-PR": average_precision_recall,
                    "Average F1-Score": average_f1,
                    "Loss": metrics['loss'],
                }
            )
            
            test_rows = []
            for target in target_names:
                test_rows.append([
                    target,
                    f"{metrics.get(f'{target}_auc_roc', 0):.4f}",
                    f"{metrics.get(f'{target}_auc_pr', 0):.4f}",
                    f"{metrics.get(f'{target}_f1', 0):.4f}",
                    f"{metrics.get(f'{target}_threshold', 0.5):.3f}"
                ])
            
            self.logger.metrics_table(
                headers=["Target", "AUC-ROC", "AUC-PR", "F1-Score", "Threshold"],
                rows=test_rows,
                title="Per-Target Test Results"
            )
        
        return metrics
