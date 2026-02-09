import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import median_absolute_error
import numpy as np
import itertools
import copy
from tqdm.auto import tqdm


class Loss(nn.Module):
    def __init__(
        self,
        huber_delta: float = 1.0,
        quantiles: list = [0.1, 0.5, 0.9],
        quantile_weight: float = 0.3,
        threshold_weight: float = 0.2,
        thresholds: list = [15.0, 30.0],
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.quantiles = torch.tensor(quantiles)
        self.quantile_weight = quantile_weight
        self.threshold_weight = threshold_weight
        self.thresholds = thresholds
        
    def huber_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        diff = preds - targets
        abs_diff = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=self.huber_delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic.pow(2) + self.huber_delta * linear
        return loss
    
    def quantile_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        quantiles = self.quantiles.to(preds.device)
        errors = targets.unsqueeze(-1) - preds.unsqueeze(-1)
        loss = torch.max(quantiles * errors, (quantiles - 1) * errors)
        return loss.mean(dim=-1)
    
    def threshold_focused_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros_like(preds)
        for threshold in self.thresholds:
            pred_above = (preds > threshold).float()
            target_above = (targets > threshold).float()
            threshold_proximity = torch.exp(-torch.abs(targets - threshold) / 5.0)
            misclassification = torch.abs(pred_above - target_above)
            loss += misclassification * threshold_proximity * torch.abs(preds - targets)
        
        return loss
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        huber = self.huber_loss(preds, targets)
        quantile = self.quantile_loss(preds, targets)
        threshold = self.threshold_focused_loss(preds, targets)
        
        total_loss = (
            huber + 
            self.quantile_weight * quantile + 
            self.threshold_weight * threshold
        )
        
        return total_loss


class Warmup:
    def __init__(self, optimizer, warmup_steps: int, warmup_start_factor: float = 0.1, enabled: bool = True):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.warmup_start_factor = warmup_start_factor
        self.enabled = enabled
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
        self.warmup_finished = False
        
        if self.enabled and self.warmup_steps > 0:
            self._apply_warmup_factor(self.warmup_start_factor)
    
    def _apply_warmup_factor(self, factor: float) -> None:
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * factor
    
    def step(self) -> None:
        if not self.enabled or self.warmup_steps <= 0:
            return
        
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
            self._apply_warmup_factor(factor)
        elif not self.warmup_finished:
            self._apply_warmup_factor(1.0)
            self.warmup_finished = True
    
    def is_finished(self) -> bool:
        return self.warmup_finished or not self.enabled or self.warmup_steps <= 0


class Scheduler:
    def __init__(self, optimizer, scheduler_type: str = 'cosine', warmup: Warmup = None, **scheduler_kwargs):
        self.optimizer = optimizer
        self.warmup = warmup
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(optimizer, **scheduler_kwargs)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def step(self, epoch: bool = True) -> None:
        if self.warmup and not self.warmup.is_finished():
            return
        
        if epoch:
            self.scheduler.step()
    
    def state_dict(self) -> dict:
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict: dict) -> None:
        self.scheduler.load_state_dict(state_dict)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.backup[name] = param.detach().clone()
            param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = state.get("decay", self.decay)
        self.shadow = state.get("shadow", self.shadow)
        self.backup = {}


class Trainer: 
    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        target_scaler=None,
        logger = None,
        config=None,
    ):
        self.logger = logger
        self.config = config 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_loader      = train_loader
        self.validation_loader = validation_loader
        self.target_scaler     = target_scaler
    
        self.high_target_weight = self.config.training.high_target_weight
        self.logger.info(f"[High Target Weight] Weight: {self.high_target_weight}\n")
  
        self.criterion = Loss(
            huber_delta=self.config.loss.huber_delta,
            quantiles=self.config.loss.quantiles,
            quantile_weight=self.config.loss.quantile_weight,
            threshold_weight=self.config.loss.threshold_weight,
            thresholds=self.config.loss.thresholds,
        )
        
        self.logger.info(f"[Loss Function] Loss - Huber(δ={self.config.loss.huber_delta}), "
                        f"Quantile(w={self.config.loss.quantile_weight}), "
                        f"Threshold(w={self.config.loss.threshold_weight}, t={self.config.loss.thresholds})\n")
        
        self.scaler    = GradScaler() if self.config.training.mixed_precision else None

        self.layerwise_optimizer()

        self.grad_accum_steps = max(1, self.config.training.grad_accum_steps)
        self.logger.info(f"[Grad Accumulation] Effective batch size: {self.train_loader.batch_size * self.grad_accum_steps}\n")

        self.warmup = Warmup(
            optimizer=self.optimizer,
            warmup_steps=self.config.training.warmup_steps,
            warmup_start_factor=self.config.training.warmup_start_factor,
            enabled=self.config.training.warmup_enabled
        )
        self.logger.info(f"[Warmup] Enabled: {self.config.training.warmup_enabled}, Steps: {self.config.training.warmup_steps}, Start Factor: {self.config.training.warmup_start_factor}")
        
        t_max = self.config.training.epochs if self.config.scheduler.t_max is None else self.config.scheduler.t_max
        self.scheduler_manager = Scheduler(
            optimizer=self.optimizer,
            scheduler_type='cosine',
            warmup=self.warmup,
            T_max=t_max,
            eta_min=self.config.scheduler.eta_min
        )
        
        self.logger.info(f"[Scheduler] CosineAnnealingLR with T_max={t_max}, eta_min={self.config.scheduler.eta_min}\n")

        self.ema_enabled = self.config.ema.use_ema
        self.ema = EMA(self.model, decay=self.config.ema.ema_decay) if self.ema_enabled else None
        self.ema_warmup_steps = self.config.ema.ema_warmup_steps
        self.ema_warmup_denominator = self.config.ema.ema_warmup_denominator
        self.logger.info(f"[EMA] Enabled: {self.ema_enabled}, Decay: {self.config.ema.ema_decay}, Warmup Steps: {self.ema_warmup_steps}\n")
        
        self.global_step = 0
        self.checkpoint = None

        self.logger.section("Trainer Initialization")
        self.logger.info(f"[High Target Weight] Using high target weight: {self.high_target_weight}\n")
        self.log_parameters(model)

    def log_parameters(self, model):
        self.logger.section("[Model Parameter Counts]")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.subsection(f"Full Model - Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

        children = []
        for name, module in model.named_children():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            children.append((name, total, trainable))

        children.sort(key=lambda x: x[1], reverse=True)

        for name, total, trainable in children:
            self.logger.info(f"[{name}] Total parameters: {total:,}, Trainable: {trainable:,}")

        self.logger.info("")

    def layerwise_optimizer(self) -> None:
        layerwise = self.config.layerwise
        param_groups = []
        
        lr_map = {
            "tokenizer": layerwise.tokenizer_lr,
            "invoice_encoder": layerwise.invoice_encoder_lr,
            "sequence_encoder": layerwise.sequence_encoder_lr,
            "temporal_attention": layerwise.cross_attention_lr,
            "head_days": layerwise.head_lr,
        }
        
        for module_name, lr in lr_map.items():
            module = getattr(self.model, module_name, None)
            param_groups.append({"params": module.parameters(), "lr": lr})
     
        self.optimizer = optim.AdamW(param_groups, weight_decay=self.config.training.weight_decay)
        self.logger.info(f"[Optimizer] Configured layer-wise learning rates:")
        for module_name, lr in lr_map.items():
            self.logger.info(f" [{module_name}] : {lr}'")

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = self.criterion(preds, targets).view(-1)

        if self.high_target_weight and self.target_scaler is not None and self.high_target_weight > 0:
            den_targets = np.expm1(self.target_scaler.inverse_transform(targets.detach().cpu().numpy().reshape(-1, 1))).reshape(-1)
            den_targets = torch.from_numpy(den_targets).to(self.device).float()
            mean_den = den_targets.mean().clamp_min(1e-8)
            weights = 1.0 + self.high_target_weight * (den_targets / mean_den)
            weighted = losses * weights
            return weighted.mean()

        return losses.mean()

    def update_ema(self) -> None:
        if not self.ema_enabled or self.ema is None:
            return
        if self.global_step < self.ema_warmup_steps:
            return
        self.ema.update(self.model)
   
    def backward(self, loss, step: bool):
        if self.scaler:
            self.scaler.scale(loss).backward()
            if step:
                self.warmup.step()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self.update_ema()
        else:
            loss.backward()
            if step:
                self.warmup.step()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self.update_ema()

    def train_epoch(self, epoch=None):
        self.model.train()
        running_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
   
        if self.config.overfit.overfit_single_batch:
            single_batch = next(iter(self.train_loader))
            batch_iterable = itertools.repeat(single_batch, len(self.train_loader))
            loop = tqdm(batch_iterable, desc=f"Train Epoch {epoch} (Overfit Single Batch)", total=len(self.train_loader))
        else:
            loop = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")

        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(loop):
            categorical_features, continuous_features, targets, lengths = batch

            categorical_features = categorical_features.to(self.device, non_blocking=True)
            continuous_features  = continuous_features.to(self.device, non_blocking=True)
            targets              = targets.to(self.device, non_blocking=True)
            lengths              = lengths.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.config.training.mixed_precision):
                preds         = self.model(categorical_features, continuous_features, lengths)
                target_tensor = targets.view(-1)
                raw_loss      = self.loss(preds, target_tensor)
                loss          = raw_loss / self.grad_accum_steps

            is_last = (batch_idx + 1) == len(self.train_loader)
            should_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or is_last
            self.backward(loss, step=should_step)
            running_loss += raw_loss.detach()
            num_batches += 1
        
        average_loss = (running_loss / max(num_batches, 1)).item()    
        return average_loss
    
    def compute_metrics(self, den_targets, den_preds, average_loss=None) -> dict:
        den_preds   = np.asarray(den_preds).flatten()
        den_targets = np.asarray(den_targets).flatten()

        mae = float(np.mean(np.abs(den_preds - den_targets)))
        rmse = float(np.sqrt(np.mean((den_preds - den_targets) ** 2)))
        ss_res = float(np.sum((den_targets - den_preds) ** 2))
        ss_tot = float(np.sum((den_targets - np.mean(den_targets)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
        std = float(np.std(den_targets - den_preds))
        medae = float(median_absolute_error(den_targets, den_preds))
        max_error = float(np.max(np.abs(den_targets - den_preds)))
        p50 = np.percentile(np.abs(den_targets - den_preds), 50)
        p90 = np.percentile(np.abs(den_targets - den_preds), 90)
        p95 = np.percentile(np.abs(den_targets - den_preds), 95)

        abs_err = np.abs(den_targets - den_preds)
        error_bin_0_5 = float(np.mean(abs_err <= 5) * 100)
        error_bin_5_10 = float(np.mean((abs_err > 5) & (abs_err <= 10)) * 100)
        error_bin_10_15 = float(np.mean((abs_err > 10) & (abs_err <= 15)) * 100)
        error_bin_15_20 = float(np.mean((abs_err > 15) & (abs_err <= 20)) * 100)
        error_bin_20_25 = float(np.mean((abs_err > 20) & (abs_err <= 25)) * 100)
        error_bin_above_25 = float(np.mean(abs_err > 25) * 100)

        def mean_target_range(low, high=None):
            mask = (den_targets > low) & (den_targets <= high)
            vals = abs_err[mask]
            return float(np.mean(vals)) if len(vals) > 0 else float('nan')

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'std': std,
            'p50': p50,
            'p90': p90,
            'p95': p95,
            'medae': medae,
            'max_error': max_error,
            'loss': average_loss,
            
            'error_0_5': error_bin_0_5,
            'error_5_10': error_bin_5_10,
            'error_10_15': error_bin_10_15,
            'error_15_20': error_bin_15_20,
            'error_20_25': error_bin_20_25,
            'error_above_25': error_bin_above_25,
    
            'target_0_5': mean_target_range(0, 5),
            'target_5_10': mean_target_range(5, 10),
            'target_10_15': mean_target_range(10, 15),
            'target_15_20': mean_target_range(15, 20),
            'target_20_25': mean_target_range(20, 25),
            'target_above_25': mean_target_range(25, 30),
        }
        
        return metrics

    def save_checkpoint(self, metrics: dict = None) -> None:
        if not self.logger or not getattr(self.logger, "log_dir", None):
            return

        os.makedirs(self.logger.log_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.logger.log_dir, "checkpoint.pt")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler_manager.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "ema_state_dict": self.ema.state_dict() if self.ema else None,
            "global_step": self.global_step,
            "best_metrics": metrics if metrics is not None else {},
            "config": self.config,
            "embedding_dimensions": getattr(self.model, "embedding_dimensions", None),
            "num_continuous": getattr(self.model, "num_continuous", None),
            "target_scaler": getattr(self.model, "target_scaler", None),
            "feature_scaler": getattr(self.model, "feature_scaler", None),
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"[Checkpoint] Saved checkpoint to: {checkpoint_path}")

    @torch.no_grad()
    def evaluate(self, loader):
        if self.ema_enabled and self.ema is not None:
            self.ema.apply_to(self.model)
        
        all_preds = []
        all_targets = []
        running_loss = torch.tensor(0.0, device=self.device)
        
        num_batches = 0
        for categorical_features, continuous_features, targets, lengths in loader:
            categorical_features = categorical_features.to(self.device, non_blocking=True)
            continuous_features  = continuous_features.to(self.device, non_blocking=True)
            targets              = targets.to(self.device, non_blocking=True)
            lengths              = lengths.to(self.device, non_blocking=True)
            
            preds         = self.model(categorical_features, continuous_features, lengths)
            target_tensor = targets.view(-1)
            loss          = self.criterion(preds, target_tensor).mean()
            running_loss += loss.detach()
            
            num_batches += 1
            all_preds.append(preds.cpu())
            all_targets.append(target_tensor.cpu())
             
        average_loss = (running_loss / max(num_batches, 1)).item()

        all_preds_tensor = torch.cat(all_preds, dim=0).numpy()
        all_targets_tensor = torch.cat(all_targets, dim=0).numpy()
        
        den_targets = np.expm1(self.target_scaler.inverse_transform(all_targets_tensor.reshape(-1, 1)))
        den_preds   = np.expm1(self.target_scaler.inverse_transform(all_preds_tensor.reshape(-1, 1)))

        den_preds   = np.clip(den_preds, 0, None)
        den_targets = np.clip(den_targets, 0, None)

        metrics = self.compute_metrics(den_targets, den_preds, average_loss)
        if self.ema_enabled and self.ema is not None:
            self.ema.restore(self.model)
        return metrics
    
    def fit(self):
        torch.cuda.empty_cache()
        
        self.logger.section("Model Training")
        self.logger.subsection("Training Progress")
        
        best_rmse = float('inf')
        best_model_state = None
        
        patience_counter = 0
        for epoch in range(1, self.config.training.epochs + 1):
            train_loss         = self.train_epoch(epoch=epoch)
            validation_metrics = self.evaluate(self.validation_loader)
            validation_rmse    = validation_metrics['rmse']
            
            self.scheduler_manager.step(epoch=True)
            
            self.logger.info(
                f"Epoch {epoch}:\n"
                f"  Train Loss = {train_loss:.4f}\n"
                f"  Val   Loss = {validation_metrics['loss']:.4f}\n"
                f"  MAE        = {validation_metrics['mae']:.4f} | RMSE = {validation_rmse:.4f}\n"
                f"  R2         = {validation_metrics['r2']:.4f} | StdErr = {validation_metrics['std']:.4f}\n"
                f"  P50 = {validation_metrics['p50']:.4f} | P90 = {validation_metrics['p90']:.4f} | P95 = {validation_metrics['p95']:.4f} \n"
                f"  MedAE     = {validation_metrics['medae']:.4f} | MaxErr = {validation_metrics['max_error']:.4f}\n"
                f"error bins (%): [0-5]={validation_metrics['error_0_5']:.2f}, [5-10]={validation_metrics['error_5_10']:.2f}, [10-15]={validation_metrics['error_10_15']:.2f}, [15-20]={validation_metrics['error_15_20']:.2f}, [20-25]={validation_metrics['error_20_25']:.2f}, >25={validation_metrics['error_above_25']:.2f}\n"
                f"target bins (MAE): [0-5]={validation_metrics['target_0_5']:.4f}, [5-10]={validation_metrics['target_5_10']:.4f}, [10-15]={validation_metrics['target_10_15']:.4f}, [15-20]={validation_metrics['target_15_20']:.4f}, [20-25]={validation_metrics['target_20_25']:.4f}, >25={validation_metrics['target_above_25']:.4f}\n"
            )
            
            if validation_rmse < best_rmse:
                best_rmse = validation_rmse
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                self.logger.info(f" New Best Model: RMSE={validation_rmse:.4f}")
                self.save_checkpoint(validation_metrics)
            else:
                patience_counter += 1
                if patience_counter >= self.config.training.patience:
                    self.logger.warning(f"[Early Stopping] Training halted at epoch {epoch} (patience={self.config.training.patience}). Best RMSE: {best_rmse:.4f}")
                    break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.model