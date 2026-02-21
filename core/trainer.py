import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import median_absolute_error
import numpy as np
import itertools
import copy
from tqdm.auto import tqdm
from .logger import ShapeLogger
from pathlib import Path
from torch.amp import GradScaler


class Loss(nn.Module):
    def __init__(
        self,
        config,
        logger,
        target_scaler=None,
    ):
        super().__init__()
        self.logger                    = logger
        self.target_scaler             = target_scaler
        self.huber_delta_raw           = float(config.loss.huber_delta)
        self.threshold_weight          = config.loss.threshold_weight
        self.thresholds_raw            = [float(value) for value in config.loss.thresholds]
        self.threshold_proximity_raw   = float(config.loss.threshold_proximity_width)
        
        self.huber_delta               = self.norm(self.huber_delta_raw) - self.norm(0.0)
        self.thresholds                = [self.norm(value) for value in self.thresholds_raw]
        self.threshold_proximity_width = max(self.norm(self.threshold_proximity_raw) - self.norm(0.0), 1e-6)

        self.logger.section(f"[Loss Function]")
        self.logger.subsection(f"Huber Loss             : Delta(raw)      = {self.huber_delta_raw}, Delta(normed) = {self.huber_delta:.6f}")
        self.logger.subsection(f"Threshold-Focused Loss : Thresholds(raw) = {self.thresholds_raw}, Thresholds(normed) = {[round(value, 6) for value in self.thresholds]}, Weight = {self.threshold_weight}")
        self.logger.subsection(f"Threshold Proximity    : Width(raw)      = {self.threshold_proximity_raw}, Width(normed) = {self.threshold_proximity_width:.6f}\n")

    def norm(self, value: float) -> float:
        if self.target_scaler is None:
            return float(value)

        value_arr    = np.array([[max(float(value), 0.0)]], dtype=np.float64)
        value_log    = np.log1p(value_arr)
        value_normed = self.target_scaler.transform(value_log)
        
        return float(value_normed.reshape(-1)[0])

    def huber_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        diff      = preds - targets
        abs_diff  = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=self.huber_delta)
        linear    = abs_diff - quadratic
        loss      = 0.5 * quadratic.pow(2) + self.huber_delta * linear
        
        return loss
    
    def threshold_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros_like(preds)
        for threshold in self.thresholds:
            pred_above          = (preds > threshold).float()
            target_above        = (targets > threshold).float()
            threshold_proximity = torch.exp(-torch.abs(targets - threshold) / self.threshold_proximity_width)
            misclassification   = torch.abs(pred_above - target_above)
            loss               += misclassification * threshold_proximity * torch.abs(preds - targets)
        
        return loss
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        huber     = self.huber_loss(preds, targets)
        threshold = self.threshold_loss(preds, targets)
        total_loss = huber + self.threshold_weight * threshold
        return total_loss, huber, threshold


class Warmup:
    def __init__(self, optimizer, config, logger, tracker=None):
        self.optimizer           = optimizer
        self.warmup_steps        = config.training.warmup_steps
        self.warmup_start_factor = config.training.warmup_start_factor
        self.enabled             = config.training.warmup_enabled
        self.base_lrs            = [group['lr'] for group in optimizer.param_groups]
        self.logger              = logger
        self.tracker             = tracker

        self.logger.info("\n")
        self.logger.section(f"[Warmup]")
        self.logger.subsection(f" Enabled      : {self.enabled}")
        self.logger.subsection(f" Warmup Steps : {self.warmup_steps}")
        self.logger.subsection(f" Start Factor : {self.warmup_start_factor} \n")

        self.current_step        = 0
        self.warmup_finished     = False
        
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
            factor   = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
            self._apply_warmup_factor(factor)
            if self.tracker:
                self.tracker.log_scalar("warmup/factor", factor, step=self.current_step)
        elif not self.warmup_finished:
            factor = 1.0
            self._apply_warmup_factor(factor)
            
            self.logger.section(f"[Warmup]")
            self.logger.subsection(f"Warmup finished at step {self.current_step}. Learning rates set to base values \n")
            self.warmup_finished = True
            if self.tracker:
                self.tracker.log_scalar("warmup/factor", factor, step=self.current_step)
    
    def is_finished(self) -> bool:
        return self.warmup_finished or not self.enabled or self.warmup_steps <= 0


class Scheduler:
    def __init__(self, optimizer, config, warmup: Warmup = None, logger=None, tracker=None):
        self.optimizer = optimizer
        self.warmup    = warmup
        self.config    = config
        self.logger    = logger
        self.tracker   = tracker
        
        self.scheduler = CosineAnnealingLR(
            optimizer = self.optimizer, 
            T_max     = self.config.scheduler.t_max, 
            eta_min   = self.config.scheduler.eta_min
        )

        self.logger.section(f"[Scheduler]")
        self.logger.subsection(f" Type    : CosineAnnealingLR")
        self.logger.subsection(f" T_max   : {self.config.scheduler.t_max}")
        self.logger.subsection(f" Eta_min : {self.config.scheduler.eta_min} \n")

       
    def step(self, epoch: bool = True) -> None:
        if self.warmup and not self.warmup.is_finished():
            return
        
        if epoch:
            self.scheduler.step()
            if self.tracker:
                self.tracker.log_dict(
                    "scheduler/lr",
                    {group.get("name", f"group_{i}"): group["lr"] for i, group in enumerate(self.optimizer.param_groups)},
                    step=max(0, self.scheduler.last_epoch),
                )
    
    def state_dict(self) -> dict:
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict: dict) -> None:
        self.scheduler.load_state_dict(state_dict)


class EMA:
    def __init__(self, model: nn.Module, config, logger, tracker=None):
        self.logger       = logger
        self.tracker      = tracker
        self.enabled      = config.ema.use_ema
        self.decay        = config.ema.ema_decay
        self.warmup_steps = config.ema.ema_warmup_steps
        self.shadow       = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad} if self.enabled else {}

        self.logger.section(f"[EMA] Exponential Moving Average")
        self.logger.subsection(f"Enabled      : {self.enabled}")
        self.logger.subsection(f"Decay        : {self.decay}")
        self.logger.subsection(f"Warmup Steps : {self.warmup_steps}\n")

        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module, global_step: int) -> None:
        if not self.enabled:
            return
        if global_step < self.warmup_steps:
            return
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        self.backup = {}
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.backup[name] = param.detach().clone()
            param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        for name, param in model.named_parameters():
            if name in self.backup:
                param.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "warmup_steps": self.warmup_steps,
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict) -> None:
        self.enabled      = state.get("enabled", self.enabled)
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
        self.decay        = state.get("decay", self.decay)
        self.shadow       = state.get("shadow", self.shadow)
        self.backup       = {}


class Metrics:
    def __init__(self, tracker):
        self.tracker = tracker
    
    def compute(self, den_targets, den_preds, average_loss=None, phase="Validation", epoch=None) -> dict:
        den_preds   = np.asarray(den_preds).flatten()
        den_targets = np.asarray(den_targets).flatten()

        mae       = float(np.mean(np.abs(den_preds - den_targets)))
        rmse      = float(np.sqrt(np.mean((den_preds - den_targets) ** 2)))
        ss_res    = float(np.sum((den_targets - den_preds) ** 2))
        ss_tot    = float(np.sum((den_targets - np.mean(den_targets)) ** 2))
        r2        = float(1 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
        std       = float(np.std(den_targets - den_preds))
        medae     = float(median_absolute_error(den_targets, den_preds))
        max_error = float(np.max(np.abs(den_targets - den_preds)))
        p50       = np.percentile(np.abs(den_targets - den_preds), 50)
        p90       = np.percentile(np.abs(den_targets - den_preds), 90)
        p95       = np.percentile(np.abs(den_targets - den_preds), 95)

        abs_err = np.abs(den_targets - den_preds)
        error_bin_0_5      = float(np.mean(abs_err <= 5) * 100)
        error_bin_5_10     = float(np.mean((abs_err > 5) & (abs_err <= 10)) * 100)
        error_bin_10_15    = float(np.mean((abs_err > 10) & (abs_err <= 15)) * 100)
        error_bin_15_20    = float(np.mean((abs_err > 15) & (abs_err <= 20)) * 100)
        error_bin_20_25    = float(np.mean((abs_err > 20) & (abs_err <= 25)) * 100)
        error_bin_above_25 = float(np.mean(abs_err > 25) * 100)

        def mean_target_range(low, high):
            mask = (den_targets > low) & (den_targets <= high)
            vals = abs_err[mask]
            return float(np.mean(vals)) if len(vals) > 0 else float('nan')

        metrics = {
            'mae'       : mae,
            'rmse'      : rmse,
            'r2'        : r2,
            'std'       : std,
            'p50'       : p50,
            'p90'       : p90,
            'p95'       : p95,
            'medae'     : medae,
            'max_error' : max_error,
            'loss'      : average_loss,

            'error_0_5'      : error_bin_0_5,
            'error_5_10'     : error_bin_5_10,
            'error_10_15'    : error_bin_10_15,
            'error_15_20'    : error_bin_15_20,
            'error_20_25'    : error_bin_20_25,
            'error_above_25' : error_bin_above_25,

            'target_0_5'      : mean_target_range(0, 5),
            'target_5_10'     : mean_target_range(5, 10),
            'target_10_15'    : mean_target_range(10, 15),
            'target_15_20'    : mean_target_range(15, 20),
            'target_20_25'    : mean_target_range(20, 25),
            'target_above_25' : mean_target_range(25, 30),
        }

        self.tracker.log_dict(f"{phase}/metrics", metrics, step=epoch)
        return metrics


class EarlyStopping:
    def __init__(self, config, logger):
        self.logger   = logger
        self.enabled  = config.early_stopping.enabled
        self.patience = config.early_stopping.patience
        self.mode     = config.early_stopping.mode
        self.reset()

        self.logger.section(f"[Early Stopping]")
        self.logger.subsection(f"Enabled   : {self.enabled}")
        self.logger.subsection(f"Patience  : {self.patience}")
        self.logger.subsection(f"Mode      : {self.mode}\n")

    def reset(self) -> None:    
        self.best_metric = float('inf') if self.mode == "min" else float('-inf')
        self.best_model_state = None
        self.counter = 0

    def _is_improvement(self, metric: float) -> bool:
        if self.mode == "min":
            return metric < self.best_metric
        return metric > self.best_metric

    def step(self, metric: float, model_state: dict) -> bool:
        if self._is_improvement(metric):
            self.best_metric = metric
            self.best_model_state = copy.deepcopy(model_state)
            self.counter = 0
            return True

        self.counter += 1
        return False

    def should_stop(self) -> bool:
        if not self.enabled:
            return False
        return self.counter >= self.patience


class Optimizer:
    def __init__(self, model: nn.Module, config, logger):
        self.model  = model
        self.config = config
        self.logger = logger
   
    def build(self):
        layerwise = self.config.layerwise
        param_groups = []

        optimizer_map = {
            "tokenizer": {
                "name"         : layerwise.tokenizer_name,
                "lr"           : layerwise.tokenizer_lr,
                "weight_decay" : layerwise.tokenizer_weight_decay,
            },
            "invoice_encoder": {
                "name"         : layerwise.invoice_encoder_name,
                "lr"           : layerwise.invoice_encoder_lr,
                "weight_decay" : layerwise.invoice_encoder_weight_decay,
            },
            "sequence_encoder": {
                "name"         : layerwise.sequence_encoder_name,
                "lr"           : layerwise.sequence_encoder_lr,
                "weight_decay" : layerwise.sequence_encoder_weight_decay,
            },
            "temporal_attention": {
                "name"         : layerwise.cross_attention_name,
                "lr"           : layerwise.cross_attention_lr,
                "weight_decay" : layerwise.cross_attention_weight_decay,
            },
            "head_days": {
                "name"         : layerwise.head_name,
                "lr"           : layerwise.head_lr,
                "weight_decay" : layerwise.head_weight_decay,
            },
        }

        for module_name, hyperparams in optimizer_map.items():
            module = getattr(self.model, module_name, None)
            if module is None:
                continue

            group_weight_decay = hyperparams["weight_decay"]
            if group_weight_decay is None:
                group_weight_decay = self.config.training.weight_decay

            param_groups.append({
                "name"         : hyperparams["name"],
                "params"       : module.parameters(),
                "lr"           : hyperparams["lr"],
                "weight_decay" : group_weight_decay,
            })

        optimizer = optim.AdamW(param_groups)

        self.logger.section(f"[Optimizer]:")
        for module_name, hyperparams in optimizer_map.items():
            wd = hyperparams["weight_decay"]
            wd = self.config.training.weight_decay if wd is None else wd
            self.logger.subsection(f" [{hyperparams['name']}] ({module_name}) : lr={hyperparams['lr']}, weight_decay={wd}")

        return optimizer


class Checkpoint:
    def __init__(self, logger):
        self.logger          = logger
        self.checkpoint_path = os.path.join(self.logger.log_dir, "model", "checkpoint.pt")
        os.makedirs(os.path.join(self.logger.log_dir, "model"), exist_ok=True)

        self.logger.section(f"[Checkpoint]")
        self.logger.subsection(f"Checkpoint Path: {self.checkpoint_path} \n")

    def save(self, trainer, metrics: dict = None) -> None:
        checkpoint = {
            "model_state_dict"     : trainer.model.state_dict(),
            "optimizer_state_dict" : trainer.optimizer.state_dict(),
            "scheduler_state_dict" : trainer.scheduler.state_dict(),
            "ema_state_dict"       : trainer.ema.state_dict(),
            "epoch"                : trainer.epoch,
            "batch"                : trainer.batch,
            "best_metrics"         : metrics if metrics is not None else {},
            "config"               : trainer.config,
            "embedding_dimensions" : trainer.model.embedding_dimensions,
            "num_continuous"       : trainer.model.num_continuous,
            "target_scaler"        : trainer.model.target_scaler,
            "feature_scaler"       : trainer.model.feature_scaler,
        }

        torch.save(checkpoint, self.checkpoint_path)
        self.logger.info(f"[Checkpoint] Saved checkpoint to: {self.checkpoint_path}")


class Trainer: 
    def __init__(self, model, train_loader, validation_loader, target_scaler, logger, config, tracker):
        self.logger  = logger
        self.config  = config
        self.tracker = tracker
        
        self.logger.section("[Trainer Initialization]")
        self.logger.subsection(f"[GPU] Name: {torch.cuda.get_device_name(0)}")
        self.logger.subsection(f"[GPU] Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        self.logger.subsection(f"[GPU] CUDA Version: {torch.version.cuda} \n")
        
        self.device = torch.device(config.training.device)
        self.model  = model.to(self.device)

        self.shape_logger   = ShapeLogger(model = self.model, logger=self.logger).attach()
    
        self.train_loader      = train_loader
        self.validation_loader = validation_loader
        self.target_scaler     = target_scaler

        self.optimizer_manager = Optimizer(model=self.model, config=self.config, logger=self.logger)
        self.optimizer         = self.optimizer_manager.build()
    
        self.warmup         = Warmup(optimizer=self.optimizer, config=self.config, logger=self.logger, tracker=self.tracker)
        self.scheduler      = Scheduler(optimizer=self.optimizer, warmup=self.warmup, config=self.config, logger=self.logger, tracker=self.tracker)
        self.ema            = EMA(self.model, config=self.config, logger=self.logger, tracker=self.tracker)
        self.early_stopping = EarlyStopping(self.config, self.logger)
        self.criterion      = Loss(self.config, self.logger, target_scaler=self.target_scaler)
        
        self.metrics        = Metrics(tracker=self.tracker)
        self.checkpoint     = Checkpoint(self.logger)

        self.grad_accum_steps = max(1, self.config.training.grad_accum_steps)
        self.logger.section(f"[Grad Accumulation]")
        self.logger.subsection(f"Enabled         : {self.grad_accum_steps > 1}")
        self.logger.subsection(f"Steps           : {self.grad_accum_steps}")
        self.logger.subsection(f"Effective batch : {self.train_loader.batch_size * self.grad_accum_steps} \n")

        self.high_target_weight = self.config.training.high_target_weight
        self.logger.section(f"[High Target Weight]")
        self.logger.subsection(f"Enabled : {self.high_target_weight > 0}")
        self.logger.subsection(f"Weight  : {self.high_target_weight} \n")

        self.epoch = 0
        self.batch = 0
        self.step  = 0

        self.use_amp = getattr(self.config.training, 'mixed_precision', False) and self.device.type == 'cuda'
        self.scaler  = GradScaler('cuda', enabled=self.use_amp)

        self._scaler_mean = torch.tensor(float(target_scaler.mean_[0]),  device=self.device, dtype=torch.float32)
        self._scaler_std  = torch.tensor(float(target_scaler.scale_[0]), device=self.device, dtype=torch.float32)

        self.logger.section(f"[Mixed Precision (AMP)]")
        self.logger.subsection(f"Enabled : {self.use_amp}\n")

    def forward(self, categorical_features, continuous_features, lengths, targets):
        self._validate_inputs(categorical_features, continuous_features, lengths)

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            preds = self.model(categorical_features, continuous_features, lengths)

            if self.step % 500 == 0 and (torch.isnan(preds).any() or torch.isinf(preds).any()):
                nan_count = int(torch.isnan(preds).sum())
                inf_count = int(torch.isinf(preds).sum())
                self.logger.warning(f"[Forward] NaN/Inf in predictions at step {self.step}: {nan_count} NaN(s), {inf_count} Inf(s)")

            target_values = targets.view(-1)
            batch_loss, loss_components = self.loss(preds, target_values)

        self.shape_logger.detach()
        return preds, batch_loss, loss_components
                
    def loss(self, preds: torch.Tensor, targets: torch.Tensor):
        sample_losses, huber_losses, threshold_losses = self.criterion(preds, targets)
        sample_losses    = sample_losses.view(-1)
        huber_losses     = huber_losses.view(-1)
        threshold_losses = threshold_losses.view(-1)

        den_targets = torch.expm1(targets * self._scaler_std + self._scaler_mean)

        mean_den        = den_targets.mean().clamp_min(1e-8)
        weights         = 1.0 + self.high_target_weight * (den_targets / mean_den).clamp(max=5.0)
        batch_loss      = (sample_losses * weights).mean()

        components = {
            'total'     : batch_loss.detach().item(),
            'huber'     : (huber_losses * weights).mean().detach().item(),
            'threshold' : (self.criterion.threshold_weight * threshold_losses * weights).mean().detach().item(),
        }

        return batch_loss, components

    def _validate_inputs(
        self,
        categorical_features: torch.Tensor,
        continuous_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        if self.step % 500 != 0:
            return
        if torch.isnan(continuous_features).any() or torch.isinf(continuous_features).any():
            self.logger.warning(f"[Forward] NaN/Inf in continuous_features at step {self.step}")
        if torch.isnan(categorical_features.float()).any():
            self.logger.warning(f"[Forward] NaN in categorical_features at step {self.step}")
        if (lengths == 0).any():
            self.logger.warning(f"[Forward] Zero-length sequences at step {self.step}: {(lengths == 0).sum().item()} sample(s)")

    def _sanitize_predictions(self, preds: np.ndarray, phase: str):
        nan_count = int(np.isnan(preds).sum())
        inf_count = int(np.isinf(preds).sum())
        if nan_count > 0 or inf_count > 0:
            raise ValueError(
                f"[{phase}] Predictions contain {nan_count} NaN(s) and {inf_count} Inf(s) — "
                f"fix the root cause before proceeding."
            )
      
    def backward(self, loss, step: bool, accum_steps_in_window: int):
        loss = loss / max(accum_steps_in_window, 1)
        self.scaler.scale(loss).backward()
        if step:
            self.scaler.unscale_(self.optimizer)
            self.warmup.step()
            self.tracker.log_gradients(self.model, step=self.step + 1, max_grad_norm=self.config.training.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            self.ema.update(self.model, self.step)
            self.tracker.log_optimizer(self.optimizer, step=self.step)

    def train_epoch(self, data_loader):
        self.model.train()
        if hasattr(self.train_loader, 'dataset'):
            self.train_loader.dataset.training_mode = True
        
        loop = tqdm(data_loader, desc=f"Train Epoch {self.epoch}", total=len(data_loader))

        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(loop):
            categorical_features, continuous_features, targets, lengths = batch

            categorical_features = categorical_features.to(self.device, non_blocking=True)
            continuous_features  = continuous_features.to(self.device, non_blocking=True)
            targets              = targets.to(self.device, non_blocking=True)
            lengths              = lengths.to(self.device, non_blocking=True)

            _, batch_loss, loss_components = self.forward(
                categorical_features,
                continuous_features,
                lengths,
                targets,
            )
            self.tracker.log_dict("train/loss_components", loss_components, step=self.step)

            is_last = (batch_idx + 1) == len(data_loader)

            window_start          = (batch_idx // self.grad_accum_steps) * self.grad_accum_steps
            window_end            = min(window_start + self.grad_accum_steps, len(data_loader))
            accum_steps_in_window = window_end - window_start

            should_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or is_last
            self.backward(batch_loss, step=should_step, accum_steps_in_window=accum_steps_in_window)
            self.batch += 1

        if hasattr(self.train_loader, 'dataset'):
            self.train_loader.dataset.training_mode = False
            
    @torch.no_grad()
    def evaluate(self, loader, phase="Validation"):
        self.ema.apply_to(self.model)
        
        self.model.eval()
        all_preds    = []
        all_targets  = []
        eval_loss_sum = torch.tensor(0.0, device=self.device)
        
        num_batches = 0
        accumulated_components = {}
        for categorical_features, continuous_features, targets, lengths in loader:
            categorical_features = categorical_features.to(self.device, non_blocking=True)
            continuous_features  = continuous_features.to(self.device, non_blocking=True)
            targets              = targets.to(self.device, non_blocking=True)
            lengths              = lengths.to(self.device, non_blocking=True)
            
            preds, batch_loss, loss_components = self.forward(
                categorical_features,
                continuous_features,
                lengths,
                targets,
            )
            eval_loss_sum    += batch_loss.detach()

            for key, val in loss_components.items():
                accumulated_components[key] = accumulated_components.get(key, 0.0) + val
            
            num_batches += 1
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
       
        average_loss = (eval_loss_sum / max(num_batches, 1)).item()
        avg_components = {k: v / max(num_batches, 1) for k, v in accumulated_components.items()}
        self.tracker.log_dict(f"{phase}/loss_components", avg_components, step=self.epoch)

        all_preds_tensor   = torch.cat(all_preds, dim=0).numpy()
        all_targets_tensor = torch.cat(all_targets, dim=0).numpy()
        
        den_targets = np.expm1(self.target_scaler.inverse_transform(all_targets_tensor.reshape(-1, 1)))
        den_preds   = np.expm1(self.target_scaler.inverse_transform(all_preds_tensor.reshape(-1, 1)))

        self._sanitize_predictions(den_preds, phase)

        den_preds   = np.clip(den_preds, 0, None)
        den_targets = np.clip(den_targets, 0, None)

        metrics = self.metrics.compute(den_targets, den_preds, average_loss, phase=phase, epoch=self.epoch)
                
        self.ema.restore(self.model)
        
        return metrics
    
    def fit(self):
        self.early_stopping.reset()

        train_loader = self.train_loader
        train_loader_len = len(train_loader)
        if self.config.overfit.overfit_single_batch:
            single_batch = next(iter(train_loader))
            categorical_features, continuous_features, targets, lengths = single_batch
            k = min(self.config.overfit.overfit_sequence_count, categorical_features.size(0))
            single_batch = (categorical_features[:k], continuous_features[:k], targets[:k], lengths[:k])
            data_loader = [single_batch] * train_loader_len
            eval_train_loader = [single_batch]
            self.logger.warning(f"Overfitting mode enabled: training on a single batch ({k} sequences) repeated {train_loader_len} times.")
        else:
            data_loader = train_loader
            eval_train_loader = train_loader

        for epoch in range(1, self.config.training.epochs + 1):
            self.train_epoch(data_loader)
            training_metrics   = self.evaluate(eval_train_loader,      phase="Training")
            validation_metrics = self.evaluate(self.validation_loader, phase="Validation")
            
            self.tracker.log_scalar("train/average_loss",      training_metrics['loss'],   step=self.epoch)
            self.tracker.log_scalar("validation/average_loss", validation_metrics['loss'], step=self.epoch)

            validation_rmse    = validation_metrics['rmse']
            self.scheduler.step(epoch=True)
                  
            self.logger.info(
                f"Epoch {self.epoch}:\n"
                f"  Train Loss = {training_metrics['loss']:.4f}\n"
                f"  Val   Loss = {validation_metrics['loss']:.4f}\n"
                f"  MAE        = {validation_metrics['mae']:.4f} | RMSE = {validation_rmse:.4f}\n"
                f"  R2         = {validation_metrics['r2']:.4f} | StdErr = {validation_metrics['std']:.4f}\n"
                f"  P50 = {validation_metrics['p50']:.4f} | P90 = {validation_metrics['p90']:.4f} | P95 = {validation_metrics['p95']:.4f} \n"
                f"  MedAE     = {validation_metrics['medae']:.4f} | MaxErr = {validation_metrics['max_error']:.4f}\n"
                f"error bins (%)    : [0-5]={validation_metrics['error_0_5']:.2f}%, [5-10]={validation_metrics['error_5_10']:.2f}%, [10-15]={validation_metrics['error_10_15']:.2f}%, [15-20]={validation_metrics['error_15_20']:.2f}%, [20-25]={validation_metrics['error_20_25']:.2f}%, >25={validation_metrics['error_above_25']:.2f}%\n"
                f"target bins (MAE) : [0-5]={validation_metrics['target_0_5']:.4f}, [5-10]={validation_metrics['target_5_10']:.4f}, [10-15]={validation_metrics['target_10_15']:.4f}, [15-20]={validation_metrics['target_15_20']:.4f}, [20-25]={validation_metrics['target_20_25']:.4f}, >25={validation_metrics['target_above_25']:.4f}\n"
            )
            
            improved    = self.early_stopping.step(validation_rmse, self.model.state_dict())
            should_stop = self.early_stopping.should_stop()

            if improved:
                self.logger.info(f" New Best Model: RMSE={validation_rmse:.4f}")
                self.checkpoint.save(self, validation_metrics)

            if should_stop:
                self.logger.warning(f"[Early Stopping]")
                self.logger.warning(f"Training at : epoch {self.epoch}")
                self.logger.warning(f"Best RMSE   : {self.early_stopping.best_metric:.4f}\n")
                self.tracker.log_scalar("early_stopping/best_rmse", self.early_stopping.best_metric, step=self.epoch)
                self.tracker.log_scalar("early_stopping/best_epoch", self.epoch - self.early_stopping.patience, step=self.epoch)
                break

            self.epoch += 1
            
        self.shape_logger.save_markdown(path=Path(self.logger.log_dir) / "tensor_shape.md", sort_by_layer=True)
        self.model.load_state_dict(self.early_stopping.best_model_state)
        
        return self.model