import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import median_absolute_error
import numpy as np
import itertools
import copy
from tqdm.auto import tqdm
from .logger import ShapeLogger
from pathlib import Path


class Loss(nn.Module):
    def __init__(
        self,
        config,
        logger
    ):
        super().__init__()
        self.logger           = logger
        self.huber_delta      = config.huber_delta
        self.quantiles        = torch.tensor(config.quantiles)
        self.quantile_weight  = config.quantile_weight
        self.threshold_weight = config.threshold_weight
        self.thresholds       = config.thresholds

        self.logger.section(f"[Loss Function]")
        self.logger.subsection(f"Huber Loss             : Delta = {self.huber_delta}")
        self.logger.subsection(f"Quantile Loss          : Quantiles = {self.quantiles.tolist()}, Weight = {self.quantile_weight}")
        self.logger.subsection(f"Threshold-Focused Loss : Thresholds = {self.thresholds}, Weight = {self.threshold_weight}\n")

        
    def huber_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        diff      = preds - targets
        abs_diff  = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=self.huber_delta)
        linear    = abs_diff - quadratic
        loss      = 0.5 * quadratic.pow(2) + self.huber_delta * linear
        
        return loss
    
    def quantile_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        quantiles = self.quantiles.to(preds.device)
        errors    = targets.unsqueeze(-1) - preds.unsqueeze(-1)
        loss      = torch.max(quantiles * errors, (quantiles - 1) * errors)
        
        return loss.mean(dim=-1)
    
    def threshold_focused_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros_like(preds)
        for threshold in self.thresholds:
            pred_above          = (preds > threshold).float()
            target_above        = (targets > threshold).float()
            threshold_proximity = torch.exp(-torch.abs(targets - threshold) / 5.0)
            misclassification   = torch.abs(pred_above - target_above)
            loss               += misclassification * threshold_proximity * torch.abs(preds - targets)
        
        return loss
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        huber     = self.huber_loss(preds, targets)
        quantile  = self.quantile_loss(preds, targets)
        threshold = self.threshold_focused_loss(preds, targets)
        
        total_loss = (
            huber + 
            self.quantile_weight * quantile + 
            self.threshold_weight * threshold
        )
        
        components = {
            'total'     : total_loss.detach().mean().item(),
            'huber'     : huber.detach().mean().item(),
            'quantile'  : quantile.detach().mean().item(),
            'threshold' : threshold.detach().mean().item()
        }
        
        return total_loss, components


class Warmup:
    def __init__(self, optimizer, config, logger, tracker=None):
        self.optimizer           = optimizer
        self.warmup_steps        = config.training.warmup_steps
        self.warmup_start_factor = config.training.warmup_start_factor
        self.enabled             = config.training.warmup_enabled
        self.base_lrs            = [group['lr'] for group in optimizer.param_groups]
        self.logger              = logger
        self.tracker             = tracker

        self.logger.section(f"\n [Warmup]")
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
        elif not self.warmup_finished:
            factor = 1.0
            self._apply_warmup_factor(factor)
            
            if self.warmup_finished == False:
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

        def mean_target_range(low, high=None):
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
        self.logger.subsection(f"Enabled  : {self.enabled}")
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
            "scaler_state_dict"    : trainer.scaler.state_dict() if trainer.scaler else None,
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
        
        self.device = config.training.device
        self.model  = model.to(self.device)

        self.shape_logger   = ShapeLogger(model = self.model, logger = self.logger).attach()
    
        self.train_loader      = train_loader
        self.validation_loader = validation_loader
        self.target_scaler     = target_scaler

        self.optimizer_manager = Optimizer(model=self.model, config=self.config, logger=self.logger)
        self.optimizer         = self.optimizer_manager.build()
    
        self.warmup         = Warmup(optimizer=self.optimizer, config=self.config, logger=self.logger, tracker=self.tracker)
        self.scheduler      = Scheduler(optimizer=self.optimizer, warmup=self.warmup, config=self.config, logger=self.logger, tracker=self.tracker)
        self.ema            = EMA(self.model, config=self.config, logger=self.logger, tracker=self.tracker)
        self.early_stopping = EarlyStopping(self.config, self.logger)
        self.criterion      = Loss(self.config, self.logger)
        self.scaler         = GradScaler() if self.config.training.mixed_precision else None
        
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

    def forward(self, categorical_features, continuous_features, lengths, targets):
        with autocast(device_type=self.device.type, enabled=self.config.training.mixed_precision):
            preds = self.model(categorical_features, continuous_features, lengths)
            target_values = targets.view(-1)
            batch_loss, loss_components = self.loss(preds, target_values)
            batch_loss_for_backward = batch_loss / self.grad_accum_steps
            self.shape_logger.detach()

        return preds, batch_loss_for_backward, batch_loss, loss_components
                
    def loss(self, preds: torch.Tensor, targets: torch.Tensor):
        sample_losses, components = self.criterion(preds, targets)
        sample_losses = sample_losses.view(-1)

        den_targets = np.expm1(self.target_scaler.inverse_transform(targets.detach().cpu().numpy().reshape(-1, 1))).reshape(-1)
        den_targets = torch.from_numpy(den_targets).to(self.device).float()
        
        mean_den    = den_targets.mean().clamp_min(1e-8)
        weights     = 1.0 + self.high_target_weight * (den_targets / mean_den)
        weighted_losses = sample_losses * weights
        batch_loss = weighted_losses.mean()

        return batch_loss, components

    def backward(self, loss, step: bool):
        if self.scaler:
            self.scaler.scale(loss).backward()
            if step:
                self.warmup.step()
                self.scaler.unscale_(self.optimizer)
                self.tracker.log_gradients(self.model, step=self.step + 1, max_grad_norm=self.config.training.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                self.ema.update(self.model, self.step)
                self.tracker.log_optimizer(self.optimizer, step=self.step)
        else:
            loss.backward()
            if step:
                self.warmup.step()
                self.tracker.log_gradients(self.model, step=self.step + 1, max_grad_norm=self.config.training.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                self.ema.update(self.model, self.step)
                self.tracker.log_optimizer(self.optimizer, step=self.step)

    def train_epoch(self):
        self.model.train()
        epoch_loss_sum = torch.tensor(0.0, device=self.device)
        
        if self.config.overfit.overfit_single_batch:
            single_batch = next(iter(self.train_loader))
            batch_iterable = itertools.repeat(single_batch, len(self.train_loader))
            loop = tqdm(batch_iterable, desc=f"Train Epoch {self.epoch} (Overfit Single Batch)", total=len(self.train_loader))
        else:
            loop = tqdm(self.train_loader, desc=f"Train Epoch {self.epoch}")

        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(loop):
            categorical_features, continuous_features, targets, lengths = batch

            categorical_features = categorical_features.to(self.device, non_blocking=True)
            continuous_features  = continuous_features.to(self.device, non_blocking=True)
            targets              = targets.to(self.device, non_blocking=True)
            lengths              = lengths.to(self.device, non_blocking=True)

            _, batch_loss_for_backward, batch_loss, _ = self.forward(
                categorical_features,
                continuous_features,
                lengths,
                targets,
            )

            is_last = (batch_idx + 1) == len(self.train_loader)
            should_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or is_last
            self.backward(batch_loss_for_backward, step=should_step)
            epoch_loss_sum += batch_loss.detach()
            self.batch += 1
            
    @torch.no_grad()
    def evaluate(self, loader, phase="Validation"):
        self.ema.apply_to(self.model)
        
        self.model.eval()
        all_preds    = []
        all_targets  = []
        eval_loss_sum = torch.tensor(0.0, device=self.device)
        
        num_batches = 0
        for categorical_features, continuous_features, targets, lengths in loader:
            categorical_features = categorical_features.to(self.device, non_blocking=True)
            continuous_features  = continuous_features.to(self.device, non_blocking=True)
            targets              = targets.to(self.device, non_blocking=True)
            lengths              = lengths.to(self.device, non_blocking=True)
            
            preds, _, batch_loss, loss_components = self.forward(
                categorical_features,
                continuous_features,
                lengths,
                targets,
            )
            eval_loss_sum += batch_loss.detach()
            
            num_batches += 1
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
       
        average_loss = (eval_loss_sum / max(num_batches, 1)).item()

        all_preds_tensor   = torch.cat(all_preds, dim=0).numpy()
        all_targets_tensor = torch.cat(all_targets, dim=0).numpy()
        
        den_targets = np.expm1(self.target_scaler.inverse_transform(all_targets_tensor.reshape(-1, 1)))
        den_preds   = np.expm1(self.target_scaler.inverse_transform(all_preds_tensor.reshape(-1, 1)))

        den_preds   = np.clip(den_preds, 0, None)
        den_targets = np.clip(den_targets, 0, None)

        metrics = self.metrics.compute(den_targets, den_preds, average_loss, phase=phase, epoch=self.epoch)
                
        self.ema.restore(self.model)
        
        return metrics
    
    def fit(self):
        self.early_stopping.reset()

        for epoch in range(1, self.config.training.epochs + 1):
            self.train_epoch()
            training_metrics   = self.evaluate(self.train_loader,      phase="Training")
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