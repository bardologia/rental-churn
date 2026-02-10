import logging
import os
import sys
from datetime import datetime
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from core.model import StochasticDepth
from core.model import (
    FourierFeatures,
    GRN,
    SwiGLU,
    RoPE,
    TransformerBlock,
    InvoiceEncoder,
    SequenceEncoder,
    CrossAttention,
    PredictionHead,
    FeatureTokenizer,
    )

class Logger:
    
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __init__(self, log_dir="logs", name="experiment", level="INFO", config=None):
        self.log_dir = log_dir
        self.name = name
        self.start_time = datetime.now()
        self.config = config
        if log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        log_level = self.LOG_LEVELS.get(str(level).upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        
        log_filename = f'{name}_{self.start_time.strftime("%Y%m%d_%H%M%S")}.log'
        if log_dir:
            file_handler = logging.FileHandler(os.path.join(self.log_dir, log_filename), encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(log_level)
            self.logger.addHandler(file_handler)
    
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        self.logger.addHandler(console_handler)
    
    def section(self, title: str):
        self.logger.info("")
        self.logger.info(f">>> {str(title).upper()}")
    
    def subsection(self, title: str):
        self.logger.info(f"  > {title}")
    
    def progress(self, current: int, total: int, prefix: str = "", suffix: str = ""):
        percentage = 100 * (current / float(total))
        self.logger.info(f"{prefix} [{current}/{total}] ({percentage:.1f}%) {suffix}")

    def debug(self, message: str):
        self.logger.debug(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
        
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
            
    def close(self):
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.info(f"[End] Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class TensorLogger:
    def __init__(self, model, include_types = (
    nn.Embedding,
    nn.Dropout,
    nn.Linear,
    nn.LayerNorm,
    nn.MultiheadAttention,
    StochasticDepth,
    FourierFeatures,
    GRN,
    SwiGLU,
    RoPE,
    TransformerBlock,
    InvoiceEncoder,
    SequenceEncoder,
    CrossAttention,
    PredictionHead,
    FeatureTokenizer,
)):
        self.model = model
        self.include_types = include_types
        self.records = []
        self.hooks = []
    
    def _hook(self, name):
        def fn(module, inputs, output):
            x = inputs[0]
            in_shape  = tuple(x.shape) if hasattr(x, "shape") else str(type(x))
            out_shape = tuple(output.shape) if hasattr(output, "shape") else str(type(output))
            self.records.append((name, module.__class__.__name__, in_shape, out_shape))
        return fn

    def attach(self):
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if isinstance(module, self.include_types):
                self.hooks.append(module.register_forward_hook(self._hook(name)))
        return self

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def clear(self):
        self.records.clear()
    
    def log_from_batch(self, batch, device="cpu"):
        categorical_features, continuous_features, targets, lengths = batch
        categorical_features = categorical_features.to(device)
        continuous_features = continuous_features.to(device)
        lengths = lengths.to(device)
        
        original_device = next(self.model.parameters()).device
        self.model.to(device)
        
        was_training = self.model.training
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(categorical_features, continuous_features, lengths)
        
        if was_training:
            self.model.train()
        
        self.model.to(original_device)
        
        self.detach()
        return self

    def to_markdown(self, title: str = "Shape Log", sort_by_layer: bool = False) -> str:
        rows = list(self.records)
        if sort_by_layer:
            rows.sort(key=lambda r: r[0])

        def s(x):
            return str(x)

        def layer_cell(name: str) -> str:
            return f"`{name}`"  # exatamente como será impresso

        col_names = ["Layer", "Type", "Input shape", "Output shape"]
        col_data = [
            [layer_cell(r[0]) for r in rows],
            [str(r[1]) for r in rows],
            [s(r[2]) for r in rows],
            [s(r[3]) for r in rows],
        ]

        widths = []
        for header, data in zip(col_names, col_data):
            widths.append(max([len(header)] + [len(v) for v in data]) if rows else len(header))

        def fmt_row(cells):
            return "| " + " | ".join(f"{c:<{w}}" for c, w in zip(cells, widths)) + " |"

        def fmt_sep():
            return "| " + " | ".join((":" + "-" * (w - 1)) if w > 1 else "-" for w in widths) + " |"

        lines = []
        lines.append(f"# {title}\n")
        lines.append(fmt_row(col_names))
        lines.append(fmt_sep())

        for (name, typ, ins, outs), layer_txt in zip(rows, col_data[0]):
            lines.append(fmt_row([layer_txt, str(typ), s(ins), s(outs)]))

        lines.append(f"\n**Records:** {len(rows)}")
        return "\n".join(lines)

    def save_markdown(self, path, title: str = "Shape Log", sort_by_layer: bool = False):
        md = self.to_markdown(title=title, sort_by_layer=sort_by_layer)
        Path(path).write_text(md, encoding="utf-8")


class ModelSummary:
    def __init__(self, model: nn.Module):
        self.model = model
        self.rows = []
        self.total_params = 0
     
    def _count_params(self, module: nn.Module):
        return sum(p.numel() for p in module.parameters())

    def run(self):
        self.total_params = 0

        for name, module in self.model.named_modules():
            if name == "":
                continue

            n_params = self._count_params(module)
            self.total_params += n_params
            
            self.rows.append((name, module.__class__.__name__, n_params))
    
    def to_markdown(self, title="Model Summary") -> str:
        if not self.rows:
            return f"# {title}\n\nNo layers found."
        
        rows_fmt = [(name, typ, f"{params:,}") for name, typ, params in self.rows]
        
        col1 = max(len("Layer"), *(len(name) for name, _, _ in rows_fmt))
        col2 = max(len("Type"), *(len(typ) for _, typ, _ in rows_fmt))
        col3 = max(len("Parameters"), *(len(p) for _, _, p in rows_fmt))

        def line(a, b, c):
            return f"| {a:<{col1}} | {b:<{col2}} | {c:>{col3}} |"

        table = []
        table.append(line("Layer", "Type", "Parameters"))
        table.append(f"| {'-'*col1} | {'-'*col2} | {'-'*col3} |")
        for name, typ, params in rows_fmt:
            table.append(line(name, typ, params))

        total = f"{self.total_params:,}"

        md = []
        md.append(f"# {title}\n")
        md.extend(table)
        md.append(f"\n**Total Parameters:** `{total}`")
        return "\n".join(md)

    def save_markdown(self, path: str, title: str = "Model Summary"):
        md = self.to_markdown(title=title)
        Path(path).write_text(md, encoding="utf-8")


class TensorBoardMonitor:
    """
    Classe especializada para monitorar o treinamento do modelo usando TensorBoard.
    Monitora métricas, gradientes, pesos, learning rates e cria visualizações detalhadas.
    """
    def __init__(self, log_dir: str = "runs", enabled: bool = True):
        self.enabled = enabled
        self.log_dir = log_dir
        self.writer = None
        self.global_step = 0
        
        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(self.log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=self.log_dir)
                print(f"[TensorBoard] Monitoring enabled at: {self.log_dir}")
                print(f"[TensorBoard] Run: tensorboard --logdir={self.log_dir}")
            except ImportError:
                print("[TensorBoard] Warning: tensorboard not installed. Monitoring disabled.")
                self.enabled = False
                self.writer = None
    
    def log_scalar(self, tag: str, value: float, step: int = None):
        """Log um valor escalar."""
        if not self.enabled or self.writer is None:
            return
        step = step if step is not None else self.global_step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int = None):
        """Log múltiplos valores escalares."""
        if not self.enabled or self.writer is None:
            return
        step = step if step is not None else self.global_step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int = None):
        """Log um histograma de valores."""
        if not self.enabled or self.writer is None:
            return
        step = step if step is not None else self.global_step
        if isinstance(values, torch.Tensor):
            self.writer.add_histogram(tag, values, step)
    
    def log_training_metrics(self, train_loss: float, epoch: int):
        """Log métricas de treinamento."""
        if not self.enabled:
            return
        self.log_scalar('Loss/Train', train_loss, epoch)
    
    def log_validation_metrics(self, metrics: dict, epoch: int):
        """Log métricas completas de validação."""
        if not self.enabled:
            return
        
        # Métricas principais
        main_metrics = ['loss', 'mae', 'rmse', 'r2', 'std', 'medae', 'max_error']
        for metric in main_metrics:
            if metric in metrics:
                self.log_scalar(f'Metrics/{metric.upper()}', metrics[metric], epoch)
        
        # Percentis
        percentiles = ['p50', 'p90', 'p95']
        for p in percentiles:
            if p in metrics:
                self.log_scalar(f'Percentiles/{p.upper()}', metrics[p], epoch)
        
        # Error bins (distribuição de erros) - usando log_scalar individual para evitar problemas com caracteres especiais
        error_bins = [
            ('0-5', metrics.get('error_0_5', 0)),
            ('5-10', metrics.get('error_5_10', 0)),
            ('10-15', metrics.get('error_10_15', 0)),
            ('15-20', metrics.get('error_15_20', 0)),
            ('20-25', metrics.get('error_20_25', 0)),
            ('above_25', metrics.get('error_above_25', 0))
        ]
        for bin_name, value in error_bins:
            self.log_scalar(f'ErrorBins_Distribution/{bin_name}', value, epoch)
        
        # Error bins como histograma consolidado para visualização melhor
        error_dict = {k: v for k, v in error_bins}
        self.log_scalars('ErrorBins/Overview', error_dict, epoch)
        
        # Target bins (MAE por range de target) - usando log_scalar individual
        target_bins = [
            ('0-5', metrics.get('target_0_5', float('nan'))),
            ('5-10', metrics.get('target_5_10', float('nan'))),
            ('10-15', metrics.get('target_10_15', float('nan'))),
            ('15-20', metrics.get('target_15_20', float('nan'))),
            ('20-25', metrics.get('target_20_25', float('nan'))),
            ('above_25', metrics.get('target_above_25', float('nan')))
        ]
        # Filtrar NaN values e log individualmente
        valid_target_bins = {}
        for bin_name, value in target_bins:
            if not (isinstance(value, float) and (value != value or value == float('inf') or value == float('-inf'))):
                self.log_scalar(f'TargetBins_MAE/{bin_name}', value, epoch)
                valid_target_bins[bin_name] = value
        
        # Target bins como overview
        if valid_target_bins:
            self.log_scalars('TargetBins/Overview', valid_target_bins, epoch)
    
    def log_learning_rates(self, optimizer, epoch: int):
        """Log learning rates de cada param group."""
        if not self.enabled:
            return
        
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.log_scalar(f'LearningRate/Group_{i}', lr, epoch)
    
    def log_gradients(self, model: nn.Module, epoch: int):
        """Log estatísticas dos gradientes."""
        if not self.enabled:
            return
        
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                
                # Log histograma de gradientes para camadas principais
                if any(key in name for key in ['weight', 'bias']):
                    self.log_histogram(f'Gradients/{name}', param.grad.data, epoch)
        
        total_norm = total_norm ** 0.5
        self.log_scalar('Gradients/Total_Norm', total_norm, epoch)
    
    def log_weights(self, model: nn.Module, epoch: int):
        """Log histogramas dos pesos do modelo."""
        if not self.enabled:
            return
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(key in name for key in ['weight', 'bias']):
                    self.log_histogram(f'Weights/{name}', param.data, epoch)
                
                self.log_scalar(f'Weights_Stats/{name}_mean', param.data.mean().item(), epoch)
                if param.data.numel() > 1:
                    self.log_scalar(f'Weights_Stats/{name}_std', param.data.std().item(), epoch)
    
    def log_predictions_distribution(self, predictions: torch.Tensor, targets: torch.Tensor, epoch: int, phase: str = 'Validation'):
        """Log distribuição de predições vs targets."""
        if not self.enabled:
            return
        
        self.log_histogram(f'{phase}/Predictions', predictions, epoch)
        self.log_histogram(f'{phase}/Targets', targets, epoch)
        
        # Calcular erros
        if isinstance(predictions, torch.Tensor):
            predictions_np = predictions.detach().cpu().numpy() if predictions.is_cuda else predictions.numpy()
        else:
            predictions_np = predictions
            
        if isinstance(targets, torch.Tensor):
            targets_np = targets.detach().cpu().numpy() if targets.is_cuda else targets.numpy()
        else:
            targets_np = targets
        
        errors = predictions_np - targets_np
        abs_errors = np.abs(errors)
        
        # Log histograma de erros
        self.log_histogram(f'{phase}/Errors', torch.from_numpy(errors), epoch)
        self.log_histogram(f'{phase}/Absolute_Errors', torch.from_numpy(abs_errors), epoch)
        
        # Log estatísticas
        self.log_scalar(f'{phase}/Predictions_Mean', predictions.mean().item() if isinstance(predictions, torch.Tensor) else float(predictions_np.mean()), epoch)
        self.log_scalar(f'{phase}/Predictions_Std', predictions.std().item() if isinstance(predictions, torch.Tensor) else float(predictions_np.std()), epoch)
        self.log_scalar(f'{phase}/Targets_Mean', targets.mean().item() if isinstance(targets, torch.Tensor) else float(targets_np.mean()), epoch)
        self.log_scalar(f'{phase}/Targets_Std', targets.std().item() if isinstance(targets, torch.Tensor) else float(targets_np.std()), epoch)
        
        # Log estatísticas de erro
        self.log_scalar(f'{phase}/Error_Mean', float(errors.mean()), epoch)
        self.log_scalar(f'{phase}/Error_Std', float(errors.std()), epoch)
        self.log_scalar(f'{phase}/Absolute_Error_Mean', float(abs_errors.mean()), epoch)
        self.log_scalar(f'{phase}/Absolute_Error_Median', float(np.median(abs_errors)), epoch)
    
    def log_batch_stats(self, batch_idx: int, loss: float, epoch: int):
        """Log estatísticas de batch individual."""
        if not self.enabled:
            return
        
        global_batch = epoch * 1000 + batch_idx  # Aproximação
        self.log_scalar('Batch/Loss', loss, global_batch)
    
    def log_gpu_memory(self, epoch: int):
        """Log uso de memória GPU."""
        if not self.enabled or not torch.cuda.is_available():
            return
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        self.log_scalar('GPU/Memory_Allocated_GB', allocated, epoch)
        self.log_scalar('GPU/Memory_Reserved_GB', reserved, epoch)
        self.log_scalar('GPU/Max_Memory_Allocated_GB', max_allocated, epoch)
    
    def log_model_graph(self, model: nn.Module, input_sample: tuple):
        """Log o grafo computacional do modelo."""
        if not self.enabled or self.writer is None:
            return
        
        try:
            self.writer.add_graph(model, input_sample)
        except Exception as e:
            print(f"[TensorBoard] Warning: Could not log model graph: {e}")
    
    def log_hyperparameters(self, config, metrics: dict):
        """Log hiperparâmetros e métricas finais."""
        if not self.enabled or self.writer is None:
            return
        
        try:
            # Extrair principais hiperparâmetros do config
            hparams = {
                'epochs': config.training.epochs,
                'batch_size': config.training.batch_size,
                'dropout': config.training.dropout,
                'weight_decay': config.training.weight_decay,
                'warmup_steps': config.training.warmup_steps,
                'max_grad_norm': config.training.max_grad_norm,
                'mixed_precision': config.training.mixed_precision,
                'use_ema': config.ema.use_ema,
                'ema_decay': config.ema.ema_decay if config.ema.use_ema else 0,
            }
            
            # Métricas finais principais
            final_metrics = {
                'final_mae': metrics.get('mae', 0),
                'final_rmse': metrics.get('rmse', 0),
                'final_r2': metrics.get('r2', 0),
            }
            
            self.writer.add_hparams(hparams, final_metrics)
        except Exception as e:
            print(f"[TensorBoard] Warning: Could not log hyperparameters: {e}")
    
    def log_text(self, tag: str, text: str, step: int = None):
        """Log texto."""
        if not self.enabled or self.writer is None:
            return
        step = step if step is not None else self.global_step
        self.writer.add_text(tag, text, step)
    
    def log_convergence_metrics(self, current_loss: float, best_loss: float, patience_counter: int, epoch: int):
        """Log métricas de convergência e early stopping."""
        if not self.enabled:
            return
        
        self.log_scalar('Convergence/Current_Best_Loss', best_loss, epoch)
        self.log_scalar('Convergence/Patience_Counter', patience_counter, epoch)
        
        # Calcular improvement ratio
        if best_loss > 0:
            improvement_ratio = (best_loss - current_loss) / best_loss
            self.log_scalar('Convergence/Improvement_Ratio', improvement_ratio, epoch)
    
    def increment_step(self):
        """Incrementa o step global."""
        self.global_step += 1
    
    def close(self):
        """Fecha o writer do TensorBoard."""
        if self.enabled and self.writer is not None:
            self.writer.close()
            print(f"[TensorBoard] Closed writer for: {self.log_dir}")
 
