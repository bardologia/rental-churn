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
            
        log_level = self.LOG_LEVELS.get(str(level), logging.INFO)
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
        self.logger.info(f">>> {str(title)}")
    
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


class ShapeLogger:
    def __init__(self, model, logger, include_types = (
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
        self.logger = logger
        self.include_types = include_types
        self.records = []
        self.hooks = []
    
    def _hook(self, name):
        def fn(module, inputs, output):
            x = inputs[0]
            in_shape  = tuple(x.shape) if hasattr(x, "shape") else str(type(x))
            if isinstance(output, tuple):
                if hasattr(output[0], "shape"):
                    out_shape = tuple(output[0].shape)
                else:
                    out_shape = f"tuple[{len(output)}]"
            else:
                out_shape = tuple(output.shape) if hasattr(output, "shape") else str(type(output))
            
            self.records.append((name, module.__class__.__name__, in_shape, out_shape))
        
        return fn

    def attach(self):
        self.logger.subsection("Hooks attached to layers for shape logging. \n")
        
        for name, module in self.model.named_modules():
            if name == "":
                continue
           
            if isinstance(module, self.include_types):
                self.hooks.append(module.register_forward_hook(self._hook(name)))
        
        return self

    def detach(self):
        if self.hooks == []:
            return

        self.logger.subsection("Hooks detached from layers. \n")

        for h in self.hooks:
            h.remove()
        
        self.hooks.clear()

    def clear(self):
        self.records.clear()
    
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
        self.logger.subsection(f"Shape log saved to {path} \n")


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
        output_path = Path(path)

        if output_path.suffix == "":
            if isinstance(title, str) and title.lower().endswith(".md"):
                output_path = output_path / title
                title = "Model Summary"
            else:
                output_path = output_path / "model_summary.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        md = self.to_markdown(title=title)
        output_path.write_text(md, encoding="utf-8")


class Tracker:
    def __init__(self, writer):
        self.writer = writer
    
    def log_scalar(self, name: str, value, step: int):
        val = value.item() if hasattr(value, 'item') else value
        self.writer.add_scalar(name, val, step)
    
    def log_dict(self, prefix: str, data_dict: dict, step: int, add_comparison: bool = True):
        comparison_dict = {}
        for key, value in data_dict.items():
            val = value.item() if hasattr(value, 'item') else value
            self.writer.add_scalar(f'{prefix}/{key}', val, step)
            comparison_dict[key] = val
        
        if add_comparison and len(comparison_dict) > 1:
            self.writer.add_scalars(f'{prefix}/comparison', comparison_dict, step)
        
    def log_optimizer(self, optimizer, step: int):
        state_dict = {}
        for i, param_group in enumerate(optimizer.param_groups):
            component_name = param_group.get('name', f'group_{i}')
        
            lr = param_group['lr']
            self.writer.add_scalar(f'optimizer/lr_{component_name}', lr, step)
            state_dict[f'lr_{component_name}'] = lr
            
            for key in ['momentum', 'weight_decay', 'eps']:
                if key in param_group:
                    val = param_group[key]
                    self.writer.add_scalar(f'optimizer/{key}_{component_name}', val, step)
                    state_dict[f'{key}_{component_name}'] = val
        
        self.writer.add_scalars(f'optimizer/comparison', state_dict, step)
    
    def log_gradients(self, model, step: int, max_grad_norm: float = None):
        total_norm       = 0.0
        total_grad_sum   = 0.0
        total_grad_count = 0
        total_zero_grads = 0
        
        layer_norms       = {}
        layer_stats       = {}
        grad_param_ratios = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad       = param.grad.detach()
                param_data = param.detach()
                
                param_norm  = grad.norm(2).item()
                total_norm += param_norm ** 2
            
                grad_flat     = grad.flatten()
                grad_mean     = grad_flat.mean().item()
                grad_std      = grad_std = grad_flat.std(unbiased=False).item() if grad_flat.numel() > 1 else 0.0
                grad_min      = grad_flat.min().item()
                grad_max      = grad_flat.max().item()
                grad_abs_mean = grad_flat.abs().mean().item()
                
                total_grad_sum   += grad_flat.sum().item()
                total_grad_count += grad_flat.numel()
                
                zero_grads        = (grad_flat.abs() < 1e-8).sum().item()
                total_zero_grads += zero_grads
                zero_percent      = 100.0 * zero_grads / grad_flat.numel()
                
                param_norm_val = param_data.norm(2).item()
                if param_norm_val > 1e-8:
                    grad_param_ratio = param_norm / (param_norm_val + 1e-8)
                else:
                    grad_param_ratio = 0.0
                
                layer_norms[name] = param_norm
                layer_stats[name] = {
                    'mean': grad_mean,
                    'std': grad_std,
                    'min': grad_min,
                    'max': grad_max,
                    'abs_mean': grad_abs_mean,
                    'zero_percent': zero_percent
                }
                grad_param_ratios[name] = grad_param_ratio
             
                self.writer.add_scalar(f'gradients/layer_norm/{name}',         param_norm, step)
                self.writer.add_scalar(f'gradients/layer_abs_mean/{name}',     grad_abs_mean, step)
                self.writer.add_scalar(f'gradients/layer_zero_percent/{name}', zero_percent, step)
                self.writer.add_scalar(f'gradients/grad_param_ratio/{name}',   grad_param_ratio, step)
                if grad_flat.numel() > 0:
                    grad_finite = grad_flat[torch.isfinite(grad_flat)]
                    if grad_finite.numel() > 0:
                        self.writer.add_histogram(f'gradients/histogram/{name}', grad_finite, step)
        
        total_norm = total_norm ** 0.5
        
        clip_ratio     = total_norm / max_grad_norm
        is_clipped     = total_norm > max_grad_norm
        effective_norm = min(total_norm, max_grad_norm)
        
        self.writer.add_scalar(f'gradients/clip_ratio',       clip_ratio, step)
        self.writer.add_scalar(f'gradients/is_clipped',       float(is_clipped), step)
        self.writer.add_scalar(f'gradients/effective_norm',   effective_norm, step)
        self.writer.add_scalar(f'gradients/clip_coefficient', min(1.0, max_grad_norm / (total_norm + 1e-6)), step)
        
        avg_grad = total_grad_sum / max(1, total_grad_count)
        zero_grad_percent = 100.0 * total_zero_grads / max(1, total_grad_count)
        
        self.writer.add_scalar(f'gradients/total_norm',        total_norm, step)
        self.writer.add_scalar(f'gradients/avg_gradient',      avg_grad, step)
        self.writer.add_scalar(f'gradients/zero_grad_percent', zero_grad_percent, step)

    def close(self):
        self.writer.close()
