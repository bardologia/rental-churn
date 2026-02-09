import logging
import os
import sys
from datetime import datetime
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import torch
import torch
import torch.nn as nn
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
 
