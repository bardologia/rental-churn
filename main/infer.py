"""Inference module for rental churn prediction model."""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import pyarrow.parquet as pq
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from main.config import config
from core.model import Model
from core.inference import InferenceEngine
from core.logger import Logger


def infer(model_path: str = None, save_predictions: bool = True, batch_size: int = 256):
    project_root = Path(os.path.abspath(os.path.join(current_dir, "..")))
    workspace_root = project_root.parent

    input_path    = project_root / "data" / "inference_data.parquet"
    model_dir     = Path(model_path) if model_path else workspace_root / "churn-model" / "small_model"
    metadata_path = model_dir / "model_metadata.pt"
    
    logger = Logger(name="infer", level="INFO", log_dir=None)
    logger.info(f"Loading data from {input_path}")
    
    table = pq.read_table(input_path)
    df = table.to_pandas(ignore_metadata=True, strings_to_categorical=False)
    
    logger.info(f"[Data loaded] : {df.shape}")
    logger.info(f"[Using model] : {model_dir}")
    state_dict_path = model_dir / "model_state_dict.pt"
    
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    logger.info("[Metadata loaded]")
    
    model = Model(
        embedding_dimensions=metadata["embedding_dimensions"],
        num_continuous=metadata["num_continuous"],
        target_scaler=metadata.get("target_scaler"),
        feature_scaler=metadata.get("feature_scaler"),
        config=metadata["config"],
    )
    
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    logger.info("[State dict loaded] : " + str(state_dict_path))
   
    engine = InferenceEngine(
        model            = model,
        target_scaler    = metadata['target_scaler'],
        feature_scalers  = metadata['feature_scaler'],
        config           = metadata["config"],
        categorical_maps = metadata['categorical_maps'],
        logger=logger,
        device= "cuda"
    )
    
    num_users = df[config.columns.user_id_col].nunique()
    logger.info(f"[Inference] Processing {len(df)} rows from {num_users} users with batch_size={batch_size}")
    predictions = engine.predict(df, batch_size=batch_size)
    logger.info(f"[Inference] Inference complete: {len(predictions)} predictions")
    
    if save_predictions:
        output_file = model_dir / "inference_results.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_file, index=False)
        logger.info(f"[Inference] Results saved to: {output_file}")
    
    return predictions, str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained rental churn prediction model")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the model directory (default: churn-model/small_model)",)
    parser.add_argument("--save", action="store_true", help="Whether to save predictions to a CSV file",)
    parser.add_argument("--batch-size", type=int, default=256,   help="Batch size for processing (default: 256)",)
    args = parser.parse_args()

    predictions, output_path = infer(
        model_path=args.model_path,
        save_predictions=args.save,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
