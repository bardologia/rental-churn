import os
import sys
import argparse
from pathlib import Path

import pyarrow.parquet as pq
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import torch

torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from main.config import config
from core.model import Model
from core.inference import InferenceEngine
from core.logger import Logger


def infer(model_path: str = None, input_path: str = None, batch_size: int = 256):
    model_dir       = Path(model_path)
    metadata_path   = model_dir / "model" / "model_metadata.pt"
    state_dict_path = model_dir / "model" / "model_state_dict.pt"
    output_file     = model_dir / "inference" / "inference_results.csv"

    logger = Logger(name="inference", level="INFO", log_dir=model_dir / "inference")
    
    table    = pq.read_table(input_path)
    df       = table.to_pandas(ignore_metadata=True, strings_to_categorical=False)
    
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    logger.section("[Metadata loaded]")
    
    model = Model(
        embedding_dimensions = metadata["embedding_dimensions"],
        num_continuous       = metadata["num_continuous"],
        target_scaler        = metadata.get("target_scaler"),
        feature_scaler       = metadata.get("feature_scaler"),
        config               = metadata["config"],
    )
    
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    logger.section("[State dict loaded] : " + str(state_dict_path))
   
    engine = InferenceEngine(
        model            = model,
        target_scaler    = metadata['target_scaler'],
        feature_scalers  = metadata['feature_scaler'],
        config           = metadata["config"],
        categorical_maps = metadata['categorical_maps'],
        logger           = logger,
        device           = "cuda"
    )
    
    predictions = engine.predict(df, batch_size=batch_size)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_file, index=False)
    logger.info(f"[Inference] Results saved to: {output_file}")
    
    return predictions, str(output_file)


def main():
    model_path = r"C:\Users\victo\Desktop\rental-churn - 2\churn-model\small_model"
    input_path = r"C:\Users\victo\Desktop\rental-churn - 2\data\inference.parquet"

    predictions, output_path = infer(
        model_path=model_path,
        input_path=input_path,
        batch_size=256,
    )


if __name__ == "__main__":
    main()
