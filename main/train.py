import sys
import os
import pandas as pd
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from core.data    import DataPipeline
from core.dataset import DatasetLoader
from core.model   import Model
from core.trainer import Trainer
from core.logger  import Logger
from core.results import Results
from core.logger  import TensorLogger, ModelSummary, TensorBoardMonitor


def train(config, use_cache, save_cache, save_results=False):
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, config.paths.runs_dir, run_id)

    raw_path   = os.path.join(project_root, config.paths.raw_data)
    train_path = os.path.join(project_root, config.paths.train_data)
    
    state_dict_path = os.path.join(run_dir, "model_state_dict.pt")
    metadata_path   = os.path.join(run_dir,"model_metadata.pt")
    plots_dir = os.path.join(run_dir, "plots")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(state_dict_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path),   exist_ok=True)

    logger = Logger(name="train", level="INFO", log_dir=run_dir)

    try:
        logger.info(f"[GPU] Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"[GPU] Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logger.info(f"[GPU] CUDA Version: {torch.version.cuda} \n")
    except Exception:
        logger.info("[GPU] CUDA not available or query failed.\n")

    if config.overfit.overfit_single_batch:
        config.training.dropout  = config.overfit.overfit_dropout
        config.architecture.embedding_dropout = config.overfit.overfit_dropout
        config.architecture.use_augmentation = config.overfit.overfit_use_augmentation
        config.training.weight_decay = config.overfit.overfit_weight_decay
        config.training.patience = config.overfit.overfit_patience
        config.training.epochs = config.overfit.overfit_epochs
        config.scheduler.scheduler_patience = config.overfit.overfit_patience
        config.training.mixed_precision = config.overfit.overfit_mixed_precision
        config.ema.use_ema = config.overfit.overfit_use_ema
        config.split.val_size = config.overfit.overfit_val_size
        config.split.test_size = config.overfit.overfit_test_size
        config.load.user_sample_count = config.overfit.overfit_number_of_users
        config.architecture.min_seq_len = config.overfit.overfit_min_length
        logger.warning("Overfit single batch mode: Disabled dropout, augmentation, weight decay, mixed precision, and increased epochs and patience.")

    if use_cache:
        dataframe = pd.read_parquet(train_path)
    else:
        pipeline  = DataPipeline(config, logger)
        dataframe = pipeline.run(raw_path, train_path)
        if save_cache:
            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            dataframe.to_parquet(train_path, index=False)
            logger.info(f"[Cache] Training data cached at: {train_path}")
            
    dataset_loader = DatasetLoader(dataframe=dataframe, cfg=config, logger=logger)
    
    train_loader, validation_loader, test_loader = dataset_loader.run()
    
    target_scaler = dataset_loader.target_scalers[config.columns.target_col_name]
    continuous_scalers  = dataset_loader.continuous_scalers 

    model = Model(
        embedding_dimensions = dataset_loader.embedding_dimensions,             
        num_continuous = len(dataset_loader.continuous_columns), 
        target_scaler = target_scaler,
        feature_scaler = continuous_scalers,
        config = config
    )    

    summary = ModelSummary(model)
    summary.run()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_batch = next(iter(train_loader))
    tensor_logger = TensorLogger(model=model).attach().log_from_batch(sample_batch, device)

    tb_log_dir = os.path.join(run_dir, "tensorboard")
    tb_monitor = TensorBoardMonitor(log_dir=tb_log_dir, enabled=True)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        target_scaler=target_scaler,
        logger = logger,
        config=config,
        tb_monitor=tb_monitor
    )
    
    trained_model = trainer.fit()
  
    evaluator = Results(
        model=trained_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    train_results = evaluator.run(train_loader)
    val_results   = evaluator.run(validation_loader)
    
    test_results = evaluator.run(test_loader, make_plots=True)
    test_plots   = test_results.get('plots', {})
       
    train_metrics = train_results['metrics']
    val_metrics   = val_results['metrics']
    test_metrics  = test_results['metrics']

    metric_keys = sorted(set(train_metrics) | set(val_metrics) | set(test_metrics))

    tensor_logger.save_markdown("C:\\Users\\victo\\Desktop\\rental-churn - 2\\rental-churn\\tensor_logger.md")
    summary.save_markdown("C:\\Users\\victo\\Desktop\\rental-churn - 2\\rental-churn\\model_summary.md")

    def fmt(v):
        return "-" if v is None else f"{v:.4f}"

    logger.info("[Metrics] Train | Validation | Test (days)")
    for k in metric_keys:
        train_s = fmt(train_metrics.get(k))
        val_s   = fmt(val_metrics.get(k))
        test_s  = fmt(test_metrics.get(k))
        logger.info(f"{k:<24} = {train_s:>10} | {val_s:>10} | {test_s:>10}")
  
    metadata = {
        "embedding_dimensions" : trained_model.embedding_dimensions,
        "num_continuous"       : trained_model.num_continuous,
        "target_scaler"        : trained_model.target_scaler,
        "feature_scaler"       : trained_model.feature_scaler,
        "config"               : trained_model.config,
        "categorical_maps"     : dataset_loader.categorical_maps,
    }

    torch.save(metadata, metadata_path)
    logger.info(f"[Checkpoint] Saved metadata to: {metadata_path}")

    if save_results:
        torch.save(trained_model.state_dict(), state_dict_path)
        logger.info(f"[Checkpoint] Saved state_dict to: {state_dict_path}")
        torch.save(metadata, metadata_path)
        logger.info(f"[Checkpoint] Saved metadata to: {metadata_path}")
        
        for name, fig in test_plots.items():
            png_path = os.path.join(plots_dir, f"test_{name}.png")
            svg_path = os.path.join(plots_dir, f"test_{name}.svg")
            fig.savefig(png_path, bbox_inches='tight')
            fig.savefig(svg_path, bbox_inches='tight')
            logger.info(f"[Plots] Saved plot: {png_path}")
  
    logger.close()
    return trained_model, metadata, train_metrics, val_metrics, test_metrics


if __name__ == "__main__":
    import argparse
    from main.config import config

    parser = argparse.ArgumentParser(description="Train model for rental churn")
    parser.add_argument("--use-cache", dest="use_cache", action="store_true", help="Load preprocessed training data from cache")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Do not use cached training data")
    parser.set_defaults(use_cache=True)

    parser.add_argument("--save-cache", dest="save_cache", action="store_true", help="Save processed training data to cache")
    parser.set_defaults(save_cache=False)

    parser.add_argument("--save-results", dest="save_results", action="store_true", help="Save model state and metadata")
    parser.set_defaults(save_results=False)

    parser.add_argument("--overfit", dest="overfit", action="store_true", help="Enable overfit-single-batch mode using config.overfit settings")
    parser.set_defaults(overfit=False)

    args = parser.parse_args()

    if args.overfit:
        config.overfit.overfit_single_batch = True

    trained_model, metadata, train_metrics, val_metrics, test_metrics = train(
        config,
        use_cache=args.use_cache,
        save_cache=args.save_cache,
        save_results=args.save_results,
    )
