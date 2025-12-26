import hydra
from omegaconf import DictConfig
import logging
import mlflow
import os
import torch

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    # Setup MLflow
    mlflow.set_tracking_uri("file://" + os.path.abspath(cfg.train.log_dir + "/mlruns"))
    mlflow.set_experiment(cfg.train.exp_name)
    
    with mlflow.start_run():
        logger.info("Starting training...")
        mlflow.log_params(cfg.train)
        mlflow.log_params(cfg.model)
        
        # TODO: Load data, Initialize model, Run training loop
        
        # Dummy metrics for stub verification
        mlflow.log_metric("train_loss", 0.5)
        mlflow.log_metric("train_acc", 0.8)
        
        # Dummy model save
        os.makedirs(cfg.train.log_dir, exist_ok=True)
        dummy_ckpt_path = os.path.join(cfg.train.log_dir, "best.ckpt")
        torch.save({"model_state": "dummy"}, dummy_ckpt_path)
        logger.info(f"Saved dummy checkpoint to {dummy_ckpt_path}")

if __name__ == "__main__":
    main()
