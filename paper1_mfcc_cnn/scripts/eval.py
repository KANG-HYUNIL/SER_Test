import hydra
from omegaconf import DictConfig
import logging
import json
import os
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    logger.info("Starting evaluation...")
    
    # TODO: Load model from checkpoint, Run inference on test set
    
    # Dummy outputs
    metrics = {"test_acc": 0.75, "test_uar": 0.70}
    output_dir = cfg.train.log_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics_test.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save dummy confusion matrix
    plt.figure()
    plt.title("Dummy Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
