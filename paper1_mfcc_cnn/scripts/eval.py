"""
eval.py

Evaluation pipeline for SER (MFCC + CNN + Transformer).
Loads best.ckpt, runs inference on test set, saves metrics and confusion matrix.

"""
import hydra
from omegaconf import DictConfig
import logging
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import pandas as pd
from src.dataset import SERDataset
from src.model import SERModel

logger = logging.getLogger(__name__)

def per_class_accuracy(cm):
    """Compute per-class accuracy from confusion matrix."""
    return cm.diagonal() / cm.sum(axis=1)

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """
    Loads best checkpoint, runs inference on test set, saves metrics and confusion matrix.
    """
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    manifest_path = os.path.join(cfg.data.processed_dir, "manifest.csv")
    test_set = SERDataset(manifest_path, split="test")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    # Model
    model = SERModel(
        cnn_filters=cfg.model.cnn_filters,
        d_model=cfg.model.transformer.d_model,
        nhead=cfg.model.transformer.nhead,
        num_layers=cfg.model.transformer.num_layers,
        dim_feedforward=cfg.model.transformer.dim_feedforward,
        dropout=cfg.model.transformer.dropout,
        num_classes=cfg.model.num_classes
    ).to(device)

    # Load checkpoint
    ckpt_path = os.path.join(cfg.train.log_dir, "best.ckpt")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)


    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    uar = recall_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)
    per_class = per_class_accuracy(cm)


    # Save metrics
    output_dir = cfg.train.log_dir
    os.makedirs(output_dir, exist_ok=True)
    metrics = {"test_acc": float(acc), "test_uar": float(uar)}
    with open(os.path.join(output_dir, "metrics_test.json"), "w") as f:
        json.dump(metrics, f, indent=4)


    # Save per-class metrics
    df = pd.DataFrame({"class": list(range(len(per_class))), "accuracy": per_class})
    df.to_csv(os.path.join(output_dir, "per_class_metrics.csv"), index=False)


    # Save confusion matrix
    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.xticks(np.arange(cfg.model.num_classes))
    plt.yticks(np.arange(cfg.model.num_classes))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="red")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
