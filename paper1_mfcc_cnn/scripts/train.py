"""
train.py

Full training pipeline for SER (MFCC + CNN + Transformer).
Loads manifest, builds DataLoader, trains model, logs with MLflow, saves best checkpoint.

"""
import hydra
from omegaconf import DictConfig
import logging
import mlflow
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import SERDataset
from src.model import SERModel
import numpy as np

logger = logging.getLogger(__name__)

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy.

    Args:
        preds (torch.Tensor): Predicted class indices (shape: [N]).
        labels (torch.Tensor): True class indices (shape: [N]).
    Returns:
        float: Accuracy (0~1).
    """
    return (preds == labels).sum().item() / len(labels)

def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model (torch.nn.Module): Model to train.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to use.
    Returns:
        tuple[float, float]: (mean loss, mean accuracy)
    """
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total

def eval_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """
    Evaluate on val/test set.

    Args:
        model (torch.nn.Module): Model to evaluate.
        loader (DataLoader): Validation/test data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use.
    Returns:
        tuple[float, float]: (mean loss, mean accuracy)
    """
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, total_correct / total

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """
    Main training entry.
    Loads manifest, builds DataLoader, trains model, logs with MLflow, saves best checkpoint.
    """
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.train.log_dir, exist_ok=True)
    manifest_path = os.path.join(cfg.data.processed_dir, "manifest.csv")

    # DataLoader
    train_set = SERDataset(manifest_path, split="train")
    val_set = SERDataset(manifest_path, split="val")
    train_loader = DataLoader(train_set, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    
    
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
    
    # Optimizer, Loss
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    criterion = nn.NLLLoss()

    # MLflow
    mlflow.set_tracking_uri("file://" + os.path.abspath(cfg.train.log_dir + "/mlruns"))
    mlflow.set_experiment(cfg.train.exp_name)
    best_val_acc = 0
    patience = 0
    
    with mlflow.start_run():
        mlflow.log_params(dict(cfg.train))
        mlflow.log_params(dict(cfg.model))

        for epoch in range(cfg.train.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
            mlflow.log_metrics({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            logger.info(f"Epoch {epoch+1}/{cfg.train.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
           
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                best_path = os.path.join(cfg.train.log_dir, "best.ckpt")
                torch.save({"model_state_dict": model.state_dict()}, best_path)
                logger.info(f"Saved best model to {best_path}")
            else:
                patience += 1
                if cfg.train.early_stopping and patience >= cfg.train.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

if __name__ == "__main__":
    main()
