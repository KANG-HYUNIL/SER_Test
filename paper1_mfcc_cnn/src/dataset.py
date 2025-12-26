"""
dataset.py

PyTorch Dataset for SER using manifest.csv (Paper1: MFCC + CNN + Transformer).
Loads .pt feature files and labels from manifest.

"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SERDataset(Dataset):
    """
    SERDataset loads MFCC features and labels from manifest.csv for SER experiments.

    Args:
        manifest_path (str): Path to manifest.csv file.
        split (str): Which split to load ('train', 'val', 'test').

    Returns:
        Tuple (feature_tensor, label)
    """
    def __init__(self, manifest_path: str, split: str):
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.load(row['file'])  # shape: (1, 40, 282)
        y = int(row['label'])
        return x, y
