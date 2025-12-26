"""
model.py

Implements the 3-branch model for SER (MFCC + CNN2 + Transformer) as described in the paper reproduction plan.
- Two parallel CNN branches (CNN1, CNN2)
- One Transformer branch (Option C1: token=40, feature=70)
- Concatenation and final classifier


"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBranch(nn.Module):
    """
    A single CNN branch for MFCC feature extraction.

    Args:
        in_channels (int): Number of input channels (usually 1).
        filters (list[int]): List of output channels for each Conv2d layer.

    Input shape:
        (batch, 1, 40, 282)
    Output shape:
        (batch, 512)
    """
    def __init__(self, in_channels: int, filters: list[int]):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # (40,282) -> (20,141)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)   # (20,141) -> (5,35)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4)   # (5,35) -> (1,8)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 40, 282)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)  # (B, 64, 1, 8) -> (B, 512)
        return x

class TransformerBranch(nn.Module):
    """
    Transformer branch for MFCC feature extraction (Option C1).

    Args:
        in_channels (int): Number of input channels (usually 1).
        d_model (int): Transformer embedding dimension.
        nhead (int): Number of attention heads.
        num_layers (int): Number of Transformer encoder layers.
        dim_feedforward (int): Feedforward network dimension.
        dropout (float): Dropout rate.

    Input shape:
        (batch, 1, 40, 282)
    Output shape:
        (batch, 40)
    """
    def __init__(self, in_channels: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))  # (40,282) -> (40,70)
        self.input_proj = nn.Linear(70, d_model)  # (B,40,70) -> (B,40,512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_head = nn.Linear(d_model, 1)  # (B,40,512) -> (B,40,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 40, 282)
        x = self.pool(x)  # (B,1,40,70)
        x = x.squeeze(1)  # (B,40,70)
        x = self.input_proj(x)  # (B,40,512)
        x = self.transformer(x)  # (B,40,512)
        x = self.proj_head(x).squeeze(-1)  # (B,40)
        return x

class SERModel(nn.Module):
    """
    Full SER model with 2 CNN branches and 1 Transformer branch.

    Args:
        cnn_filters (list[int]): List of output channels for CNN layers.
        d_model (int): Transformer embedding dimension.
        nhead (int): Number of attention heads.
        num_layers (int): Number of Transformer encoder layers.
        dim_feedforward (int): Feedforward network dimension.
        dropout (float): Dropout rate.
        num_classes (int): Number of emotion classes.

    Input shape:
        (batch, 1, 40, 282)
    Output shape:
        (batch, num_classes)
    """
    def __init__(self, cnn_filters: list[int], d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float, num_classes: int):
        super().__init__()
        self.cnn1 = CNNBranch(1, cnn_filters)
        self.cnn2 = CNNBranch(1, cnn_filters)
        self.transformer = TransformerBranch(1, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.classifier = nn.Linear(512+512+40, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 40, 282)
        cnn1_out = self.cnn1(x)  # (B,512)
        cnn2_out = self.cnn2(x)  # (B,512)
        trans_out = self.transformer(x)  # (B,40)
        fused = torch.cat([cnn1_out, cnn2_out, trans_out], dim=-1)  # (B,1064)
        logits = self.classifier(fused)  # (B,num_classes)
        return self.log_softmax(logits)
