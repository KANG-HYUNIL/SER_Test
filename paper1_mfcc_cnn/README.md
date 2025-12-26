# Paper 1: MFCC + CNN + Transformer (RAVDESS)

Reproduction of "Speech Emotion Recognition Using Mel-Frequency Cepstral Coefficients & Convolutional Neural Networks" (IDCIoT 2024) with a 3-branch architecture (CNN + CNN + Transformer) on the RAVDESS dataset.

## Overview
This module implements a speech emotion recognition (SER) pipeline using:
- **Input**: MFCC features (n_mfcc=40, time frames=282)
- **Model**: 3-branch (2x CNN + 1x Transformer) with feature concatenation and FC classifier
- **Dataset**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

## Data Preparation
1. **Download RAVDESS**
	 - Official: [Zenodo RAVDESS](https://zenodo.org/record/1188976)
	 - Unzip all audio files into `../datasets/` (default path)
	 - Folder structure should be: `../datasets/Actor_01/03-01-01-01-01-01-01.wav`, etc.
2. **Preprocess**
	 - Extracts MFCC (40x282) for each audio, applies AWGN noise augmentation (train set only)
	 - Saves features as `.pt` files and creates a manifest CSV
	 - Run:
		 ```bash
		 python scripts/prepare_data.py
		 ```

## Model Architecture
- **Branch 1 (CNN1)**: 3-layer Conv2d + BatchNorm + ReLU + MaxPool, output 512-dim vector
- **Branch 2 (CNN2)**: Same as CNN1, separate weights
- **Branch 3 (Transformer)**: MaxPool2d (1x4) → Linear(70→512) → 4-layer TransformerEncoder (4 heads) → Linear(512→1) per token → output 40-dim vector
- **Fusion**: Concatenate [CNN1, CNN2, Transformer] (512+512+40=1064)
- **Classifier**: Linear(1064→8), LogSoftmax

## Training
- Uses manifest.csv for split/label info
- MLflow for experiment tracking, Hydra for config
- Early stopping on validation accuracy
- Run:
	```bash
	python scripts/train.py
	```

## Evaluation
- Loads best checkpoint, runs inference on test set
- Saves metrics (accuracy, UAR), per-class accuracy, confusion matrix
- Run:
	```bash
	python scripts/eval.py
	```

## Configuration
- All parameters (data, augmentation, model, training) are in `configs/default.yaml`
- You can change batch size, learning rate, model depth, augmentation SNR, etc.

## Reproducibility
- Set random seed in config for deterministic split
- All outputs (metrics, plots, checkpoints) saved in `outputs/`

## References
- [RAVDESS dataset](https://zenodo.org/record/1188976)
- Paper: IDCIoT 2024, pp. 1595–1602
