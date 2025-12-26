# Paper 1: MFCC + CNN

Implementation of "Speech Emotion Recognition Using Mel-Frequency Cepstral Coefficients & Convolutional Neural Networks" (IDCIoT 2024).

## Architecture
- **Input**: MFCC features (n_mfcc=40)
- **Model**: 1D CNN with Batch Normalization and ReLU
- **Dataset**: RAVDESS

## Usage

### 1. Prepare Data
```bash
python scripts/prepare_data.py --dataset ravdess --data_root ../datasets --out_dir ./data_processed
```

### 2. Train
```bash
python scripts/train.py
```

### 3. Evaluate
```bash
python scripts/eval.py --ckpt outputs/best.ckpt --split test
```
