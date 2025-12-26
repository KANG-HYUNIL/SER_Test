# SER 5-Paper Reproduction & Benchmarking

This repository contains independent implementations of 5 Speech Emotion Recognition (SER) methods, followed by a unified benchmarking suite.

## Project Structure

- `datasets/`: Raw datasets (RAVDESS, etc.)
- `benchmarks/`: Scripts to aggregate results and generate the final report.
- `paper1_mfcc_cnn/`: Implementation of "Speech Emotion Recognition Using Mel-Frequency Cepstral Coefficients & Convolutional Neural Networks" (IDCIoT 2024).
- `paper2_cnn_attention/`: Implementation of CNN + Attention mechanism.
- `paper3_systematic_review_baselines/`: Baseline suite (SVM, MLP, etc.).
- `paper4_noise_influence_suite/`: Noise robustness evaluation.
- `paper5_mfmc/`: MFMC feature-based method.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare datasets in `datasets/` folder.
3. Follow instructions in each paper's README.
