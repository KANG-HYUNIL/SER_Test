"""
prepare_data.py

Preprocess RAVDESS dataset for SER experiment (MFCC + CNN Transformer).
- Loads audio files, splits into train/val/test, applies AWGN augmentation,
  extracts MFCC (40x282), saves as .pt files and manifest CSV.

Usage:
    python scripts/prepare_data.py --config ../configs/default.yaml

"""
import hydra
from omegaconf import DictConfig
import logging
import os
import glob
import librosa
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Additive White Gaussian Noise (AWGN) augmentation.

    Args:
        signal (np.ndarray): 1D audio waveform (normalized to [-1, 1]).
        snr_db (float): Desired Signal-to-Noise Ratio in decibels (dB).

    Returns:
        np.ndarray: Noisy audio waveform (same shape as input).

    Purpose:
        Simulates real-world noise by adding Gaussian noise to the input signal,
        controlled by the SNR value. Used for data augmentation to improve model robustness.
    """
    rms = np.sqrt(np.mean(signal ** 2))
    snr = 10 ** (snr_db / 10)
    noise_std = np.sqrt(rms ** 2 / snr)
    noise = np.random.normal(0, noise_std, signal.shape)
    return signal + noise

def fix_length_mfcc(mfcc: np.ndarray, target_len: int, pad_mode: str = "zero", crop_mode: str = "left") -> np.ndarray:
    """
    Pad or truncate MFCC along the time axis to a fixed length.

    Args:
        mfcc (np.ndarray): MFCC feature array of shape (n_mfcc, time_frames).
        target_len (int): Desired number of time frames after padding/truncation.
        pad_mode (str): Padding mode ('zero' for zero-padding).
        crop_mode (str): Truncation mode ('left' for start, 'center' for middle).

    Returns:
        np.ndarray: MFCC array of shape (n_mfcc, target_len).

    Purpose:
        Ensures all MFCC features have the same time dimension for batch processing.
    """
    # mfcc shape 획득
    n_mfcc, t = mfcc.shape

    #if shorter than target length
    if t < target_len:
        # Pad with zeros at the end
        pad_width = target_len - t
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode="constant")
    
    # if longer than target length
    elif t > target_len:
        # Truncate
        if crop_mode == "center":
            start = (t - target_len) // 2
            mfcc = mfcc[:, start:start+target_len]
        else:
            mfcc = mfcc[:, :target_len]
    return mfcc

def extract_mfcc(wav_path: str, sr: int, n_mfcc: int, target_len: int, pad_mode: str, crop_mode: str) -> np.ndarray:
    """
    Load an audio file, extract MFCC features, and fix their shape.

    Args:
        wav_path (str): Path to the .wav audio file.
        sr (int): Target sampling rate for audio loading.
        n_mfcc (int): Number of MFCC coefficients to extract.
        target_len (int): Desired time frames for MFCC.
        pad_mode (str): Padding mode for MFCC.
        crop_mode (str): Truncation mode for MFCC.

    Returns:
        np.ndarray: MFCC feature array of shape (n_mfcc, target_len).

    Purpose:
        Standardizes audio input into fixed-size MFCC features for model input.
    """

    # librosa load 로 raw 획득
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    # Normalize amplitude to [-1, 1] 
    y = y / (np.max(np.abs(y)) + 1e-8)

    #librosa 메서드 통해 MFCC 추출
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = fix_length_mfcc(mfcc, target_len, pad_mode, crop_mode)
    return mfcc.astype(np.float32)

def save_tensor_and_meta(tensor: np.ndarray, out_dir: str, base_name: str, label: int, split: str, meta_list: list):
    """
    Save a tensor as a .pt file and record its metadata.

    Args:
        tensor (np.ndarray): Feature tensor to save (e.g., MFCC).
        out_dir (str): Directory to save the .pt file.
        base_name (str): Base filename (without extension).
        label (int): Emotion class label (0-based).
        split (str): Data split ('train', 'val', or 'test').
        meta_list (list): List to append metadata dict to.

    Returns:
        None

    Purpose:
        Persists processed features to disk and tracks their metadata for manifest creation.
    """
    tensor_path = os.path.join(out_dir, f"{base_name}.pt")

    #torch.save 로 저장
    torch.save(torch.from_numpy(tensor), tensor_path)
    meta_list.append({"file": tensor_path, "label": label, "split": split})

def get_label_from_path(path: str) -> int:
    """
    Extract the emotion label index from a RAVDESS filename.

    Args:
        path (str): Path to the RAVDESS .wav file.

    Returns:
        int: 0-based emotion class index (0=neutral, 7=disgust).

    Purpose:
        Decodes the emotion class from the standardized RAVDESS filename format.
    """
    # RAVDESS: .../03-01-05-01-01-01-01.wav (3rd field: emotion)
    fname = os.path.basename(path)
    parts = fname.split("-")
    emotion_id = int(parts[2])
    # Map RAVDESS emotion_id (1~8) to 0-based index
    return emotion_id - 1

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """
    Main entry point for data preprocessing.

    Steps:
        1. Scan all RAVDESS .wav files in the dataset directory.
        2. Extract emotion labels from filenames.
        3. Stratified split into train/val/test sets.
        4. For train set, apply AWGN augmentation (2x per sample).
        5. Extract MFCC features, fix to (40,282), and save as .pt files.
        6. Save a manifest CSV with file paths, labels, and split info.

    Args:
        cfg (DictConfig): Hydra configuration object loaded from YAML.

    Returns:
        None

    Purpose:
        Automates the full preprocessing pipeline for SER experiments, ensuring
        reproducibility and correct data splits/augmentation for fair benchmarking.
    """
    os.makedirs(cfg.data.processed_dir, exist_ok=True)
    # 1. Scan all wav files/파일 가져오기
    wav_files = glob.glob(os.path.join(cfg.data.data_root, "**", "*.wav"), recursive=True)
    logger.info(f"Found {len(wav_files)} wav files.")

    # 2. Extract labels/라벨 획득
    labels = [get_label_from_path(p) for p in wav_files]

    # 3. Split/train/val/test 분할
    train_files, testval_files, train_labels, testval_labels = train_test_split(
        wav_files, labels, test_size=cfg.data.split_ratio[1]+cfg.data.split_ratio[2],
        random_state=cfg.data.seed, stratify=labels)
    
    # validation 비율 획득
    val_size = cfg.data.split_ratio[1] / (cfg.data.split_ratio[1]+cfg.data.split_ratio[2])
    val_files, test_files, val_labels, test_labels = train_test_split(
        testval_files, testval_labels, test_size=1-val_size,
        random_state=cfg.data.seed, stratify=testval_labels)
    splits = [(train_files, train_labels, "train"), (val_files, val_labels, "val"), (test_files, test_labels, "test")]
    meta = []


    # 4. Process each split
    for files, labels, split in splits:
        logger.info(f"Processing {split} set: {len(files)} samples.")
        for i, (wav, label) in enumerate(zip(files, labels)):
            base = f"{split}_{i}"
            mfcc = extract_mfcc(wav, cfg.data.sr, cfg.data.n_mfcc, cfg.data.target_len, cfg.data.pad_mode, cfg.data.crop_mode)
            
            # Save clean sample
            save_tensor_and_meta(mfcc[None, ...], cfg.data.processed_dir, base, label, split, meta)
            
            # AWGN augmentation (train only)
            if split == "train" and cfg.augment.use_awgn:
                for j in range(cfg.augment.n_aug):
                    snr = np.random.uniform(cfg.augment.snr_min, cfg.augment.snr_max)
                    noisy = add_awgn(mfcc, snr)
                    save_tensor_and_meta(noisy[None, ...], cfg.data.processed_dir, f"{base}_awgn{j}", label, split, meta)

    # 5. Save manifest
    df = pd.DataFrame(meta)
    df.to_csv(os.path.join(cfg.data.processed_dir, "manifest.csv"), index=False)
    logger.info(f"Saved manifest with {len(df)} entries to {cfg.data.processed_dir}/manifest.csv")

if __name__ == "__main__":
    main()
