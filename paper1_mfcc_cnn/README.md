# Paper 1: MFCC + CNN + Transformer (RAVDESS)

---

## 🇰🇷 프로젝트 개요 (한글)
이 저장소는 "Speech Emotion Recognition Using Mel-Frequency Cepstral Coefficients & Convolutional Neural Networks" (IDCIoT 2024) 논문을 RAVDESS 데이터셋 기반 3-branch 구조(CNN + CNN + Transformer)로 재구현/검증하는 프로젝트입니다.

### 주요 특징
- **입력**: MFCC 특징 (n_mfcc=40, time frames=282)
- **모델**: 3-branch (CNN 2개 + Transformer 1개) → concat → FC 분류기
- **데이터셋**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

---

## Overview (English)
This module implements a speech emotion recognition (SER) pipeline using:
- **Input**: MFCC features (n_mfcc=40, time frames=282)
- **Model**: 3-branch (2x CNN + 1x Transformer) with feature concatenation and FC classifier
- **Dataset**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

---

## 🇰🇷 데이터 준비 방법
1. **RAVDESS 다운로드**
	 - 공식: [Zenodo RAVDESS](https://zenodo.org/record/1188976)
	 - 모든 오디오 파일을 `../datasets/`에 압축 해제 (폴더 구조 유지)
	 - 예시: `../datasets/Actor_01/03-01-01-01-01-01-01.wav` 등
2. **전처리 실행**
	 - 각 오디오에서 MFCC(40x282) 추출, 학습셋에만 AWGN 노이즈 증강
	 - `.pt` 파일로 저장, manifest CSV 생성
	 - 실행:
		 ```bash
		 python scripts/prepare_data.py
		 ```

## Data Preparation (English)
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

---

## 🇰🇷 모델 구조
- **Branch 1 (CNN1)**: 3단 Conv2d + BatchNorm + ReLU + MaxPool, 512차원 벡터 출력
- **Branch 2 (CNN2)**: CNN1과 동일, 가중치 별도
- **Branch 3 (Transformer)**: MaxPool2d(1x4) → Linear(70→512) → 4-layer TransformerEncoder(4 heads) → Linear(512→1) per token → 40차원 벡터
- **Fusion**: [CNN1, CNN2, Transformer] concat (512+512+40=1064)
- **Classifier**: Linear(1064→8), LogSoftmax

## Model Architecture (English)
- **Branch 1 (CNN1)**: 3-layer Conv2d + BatchNorm + ReLU + MaxPool, output 512-dim vector
- **Branch 2 (CNN2)**: Same as CNN1, separate weights
- **Branch 3 (Transformer)**: MaxPool2d (1x4) → Linear(70→512) → 4-layer TransformerEncoder (4 heads) → Linear(512→1) per token → output 40-dim vector
- **Fusion**: Concatenate [CNN1, CNN2, Transformer] (512+512+40=1064)
- **Classifier**: Linear(1064→8), LogSoftmax

---

## 🇰🇷 학습 방법
- manifest.csv를 기반으로 split/label 정보 사용
- 실험 추적: MLflow, 설정 관리: Hydra
- validation accuracy 기준 early stopping
- 실행:
	```bash
	python scripts/train.py
	```

## Training (English)
- Uses manifest.csv for split/label info
- MLflow for experiment tracking, Hydra for config
- Early stopping on validation accuracy
- Run:
	```bash
	python scripts/train.py
	```

---

## 🇰🇷 평가 방법
- best checkpoint 로드, test셋 추론
- accuracy, UAR, 클래스별 정확도, confusion matrix 저장
- 실행:
	```bash
	python scripts/eval.py
	```

## Evaluation (English)
- Loads best checkpoint, runs inference on test set
- Saves metrics (accuracy, UAR), per-class accuracy, confusion matrix
- Run:
	```bash
	python scripts/eval.py
	```

---

## 🇰🇷 설정 및 재현성
- 모든 파라미터는 `configs/default.yaml`에서 관리
- random seed 고정, outputs/에 모든 결과 저장

## Configuration & Reproducibility (English)
- All parameters (data, augmentation, model, training) are in `configs/default.yaml`
- Set random seed in config for deterministic split
- All outputs (metrics, plots, checkpoints) saved in `outputs/`

---

## References
- [RAVDESS dataset](https://zenodo.org/record/1188976)
- Paper: IDCIoT 2024, pp. 1595–1602
