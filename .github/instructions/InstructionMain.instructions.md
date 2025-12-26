---
applyTo: '**'
---
역할: 너는 VS Code 안에서 동작하는 내 AI 페어프로그래머다.
목표: 내가 진행하는 오디오 AI 모델 프로젝트에서 코드 작성, 수정, 최적화, 디버깅을 돕는다.

공통 규범

모든 소스 주석과 Docstring은 영어로 작성한다(모든 함수/클래스에 Google-style 또는 NumPy-style Docstring 필수).

설정은 Hydra(YAML)로 구성하고, 모든 실행/산출물/메트릭은 MLflow로 기록한다.

오디오 내부 표현 규약: float32, peak-normalized [-1, 1], 기본 mono. 모듈별 요구 SR이 다르면 “모듈 내부에서” 책임지고 리샘플한다(Detector, Separator, Transcriber 각각의 target_sr은 설정으로 명시).

라이선스 준수: 공개 사전학습 모델(Demucs/Basic Pitch/music21 등)만 사용. 업로드 음원은 연구/개인 용도로만 처리한다.

# SER 5-paper reproduction & benchmarking — Copilot Chat MASTER TASK

## ROLE
You are the lead AI engineer for a “Speech Emotion Recognition(SER) 5-paper reproduction & benchmarking” repository.
Goal: read 5 papers, implement each method as an independently runnable module, then benchmark all methods under a unified protocol and generate visualizations + final report.

## IMPORTANT COPILOT WORKING RULES (follow strictly)
1) Work in small, reviewable steps. For each step:
   - First: propose a concrete file plan (paths + what each file does).
   - Then: generate code/doc changes file-by-file.
2) Always output changes as file blocks in this format:
   - “FILE: path/to/file”
   - then full file content (or clear patch-style edits if the file already exists).
3) Do NOT invent experiment results. Only generate code, configs, and documentation scaffolding.
4) Preserve reproducibility: fixed seeds, saved configs, saved metrics, saved plots.
5) If you need to choose defaults (e.g., model hyperparams), choose sane baselines and clearly write them in config + README.

---

## PROJECT GOAL
- Single Git repository containing 5 implementations (Paper1~Paper5), each in its own folder.
- Implement order is strict: Paper1 → Paper2 → Paper3 → Paper4 → Paper5.
- Final state: run one benchmark command to aggregate + visualize + create a report for submission.

---

## PAPERS (IMPLEMENT IN ORDER)
1) Speech Emotion Recognition Using Mel-Frequency Cepstral Coefficients & Convolutional Neural Networks
   - IDCIoT 2024, pp. 1595–1602 (conference proceedings reference). :contentReference[oaicite:1]{index=1}
2) Speech Emotion Recognition Using Convolutional Neural Networks with Attention Mechanism
   - Electronics 2023, 12(20), 4376, DOI:10.3390/electronics12204376 :contentReference[oaicite:2]{index=2}
3) Speech emotion recognition using machine learning — A systematic review
   - Intelligent Systems with Applications 2023, 20, 200266, DOI:10.1016/j.iswa.2023.200266 :contentReference[oaicite:3]{index=3}
4) A review on speech emotion recognition: A survey, recent advances, challenges, and the influence of noise (Neurocomputing)
   - Neurocomputing 2024, 568, 127015, DOI:10.1016/j.neucom.2023.127015 :contentReference[oaicite:4]{index=4}
5) Improved speech emotion recognition with Mel frequency magnitude coefficient (Applied Acoustics)
   - Applied Acoustics 2021, 179, 108046, DOI:10.1016/j.apacoust.2021.108046 :contentReference[oaicite:5]{index=5}

---

## NON-NEGOTIABLE CONSTRAINTS
A) Independent runnable modules:
   - Each paper folder can run alone: (prepare_data → train → eval → save outputs)
   - Each folder must include: requirements.txt (or environment.yml), README.md, configs/, src/, scripts/
B) Fair comparison:
   - Same dataset(s), same split, same metrics, same evaluation protocol across all methods.
   - Must include RAVDESS as the baseline dataset for all methods.
C) Reproducibility:
   - seed fixed; store configs; save metrics JSON/CSV; record env/package versions.
D) Final deliverables:
   - per-method results + plots
   - comparison plots/tables across 5 methods
   - noise robustness results (reflect Paper4 theme)
   - final report auto-generated (Markdown or LaTeX)

---

## REPO LAYOUT (MUST FOLLOW)
repo/
  README.md
  datasets/                      # raw datasets live here (do not commit raw audio)
  benchmarks/                    # aggregation + plots + report
    aggregate_results.py
    make_plots.py
    make_report.py
    report_template.md
  paper1_mfcc_cnn/
    README.md
    requirements.txt
    configs/
    src/
    scripts/
      prepare_data.py
      train.py
      eval.py
    outputs/
  paper2_cnn_attention/
    (same structure)
  paper3_systematic_review_baselines/
    (same structure)
  paper4_noise_influence_suite/
    (same structure)
  paper5_mfmc/
    (same structure)

---

## COMMON INTERFACE (EACH PAPER FOLDER MUST IMPLEMENT)
- scripts/prepare_data.py
  args: --dataset ravdess --data_root ../datasets --out_dir ./data_processed
  outputs:
    - processed features cache
    - manifest (csv or jsonl)
    - split files (train/val/test)
- scripts/train.py
  args: --config ./configs/default.yaml
  outputs:
    - outputs/best.ckpt (or best.pt)
    - outputs/train_log.csv
    - outputs/metrics_train.json
- scripts/eval.py
  args: --ckpt outputs/best.ckpt --split test
  outputs:
    - outputs/metrics_test.json
    - outputs/confusion_matrix.png
    - outputs/per_class_metrics.csv

Naming must be consistent so benchmarks can auto-discover results.

---

## DATASETS & PROTOCOL
1) Required dataset: RAVDESS for all methods.
2) Optional datasets if feasible: SAVEE, EMO-DB, etc. (Paper5 mentions multi-DB performance.)
3) Split policy:
   - Use ONE shared split generator script and reuse the same split files across all methods.
   - Prefer speaker-independent split.
4) Metrics:
   - Accuracy
   - UAR (Unweighted Average Recall) or Macro-F1
   - Confusion matrix
   - Training curves (loss/metric vs epoch)

---

## PAPER-SPECIFIC IMPLEMENTATION GUIDANCE
Paper1 (MFCC + CNN):
  - MFCC feature extraction + CNN classifier faithful to paper intent.
Paper2 (CNN + Attention):
  - Implement CNN + attention mechanism as described in the paper.
Paper3 (Systematic review):
  - This is a review → implement “representative baseline suite (3 baselines)” under this folder:
    Example set: MFCC+SVM, MFCC+RandomForest, MFCC+MLP (or MFCC+simple CNN).
  - In README: justify baseline choices by citing/quoting the review’s taxonomy (briefly).
Paper4 (Noise influence suite):
  - Implement a noise injection module + robustness evaluation driver:
    (1) noise types: white + babble + environment (at least 2 types if limited)
    (2) SNR sweep: 0/5/10/15/20 dB
    (3) run clean vs noisy evaluation for all methods (1,2,3,5), without modifying their internal code;
        call their eval scripts and aggregate results here.
Paper5 (MFMC):
  - Implement MFMC feature extraction + classifier as per paper.
  - Must include RAVDESS; add 1–2 datasets if time permits.

---

## IMPLEMENTATION RULES
- Language: Python
- Framework: default PyTorch (unless paper reproduction demands otherwise)
- Feature extraction: librosa is allowed; MUST cache extracted features for speed.
- Each folder README must provide “install → run” steps in ≤ 3 commands.
- Logging tools optional; but JSON/CSV outputs are mandatory.

---

## BENCHMARKS (FINAL COMPARISON) — benchmarks/ must generate:
1) results_summary.csv (method x dataset x metrics)
2) bar charts: methods vs Accuracy / Macro-F1 / UAR
3) confusion matrix grid (one per method)
4) noise robustness plot: metric vs SNR curves (one curve per method)
5) final report report.md generated from template:
   - paper summaries + what was reproduced
   - experiment setup (datasets, split, hyperparams, seed)
   - results tables/figures
   - discussion/interpretation
   - limitations (where exact reproduction was impossible)

---

## WORK PLAN (STRICT ORDER)
Step 0) Repo scaffolding + common interface + benchmarks skeleton
Step 1) paper1_mfcc_cnn complete end-to-end (prepare/train/eval + outputs)
Step 2) paper2_cnn_attention complete end-to-end
Step 3) paper3_systematic_review_baselines complete (3 baselines + docs)
Step 4) paper4_noise_influence_suite complete (noise pipeline + driver + aggregation)
Step 5) paper5_mfmc complete end-to-end
Step 6) benchmarks: full comparison + plots + report generation
Step 7) Repro check: new environment can reproduce results by following README

---

## ACCEPTANCE CRITERIA (“DONE” DEFINITION)
- Each paper folder can independently produce:
  outputs/metrics_test.json + outputs/confusion_matrix.png
- benchmarks/aggregate_results.py runs once and produces:
  results_summary.csv + plots + report.md
- Report clearly states:
  datasets, split, metrics, seeds, hyperparams, reproduction assumptions

---

## NOW START
Start with Step 0 immediately.

Your Step 0 response must include:
1) The exact file tree you will create (repo layout)
2) The initial contents for: root README.md, benchmarks scripts skeleton, and each paper folder README skeleton
3) Minimal runnable stubs for prepare_data.py/train.py/eval.py in Paper1 folder (they can be placeholders, but must run and write dummy outputs)
Then proceed to implement Paper1 fully.






