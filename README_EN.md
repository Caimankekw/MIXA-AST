# MixA-AST: Speech Quality Regression with Mixed Attention

[简体中文](README.md) | **English**

A no-reference speech quality assessment project built on the Audio Spectrogram Transformer (AST). The model takes log-mel / Kaldi filterbank features as input and predicts a scalar Mean Opinion Score (MOS). This repository implements an AST baseline, MixA mixed attention, and a training and evaluation pipeline for NISQA-CORPUS.

## Implementation

- **AST regression**: adapts the DeiT/AST classification architecture to scalar regression by averaging its two special tokens and applying LayerNorm followed by a linear head.
- **MixA mixed attention**: uses ReLU quadratic attention in selected layers and SteLLA linear attention in the remaining layers. The historical k=2 configuration selects layers **1 and 11** (zero-based indices `0,10`).
- **Layer importance scoring**: ranks layers using attention gradients and the difference between original and replacement attention on calibration samples from the training split, then selects the top-k layers.
- **Training stability**: keeps MixA scaling factors positive through `softplus(raw) + 1e-4`. Supports SmoothL1 loss, gradient clipping, SpecAugment, learning-rate decay, and checkpoint selection on the validation split.
- **Ablation and evaluation**: includes a Softmax + SteLLA ablation, reports PLCC, SRCC, RMSE, and MAE, and measures inference performance in FP32 and CUDA AMP.

SteLLA rearranges attention as `ReLU(Q) @ (ReLU(K).T @ V)`, avoiding an explicit N×N matrix in its linear-attention layers. For each attention head, the core computation changes from O(N²d) to O(Nd²). The selected quadratic-attention layers retain O(N²d) computation. The backbone and parameter count remain essentially unchanged.

## Performance and Trade-offs

The following numbers are **historical single-run results** saved during development. They are not results from retraining this cleaned-up release.

| Metric | AST baseline | MixA-k2 + SmoothL1 + Clip |
| --- | ---: | ---: |
| PLCC ↑ | 0.82992 | 0.78334 |
| SRCC ↑ | 0.82716 | 0.77836 |
| RMSE ↓ | 0.56423 | 0.64048 |
| MAE ↓ | 0.43507 | 0.49298 |
| FP32 forward latency ↓ | 38.69 ms | 24.72 ms |
| AMP forward latency ↓ | 20.56 ms | 9.30 ms |
| AMP peak allocated GPU memory ↓ | 623.33 MiB | 576.53 MiB |

**The main gain is efficiency**: approximately **2.21×** speedup with AMP (**54.7%** lower latency), **1.56×** speedup in FP32, and **7.5%** lower peak allocated GPU memory with AMP. Correlations decrease and prediction errors increase, so these results show a speed–accuracy trade-off rather than an accuracy improvement.

Accuracy was evaluated on 712 test utterances. Timing used a fixed subset of 100 test samples, batch size 1, and precomputed 1024×128 filterbanks, measuring model forward passes only. The recorded environment was an RTX 5090 Laptop GPU with PyTorch 2.8.0+cu128. The source records were `ast_test_result.json`, `mixa_k2_smoothl1_clip_epoch20_test_result.json`, and the corresponding `*_bench_fp32.json` / `*_bench_amp.json` files. These experiment artifacts are not distributed in this minimal source repository.

The historical AST result uses its validation-best checkpoint at epoch 9. The MixA result in the table uses the final checkpoint at epoch 20, although its validation-best checkpoint was epoch 17; their checkpoint selection rules therefore differ. Historical normalization constants were computed from the training and validation splits. This release defaults to training-only statistics, and the workflow below uses validation-best checkpoints for both models. The pipeline, random seed, and benchmark subset may differ from the historical experiments; retrain and remeasure before making a new comparison.

## Repository Layout

```text
mixa-ast/
├── ast_regression/       # AST / MixA, data loading, training, scoring, evaluation and benchmarking
├── scripts/
│   ├── prepare_metadata.py
│   └── extract_fbank.py
├── requirements.txt
├── LICENSE              # Original AST BSD-3-Clause notice
├── .gitignore
├── README.md            # Chinese documentation
└── README_EN.md         # English documentation
```

`mixa_ast_models_regression.py` contains the stable attention implementation used for training. `mixa_attention.py` captures attention matrices and gradients for layer scoring, preserving the historical scoring formulas. `common.py` centralizes normalization settings and random seeds. The release contains source code, dependencies, READMEs, and the required license; it excludes datasets, weights, reports, papers, videos, virtual environments, and previous experiment directories.

## Installation

Python 3.10 or 3.11 is recommended. Run the following from the repository root:

```sh
python -m venv .venv
```

Activate the environment with `.venv/Scripts/Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS.

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To use the CUDA 12.8 build from the original environment, install the matching PyTorch packages first, then install the remaining dependencies:

```sh
python -m pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

`timm==0.4.5` is retained to match AST's DeiT implementation and checkpoint structure. Baseline training downloads ImageNet-pretrained weights by default. For offline debugging, add `--imagenet-pretrain false`; training from scratch does not correspond to the experiments in the table. Subsequent calibration, fine-tuning, and evaluation load local checkpoints directly.

## Data and Training

The commands below use `/` path separators and work in Windows PowerShell and common Unix shells. Replace `/path/to/NISQA-CORPUS` with your local dataset directory, preserving the original subset CSV files and audio directory structure.

### 1. Prepare Features

```sh
python scripts/prepare_metadata.py --nisqa-root /path/to/NISQA-CORPUS --output-csv data/metadata.csv --output-json data/splits.json
python scripts/extract_fbank.py --input-csv data/metadata.csv --output-dir data/fbank --target-sr 16000 --target-length 1024
```

The frontend uses 16 kHz mono audio, 128 mel bins, a 25 ms window, a 10 ms hop, and Kaldi's `hanning` window. Features are truncated or zero-padded to 1024 frames. Outputs include `fbank_train.csv`, `fbank_val.csv`, `fbank_test.csv`, `fbank_test_bench100.csv`, and `fbank_stats.json`. The benchmark CSV contains the first 100 entries of the current test manifest and may differ from the historical benchmark subset.

Normalization uses `(x - mean) / (2 * std)`. By default, statistics are computed from the training split, including padding. Pass the same `--stats-json` file to subsequent commands. `--stats-split all` is provided only for inspecting the legacy statistics procedure; keep the default `train` for new experiments. Generated manifests contain local absolute paths, so regenerate them after moving the data directory.

### 2. Train the AST Baseline

```sh
python ast_regression/run_regression.py --train-csv data/fbank/fbank_train.csv --val-csv data/fbank/fbank_val.csv --stats-json data/fbank/fbank_stats.json --exp-dir exp/ast --lr 1e-5 --batch-size 12 --num-workers 0 --n-epochs 20 --loss MSE
```

Default frequency/time masking settings are 24/192. The default random seed is 42 and can be changed with `--seed`. Training saves configuration in `args.json`, per-epoch metrics, and checkpoints. The checkpoint with the highest validation SRCC is saved as `models/best_audio_model.pth`, with RMSE used to help resolve close SRCC values. A fixed seed does not guarantee bitwise-identical results across devices.

### 3. Calibrate Layer Importance (Optional)

```sh
python ast_regression/scoring.py --calib-csv data/fbank/fbank_train.csv --baseline-ckpt exp/ast/models/best_audio_model.pth --stats-json data/fbank/fbank_stats.json --output-json exp/topk.json --k 2 --batch-size 1 --num-workers 0 --max-batches 64 --loss SmoothL1
```

Scoring explicitly stores N×N attention matrices and gradients, so start with batch size 1 to limit memory use. The output field `topk_layers_arg` contains zero-based layer indices for training and evaluation. To use the new ranking, replace every occurrence of `0,10` in the following commands with that value. Keeping `0,10` trains the historical k=2 architecture.

### 4. Fine-tune MixA

```sh
python ast_regression/run_mixa_regression.py --train-csv data/fbank/fbank_train.csv --val-csv data/fbank/fbank_val.csv --stats-json data/fbank/fbank_stats.json --exp-dir exp/mixa --baseline-ckpt exp/ast/models/best_audio_model.pth --topk-layers 0,10 --hybrid-mode mixa --lr 5e-6 --batch-size 12 --num-workers 0 --n-epochs 20 --loss SmoothL1 --clip-grad 1.0
```

Use `--hybrid-mode softmax_stella` to train the ablation that retains softmax attention in the selected layers. Reduce batch size if GPU memory is insufficient. Increase `--num-workers` once data loading is working reliably.

### 5. Evaluate and Benchmark

```sh
python ast_regression/evaluate_regression_on_test.py --test-csv data/fbank/fbank_test.csv --ckpt exp/mixa/models/best_audio_model.pth --model-type mixa --topk-layers 0,10 --stats-json data/fbank/fbank_stats.json --output-json exp/mixa/test.json --save-preds exp/mixa/test_predictions.csv
python ast_regression/benchmark_on_testcase.py --test-csv data/fbank/fbank_test_bench100.csv --ckpt exp/mixa/models/best_audio_model.pth --model-type mixa --topk-layers 0,10 --stats-json data/fbank/fbank_stats.json --batch-size 1 --precision amp --output-json exp/mixa/benchmark_amp.json
```

Use `--precision fp32` for CPU or FP32 measurements. To evaluate AST, set `--model-type ast`, select the AST checkpoint, and use separate output files. For the ablation, use `--model-type softmax_stella`. Keep input dimensions, sample subset, device, batch size, and precision consistent across comparisons. Benchmark JSON records the hardware, input dimensions, and warmup settings. Timing starts after input transfer completes and excludes audio loading and feature extraction.

## Publishing and Attribution

Use this directory as the GitHub repository root and run Git commands from here. `.gitignore` excludes local data, weights, training outputs, document attachments, and caches.

```sh
git init -b main
git add .
git commit -m "Add MixA-AST regression implementation"
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

The AST/DeiT adaptation is based on the upstream AST implementation (Yuan Gong, 2021), with its original BSD-3-Clause license retained. MixA and SteLLA are existing methods adapted here to speech quality regression. This project's contributions are task adaptation, training stability, layer selection, and efficiency evaluation; it does not claim to have originated these attention methods.
