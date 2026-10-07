# MixA-AST：语音质量回归与混合注意力加速

**简体中文** | [English](README_EN.md)

基于 Audio Spectrogram Transformer（AST）的无参考语音质量评估项目：输入语音的 log-mel / Kaldi fbank，输出一个 MOS（Mean Opinion Score）预测值。在 NISQA-CORPUS 上实现 AST 基线、MixA 混合注意力及训练与评估流程。

## 实现了什么

- **AST 回归**：将 DeiT/AST 分类结构适配为标量回归，平均两个特殊 token 后接 LayerNorm + Linear。
- **MixA 混合注意力**：在选定层使用 ReLU 二次注意力，其余层使用 SteLLA 线性注意力。历史 k=2 配置选择第 **1、11 层**（零基索引 `0,10`）。
- **层重要性评分**：在训练集校准样本上，用注意力梯度与替换前后注意力的差异评估层敏感度，选择 top-k。
- **稳定训练**：MixA 缩放因子采用 `softplus(raw) + 1e-4` 保持正值，支持 SmoothL1、梯度裁剪、SpecAugment、学习率衰减和验证集 checkpoint 选择。
- **对照与评估**：保留 Softmax + SteLLA 消融；输出 PLCC、SRCC、RMSE、MAE，并支持 FP32 / CUDA AMP 推理测量。

SteLLA 将注意力计算重排为 `ReLU(Q) @ (ReLU(K).T @ V)`，避免在线性注意力层显式构造 N×N 矩阵。对于每个注意力头，核心计算从 O(N²d) 变为 O(Nd²)；选定的二次注意力层仍保留 O(N²d) 计算。骨干网络与参数规模基本不变。

## 有什么提升

下面是原开发实验保存的**单次历史记录**，不是此次整理后重新训练的结果：

| 指标 | AST baseline | MixA-k2 + SmoothL1 + Clip |
| --- | ---: | ---: |
| PLCC ↑ | 0.82992 | 0.78334 |
| SRCC ↑ | 0.82716 | 0.77836 |
| RMSE ↓ | 0.56423 | 0.64048 |
| MAE ↓ | 0.43507 | 0.49298 |
| FP32 前向延迟 ↓ | 38.69 ms | 24.72 ms |
| AMP 前向延迟 ↓ | 20.56 ms | 9.30 ms |
| AMP 峰值分配显存 ↓ | 623.33 MiB | 576.53 MiB |

**收益主要是效率**：AMP 加速约 **2.21×**（延迟降低 **54.7%**），FP32 加速约 **1.56×**，AMP 显存降低约 **7.5%**。相关性指标下降、误差增大，因此这是速度与精度的取舍，不能解释为精度提升。

记录条件：精度评估 712 条测试语音；速度评估固定 100 条测试样本、batch size=1、1024×128 预计算 fbank，仅测模型前向。原环境记录为 RTX 5090 Laptop GPU、PyTorch 2.8.0+cu128。结果来自 `ast_test_result.json`、`mixa_k2_smoothl1_clip_epoch20_test_result.json` 及对应 `*_bench_fp32.json` / `*_bench_amp.json`；为保持源码仓库精简，这些实验产物不随仓库分发。

历史 AST 使用验证最优 epoch 9；MixA 表中使用最终 epoch 20，其验证最优为 epoch 17，两者 checkpoint 选择规则不同。旧归一化常数来自训练集与验证集；本版本默认仅统计训练集，并统一使用验证最优 checkpoint。新流程、随机种子及基准子集均可能与历史实验不同，应重新训练和测量后再比较。

## 目录

```text
mixa-ast/
├── ast_regression/       # AST / MixA、数据加载、训练、评分、评估与测速
├── scripts/
│   ├── prepare_metadata.py
│   └── extract_fbank.py
├── requirements.txt
├── LICENSE              # 保留 AST 上游 BSD-3-Clause 声明
├── .gitignore
├── README.md            # 中文说明
└── README_EN.md         # English documentation
```

`mixa_ast_models_regression.py` 是训练使用的稳定注意力实现；`mixa_attention.py` 用于层评分时保存注意力与梯度，保留历史评分公式。`common.py` 统一归一化配置和随机种子。发布目录仅包含源码、依赖、README 和必要许可证；数据、权重、报告、论文、视频、虚拟环境与旧实验目录均不包含。

## 安装

建议 Python 3.10 / 3.11。从本项目根目录运行：

```sh
python -m venv .venv
```

Windows PowerShell 激活：`.venv/Scripts/Activate.ps1`；Linux/macOS 激活：`source .venv/bin/activate`。

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如需使用原实验的 CUDA 12.8 构建，可先安装匹配的 PyTorch 三件套，再安装其余依赖：

```sh
python -m pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

保留 `timm==0.4.5` 是为了匹配 AST 的 DeiT 实现和 checkpoint 结构。基线默认下载 ImageNet 预训练权重；离线调试可以加 `--imagenet-pretrain false`，但从零训练结果不对应表中实验。后续校准、微调与评估直接加载本地 checkpoint。

## 数据与训练

以下命令均使用 `/` 路径，Windows PowerShell 与常见 Unix shell 都可运行。将 `/path/to/NISQA-CORPUS` 替换为本地数据目录；保留原始子集 CSV 与音频结构。

### 1. 准备特征

```sh
python scripts/prepare_metadata.py --nisqa-root /path/to/NISQA-CORPUS --output-csv data/metadata.csv --output-json data/splits.json
python scripts/extract_fbank.py --input-csv data/metadata.csv --output-dir data/fbank --target-sr 16000 --target-length 1024
```

前端使用 16 kHz 单声道、128 mel bins、25 ms 窗口、10 ms 步长、Kaldi `hanning` 窗，截断或补零到 1024 帧。生成 `fbank_train.csv`、`fbank_val.csv`、`fbank_test.csv`、`fbank_test_bench100.csv` 和 `fbank_stats.json`。基准 CSV 取当前测试清单的前 100 条，不保证等同于历史实验子集。

归一化使用 `(x - mean) / (2 * std)`，统计量默认来自训练集（含补零）。后续统一传入同一份 `--stats-json`。`--stats-split all` 仅用于检查旧统计流程；新实验保持默认 `train`。生成清单使用本机绝对路径，迁移数据目录后需重新生成清单。

### 2. 训练 AST 基线

```sh
python ast_regression/run_regression.py --train-csv data/fbank/fbank_train.csv --val-csv data/fbank/fbank_val.csv --stats-json data/fbank/fbank_stats.json --exp-dir exp/ast --lr 1e-5 --batch-size 12 --num-workers 0 --n-epochs 20 --loss MSE
```

默认频率/时间遮挡为 24/192，随机种子为 42，可用 `--seed` 修改。训练保存参数 `args.json`、每轮指标及权重；验证 SRCC 最高的模型写入 `models/best_audio_model.pth`，SRCC 接近时以 RMSE 辅助选择。随机种子不能保证不同设备间逐位一致。

### 3. 校准层重要性（可选）

```sh
python ast_regression/scoring.py --calib-csv data/fbank/fbank_train.csv --baseline-ckpt exp/ast/models/best_audio_model.pth --stats-json data/fbank/fbank_stats.json --output-json exp/topk.json --k 2 --batch-size 1 --num-workers 0 --max-batches 64 --loss SmoothL1
```

该步骤显式保存 N×N 注意力及梯度，内存开销较大，先使用 batch size=1。输出 `topk_layers_arg` 是可传给训练与评估的零基层索引。若使用新评分结果，将下列命令中的 `0,10` 统一替换为该值；保留 `0,10` 则训练历史 k=2 结构。

### 4. 微调 MixA

```sh
python ast_regression/run_mixa_regression.py --train-csv data/fbank/fbank_train.csv --val-csv data/fbank/fbank_val.csv --stats-json data/fbank/fbank_stats.json --exp-dir exp/mixa --baseline-ckpt exp/ast/models/best_audio_model.pth --topk-layers 0,10 --hybrid-mode mixa --lr 5e-6 --batch-size 12 --num-workers 0 --n-epochs 20 --loss SmoothL1 --clip-grad 1.0
```

使用 `--hybrid-mode softmax_stella` 可训练选定层保留 softmax 的消融版本。显存不足时降低 batch size；数据加载稳定后可增加 `--num-workers`。

### 5. 测试与测量效率

```sh
python ast_regression/evaluate_regression_on_test.py --test-csv data/fbank/fbank_test.csv --ckpt exp/mixa/models/best_audio_model.pth --model-type mixa --topk-layers 0,10 --stats-json data/fbank/fbank_stats.json --output-json exp/mixa/test.json --save-preds exp/mixa/test_predictions.csv
python ast_regression/benchmark_on_testcase.py --test-csv data/fbank/fbank_test_bench100.csv --ckpt exp/mixa/models/best_audio_model.pth --model-type mixa --topk-layers 0,10 --stats-json data/fbank/fbank_stats.json --batch-size 1 --precision amp --output-json exp/mixa/benchmark_amp.json
```

CPU 或 FP32 测量使用 `--precision fp32`。比较 AST 时设置 `--model-type ast`、换成 AST checkpoint，并写入不同输出文件。比较消融版本时使用 `--model-type softmax_stella`。保持输入、子集、设备、batch size 与精度一致；结果 JSON 保存硬件、输入尺寸和预热参数。当前计时在输入传输完成后开始，不包含音频加载和特征提取。

## 发布与来源

本目录可以作为 GitHub 仓库根目录；只在此目录执行 Git 命令。`.gitignore` 已排除本地数据、权重、训练输出、文档附件和缓存。

```sh
git init -b main
git add .
git commit -m "Add MixA-AST regression implementation"
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

AST/DeiT 适配代码基于 AST 上游实现（Yuan Gong，2021），保留原 BSD-3-Clause 许可证。MixA / SteLLA 是本项目移植到语音质量回归任务中的已有方法；本项目贡献是任务适配、稳定训练、层选择与效率评估，不将这些注意力方法表述为原创。
