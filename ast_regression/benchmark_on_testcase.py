from common import add_normalization_args, resolve_normalization, seed_everything, parse_topk_layers, strip_module_prefix_if_needed
# benchmark_on_testcase.py
import argparse
import json
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader_regression import NISQAFbankDataset
from evaluate_regression_on_test import build_model


def get_precision_context(device, precision: str):
    if precision == "fp32":
        return nullcontext()
    elif precision == "amp":
        if device.type != "cuda":
            return nullcontext()
        return torch.amp.autocast(device_type="cuda")
    else:
        raise ValueError(f"Unknown precision: {precision}")


@torch.no_grad()
def benchmark_on_loader(model, loader, device, precision="fp32", warmup=5):
    """
    Measure real testcase inference time.
    """
    # warmup on first few batches
    warmup_done = 0
    for audio_input, _ in loader:
        if warmup_done >= warmup:
            break
        audio_input = audio_input.to(device, non_blocking=True).float()
        with get_precision_context(device, precision):
            _ = model(audio_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        warmup_done += 1
        if warmup_done >= warmup:
            break

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    batch_times_ms = []
    total_samples = 0

    for audio_input, _ in loader:
        audio_input = audio_input.to(device, non_blocking=True).float()
        bs = audio_input.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with get_precision_context(device, precision):
            _ = model(audio_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        batch_times_ms.append((end - start) * 1000.0)
        total_samples += bs

    total_time_sec = float(np.sum(batch_times_ms) / 1000.0)
    mean_batch_latency_ms = float(np.mean(batch_times_ms))
    std_batch_latency_ms = float(np.std(batch_times_ms))
    throughput = float(total_samples / total_time_sec) if total_time_sec > 0 else None

    if device.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    else:
        peak_memory_mb = None

    return {
        "num_batches": int(len(batch_times_ms)),
        "num_samples": int(total_samples),
        "mean_batch_latency_ms": mean_batch_latency_ms,
        "std_batch_latency_ms": std_batch_latency_ms,
        "throughput_samples_per_sec": throughput,
        "peak_memory_mb": peak_memory_mb,
        "total_time_sec": total_time_sec,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-csv", type=str, required=True,
                        help="Use a fixed testcase csv, e.g. fbank_test_bench100.csv")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["ast", "mixa", "softmax_stella"]
    )
    parser.add_argument("--topk-layers", type=str, default="",
                        help='Only used when model-type=mixa, e.g. "0,1,3,5,9,10"')

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "amp"])
    parser.add_argument("--warmup", type=int, default=5)


    parser.add_argument("--fstride", type=int, default=10)
    parser.add_argument("--tstride", type=int, default=10)
    parser.add_argument("--input-fdim", type=int, default=128)
    parser.add_argument("--input-tdim", type=int, default=1024)
    parser.add_argument("--imagenet-pretrain", type=lambda x: x.lower() == "true", default=False)

    parser.add_argument("--output-json", type=str, default="")
    add_normalization_args(parser)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    resolve_normalization(args, parser)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.precision == 'amp' and device.type != 'cuda':
        parser.error('--precision amp requires CUDA; use fp32 on CPU.')
    if args.warmup < 0:
        parser.error('--warmup must be nonnegative.')
    print("Device:", device)
    print("Precision:", args.precision)

    test_audio_conf = {
        "freqm": 0,
        "timem": 0,
        "mean": args.dataset_mean,
        "std": args.dataset_std,
        "skip_norm": args.skip_norm,
        "mode": "eval",
    }

    test_loader = DataLoader(
        NISQAFbankDataset(args.test_csv, test_audio_conf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(args, device)
    stats = benchmark_on_loader(
        model=model,
        loader=test_loader,
        device=device,
        precision=args.precision,
        warmup=args.warmup,
    )

    result = {
        "test_csv": args.test_csv,
        "ckpt": args.ckpt,
        "model_type": args.model_type,
        "topk_layers": parse_topk_layers(args.topk_layers),
        "seed": args.seed,
        "dataset_mean": args.dataset_mean,
        "dataset_std": args.dataset_std,
        "skip_norm": args.skip_norm,
        "precision": args.precision,
        "warmup": args.warmup,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "input_shape": [args.input_tdim, args.input_fdim],
        "timing_scope": "model forward after input transfer; excludes feature extraction",
        "batch_size": args.batch_size,
        **stats
    }

    print("\n=== Benchmark Results ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True) if os.path.dirname(args.output_json) else None
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("Saved json to:", args.output_json)


if __name__ == "__main__":
    main()
