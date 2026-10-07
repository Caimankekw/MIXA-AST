from common import add_normalization_args, resolve_normalization, seed_everything, parse_topk_layers, strip_module_prefix_if_needed
import argparse
import json
import os

import numpy as np
import torch

from ast_models_regression import ASTRegressor
from dataloader_regression import NISQAFbankDataset
from mixa_scoring import compute_importance_scores, select_topk_layers


def main():
    parser = argparse.ArgumentParser(
        description="Compute MixA layer-importance scores from a calibration split."
    )
    parser.add_argument("--calib-csv", type=str, required=True)
    parser.add_argument("--baseline-ckpt", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=64)

    parser.add_argument("--loss", type=str, default="SmoothL1", choices=["MSE", "L1", "SmoothL1"])

    parser.add_argument("--fstride", type=int, default=10)
    parser.add_argument("--tstride", type=int, default=10)
    parser.add_argument("--input-fdim", type=int, default=128)
    parser.add_argument("--input-tdim", type=int, default=1024)
    parser.add_argument("--imagenet-pretrain", type=lambda x: x.lower() == "true", default=False)
    add_normalization_args(parser)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    resolve_normalization(args, parser)
    seed_everything(args.seed)

    if not 1 <= args.k <= 12 or args.max_batches < 1:
        parser.error('--k must be in [1, 12] and --max-batches must be positive.')
    torch.set_grad_enabled(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    calib_audio_conf = {
        "freqm": 0,
        "timem": 0,
        "mean": args.dataset_mean,
        "std": args.dataset_std,
        "skip_norm": args.skip_norm,
        "mode": "eval",
    }
    calib_loader = torch.utils.data.DataLoader(
        NISQAFbankDataset(args.calib_csv, calib_audio_conf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = ASTRegressor(
        fstride=args.fstride,
        tstride=args.tstride,
        input_fdim=args.input_fdim,
        input_tdim=args.input_tdim,
        imagenet_pretrain=args.imagenet_pretrain,

        model_size="base384",
        verbose=True,
    )

    ckpt = torch.load(args.baseline_ckpt, map_location=device)
    ckpt = strip_module_prefix_if_needed(ckpt)
    model.load_state_dict(ckpt, strict=True)
    model.to(device)
    model.eval()

    scores = compute_importance_scores(
        baseline_model=model,
        calib_loader=calib_loader,
        device=device,
        loss_name=args.loss,
        max_batches=args.max_batches,
    )

    topk_layers, full_order = select_topk_layers(scores, k=args.k)
    scores = np.asarray(scores, dtype=np.float64)
    full_order = np.asarray(full_order, dtype=np.int64)
    topk_layers = np.asarray(topk_layers, dtype=np.int64)

    payload = {
        "calib_csv": args.calib_csv,
        "baseline_ckpt": args.baseline_ckpt,
        "loss": args.loss,
        "seed": args.seed,
        "skip_norm": args.skip_norm,
        "k": int(args.k),
        "batch_size": int(args.batch_size),
        "max_batches": int(args.max_batches),
        "num_calibration_samples_cap": int(args.batch_size * args.max_batches),
        "dataset_mean": float(args.dataset_mean),
        "dataset_std": float(args.dataset_std),
        "scores_zero_based": {str(i): float(scores[i]) for i in range(len(scores))},
        "scores_one_based": {str(i + 1): float(scores[i]) for i in range(len(scores))},
        "full_order_zero_based": [int(x) for x in full_order.tolist()],
        "full_order_one_based": [int(x + 1) for x in full_order.tolist()],
        "topk_zero_based": [int(x) for x in topk_layers.tolist()],
        "topk_one_based": [int(x + 1) for x in topk_layers.tolist()],
        "topk_layers_arg": ",".join(str(int(x)) for x in topk_layers.tolist()),
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True) if os.path.dirname(args.output_json) else None
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("Layer importance scores:", scores)
    print("Full order zero-based:", payload["full_order_zero_based"])
    print("Top-k zero-based:", payload["topk_zero_based"])
    print("Top-k one-based:", payload["topk_one_based"])
    print("Top-k arg:", payload["topk_layers_arg"])
    print("Saved:", args.output_json)


if __name__ == "__main__":
    main()
