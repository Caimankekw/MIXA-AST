from common import add_normalization_args, resolve_normalization, seed_everything, parse_topk_layers, strip_module_prefix_if_needed
# evaluate_regression_on_test.py
import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader_regression import NISQAFbankDataset
from ast_models_regression import ASTRegressor
from mixa_ast_models_regression import ASTMixARegressor, ASTSoftmaxSteLLARegressor
from stats_regression import calculate_regression_stats


def build_model(args, device):
    if args.model_type == "ast":
        model = ASTRegressor(
            fstride=args.fstride,
            tstride=args.tstride,
            input_fdim=args.input_fdim,
            input_tdim=args.input_tdim,
            imagenet_pretrain=args.imagenet_pretrain,

            model_size="base384",
            verbose=False,
        )

    elif args.model_type == "mixa":
        model = ASTMixARegressor(
            topk_layers=parse_topk_layers(args.topk_layers),
            fstride=args.fstride,
            tstride=args.tstride,
            input_fdim=args.input_fdim,
            input_tdim=args.input_tdim,
            imagenet_pretrain=args.imagenet_pretrain,

            model_size="base384",
            verbose=False,
        )

    elif args.model_type == "softmax_stella":
        model = ASTSoftmaxSteLLARegressor(
            topk_layers=parse_topk_layers(args.topk_layers),
            fstride=args.fstride,
            tstride=args.tstride,
            input_fdim=args.input_fdim,
            input_tdim=args.input_tdim,
            imagenet_pretrain=args.imagenet_pretrain,

            model_size="base384",
            verbose=False,
        )

    else:
        raise ValueError("model_type must be 'ast', 'mixa', 'softmax_stella'")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt = strip_module_prefix_if_needed(ckpt)
    model.load_state_dict(ckpt, strict=True)

    print("Loaded checkpoint:", args.ckpt)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    preds = []
    targets = []

    for audio_input, mos in loader:
        audio_input = audio_input.to(device, non_blocking=True).float()
        mos = mos.to(device, non_blocking=True).float()

        pred = model(audio_input)  # [B]
        preds.append(pred.detach().cpu().numpy())
        targets.append(mos.detach().cpu().numpy())

    pred_arr = np.concatenate(preds, axis=0).reshape(-1)
    target_arr = np.concatenate(targets, axis=0).reshape(-1)

    stats = calculate_regression_stats(pred_arr, target_arr)
    return stats, pred_arr, target_arr


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["ast", "mixa", "softmax_stella"]
    )
    parser.add_argument("--topk-layers", type=str, default="",
                        help='Only used when model-type=mixa, e.g. "0,1,3,5,9,10"')

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)


    parser.add_argument("--fstride", type=int, default=10)
    parser.add_argument("--tstride", type=int, default=10)
    parser.add_argument("--input-fdim", type=int, default=128)
    parser.add_argument("--input-tdim", type=int, default=1024)
    parser.add_argument("--imagenet-pretrain", type=lambda x: x.lower() == "true", default=False)

    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--save-preds", type=str, default="",
                        help="Optional path to save predictions csv")

    add_normalization_args(parser)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    resolve_normalization(args, parser)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

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
    stats, pred_arr, target_arr = evaluate(model, test_loader, device)

    result = {
        "test_csv": args.test_csv,
        "ckpt": args.ckpt,
        "model_type": args.model_type,
        "topk_layers": parse_topk_layers(args.topk_layers),
        "seed": args.seed,
        "dataset_mean": args.dataset_mean,
        "dataset_std": args.dataset_std,
        "skip_norm": args.skip_norm,
        "PLCC": stats["PLCC"],
        "SRCC": stats["SRCC"],
        "RMSE": stats["RMSE"],
        "MAE": stats["MAE"],
        "MSE": stats["MSE"],
        "num_samples": int(len(pred_arr)),
    }

    print("\n=== Test Results ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True) if os.path.dirname(args.output_json) else None
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("Saved json to:", args.output_json)

    if args.save_preds:
        os.makedirs(os.path.dirname(args.save_preds), exist_ok=True) if os.path.dirname(args.save_preds) else None
        arr = np.stack([target_arr, pred_arr], axis=1)
        np.savetxt(args.save_preds, arr, delimiter=",", header="target,pred", comments="")
        print("Saved predictions to:", args.save_preds)


if __name__ == "__main__":
    main()
