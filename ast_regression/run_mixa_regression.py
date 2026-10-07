import json
from common import add_normalization_args, resolve_normalization, seed_everything, parse_topk_layers, strip_module_prefix_if_needed
# run_mixa_regression.py
import argparse
import os
import torch

from dataloader_regression import NISQAFbankDataset
from mixa_ast_models_regression import ASTMixARegressor, ASTSoftmaxSteLLARegressor
from traintest_regression import train


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train-csv', type=str, required=True)
    parser.add_argument('--val-csv', type=str, required=True)
    parser.add_argument('--exp-dir', type=str, required=True)

    parser.add_argument('--baseline-ckpt', type=str, required=True,
                        help='Path to trained AST regression baseline checkpoint.')
    parser.add_argument('--topk-layers', type=str, required=True,
                        help='Comma-separated zero-based layer indices, e.g. "0,1,3,5,9,10"')

    parser.add_argument('--hybrid-mode', type=str, default='mixa',
                    choices=['mixa', 'softmax_stella'],
                    help='mixa = top-k ReLUQuadratic + others SteLLA; softmax_stella = top-k original softmax + others SteLLA')

    parser.add_argument('--dataset', type=str, default='nisqa')
    parser.add_argument('--model', type=str, default='mixa_ast_regression')

    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--weight-decay', type=float, default=5e-7)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--n-epochs', type=int, default=20)
    parser.add_argument('--n-print-steps', type=int, default=100)

    parser.add_argument('--freqm', type=int, default=24)
    parser.add_argument('--timem', type=int, default=192)

    parser.add_argument('--fstride', type=int, default=10)
    parser.add_argument('--tstride', type=int, default=10)
    parser.add_argument('--input-fdim', type=int, default=128)
    parser.add_argument('--input-tdim', type=int, default=1024)

    parser.add_argument('--imagenet-pretrain', type=lambda x: x.lower() == 'true', default=False)


    parser.add_argument('--loss', type=str, default='MSE', choices=['MSE', 'L1', 'SmoothL1'])
    parser.add_argument('--clip-grad', type=float, default=1.0,
                        help='Max gradient norm. Set <=0 to disable gradient clipping.')

    parser.add_argument('--lrscheduler-start', type=int, default=5)
    parser.add_argument('--lrscheduler-step', type=int, default=1)
    parser.add_argument('--lrscheduler-decay', type=float, default=0.85)

    add_normalization_args(parser)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    resolve_normalization(args, parser)
    seed_everything(args.seed)

    topk_layers = parse_topk_layers(args.topk_layers)

    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, 'models'), exist_ok=True)

    with open(os.path.join(args.exp_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)

    train_audio_conf = {
        'freqm': args.freqm,
        'timem': args.timem,
        'mean': args.dataset_mean,
        'std': args.dataset_std,
        'skip_norm': args.skip_norm,
        'mode': 'train',
    }

    val_audio_conf = {
        'freqm': 0,
        'timem': 0,
        'mean': args.dataset_mean,
        'std': args.dataset_std,
        'skip_norm': args.skip_norm,
        'mode': 'eval',
    }

    train_loader = torch.utils.data.DataLoader(
        NISQAFbankDataset(args.train_csv, train_audio_conf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = torch.utils.data.DataLoader(
        NISQAFbankDataset(args.val_csv, val_audio_conf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if args.hybrid_mode == 'mixa':
        model = ASTMixARegressor(
            topk_layers=topk_layers,
            fstride=args.fstride,
            tstride=args.tstride,
            input_fdim=args.input_fdim,
            input_tdim=args.input_tdim,
            imagenet_pretrain=args.imagenet_pretrain,

            model_size='base384',
            verbose=True,
        )
    elif args.hybrid_mode == 'softmax_stella':
        model = ASTSoftmaxSteLLARegressor(
            topk_layers=topk_layers,
            fstride=args.fstride,
            tstride=args.tstride,
            input_fdim=args.input_fdim,
            input_tdim=args.input_tdim,
            imagenet_pretrain=args.imagenet_pretrain,

            model_size='base384',
            verbose=True,
        )
    else:
        raise ValueError(f"Unknown hybrid_mode: {args.hybrid_mode}")

    # load trained AST regression baseline weights
    print('\nLoading AST baseline checkpoint from:', args.baseline_ckpt)
    ckpt = torch.load(args.baseline_ckpt, map_location='cpu')
    ckpt = strip_module_prefix_if_needed(ckpt)

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if unexpected or any(not key.endswith('_raw') for key in missing):
        raise ValueError(f'Incompatible AST checkpoint: missing={missing}, unexpected={unexpected}')
    print('Baseline weights loaded; new attention scales retain their initialization.')
    print('Missing keys   :', missing)
    print('Unexpected keys:', unexpected)

    print('\nCreating experiment directory:', args.exp_dir)
    print('Now starting MixA-AST regression fine-tuning...')
    train(model, train_loader, val_loader, args)


if __name__ == '__main__':
    main()
