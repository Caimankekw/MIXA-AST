from common import add_normalization_args, resolve_normalization, seed_everything, parse_topk_layers, strip_module_prefix_if_needed
# run_regression.py
import argparse
import os
import json
import numpy as np
import torch

from dataloader_regression import NISQAFbankDataset
from ast_models_regression import ASTRegressor
from traintest_regression import train


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train-csv', type=str, required=True)
    parser.add_argument('--val-csv', type=str, required=True)
    parser.add_argument('--exp-dir', type=str, required=True)

    parser.add_argument('--dataset', type=str, default='nisqa')
    parser.add_argument('--model', type=str, default='ast_regression')

    parser.add_argument('--lr', type=float, default=1e-5)
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

    parser.add_argument('--imagenet-pretrain', type=lambda x: x.lower() == 'true', default=True)


    parser.add_argument('--loss', type=str, default='MSE', choices=['MSE', 'L1', 'SmoothL1'])
    parser.add_argument('--clip-grad', type=float, default=0.0,
                        help='Max gradient norm. Set <=0 to disable gradient clipping.')

    parser.add_argument('--lrscheduler-start', type=int, default=5)
    parser.add_argument('--lrscheduler-step', type=int, default=1)
    parser.add_argument('--lrscheduler-decay', type=float, default=0.85)

    add_normalization_args(parser)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    resolve_normalization(args, parser)
    seed_everything(args.seed)

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
        batch_size=args.batch_size * 2,
        shuffle=False,
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

        model_size='base384',
        verbose=True,
    )

    print('\nCreating experiment directory:', args.exp_dir)
    print('Now starting regression training...')
    train(model, train_loader, val_loader, args)


if __name__ == '__main__':
    main()
