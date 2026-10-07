"""Shared CLI configuration for training, calibration and evaluation."""
import argparse
import json
import random

import numpy as np
import torch


def add_normalization_args(parser):
    parser.add_argument('--stats-json', help='fbank_stats.json from feature extraction.')
    parser.add_argument('--dataset-mean', type=float)
    parser.add_argument('--dataset-std', type=float)
    parser.add_argument('--skip-norm', type=lambda x: x.lower() == 'true', default=False)


def resolve_normalization(args, parser):
    if args.stats_json:
        if args.dataset_mean is not None or args.dataset_std is not None:
            parser.error('Use --stats-json OR --dataset-mean/--dataset-std.')
        with open(args.stats_json, encoding='utf-8') as stream:
            stats = json.load(stream)
        args.dataset_mean = float(stats['dataset_fbank_mean'])
        args.dataset_std = float(stats['dataset_fbank_std'])
    if args.skip_norm:
        args.dataset_mean, args.dataset_std = 0.0, 1.0
    if args.dataset_mean is None or args.dataset_std is None:
        parser.error('Supply --stats-json or both --dataset-mean and --dataset-std.')
    if not np.isfinite(args.dataset_mean) or not np.isfinite(args.dataset_std) or args.dataset_std <= 0:
        parser.error('Normalization requires a finite mean and positive finite std.')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_topk_layers(value):
    value = value.strip()
    if not value:
        return []
    try:
        layers = [int(item.strip()) for item in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated zero-based layer indices.') from exc
    if len(set(layers)) != len(layers) or any(layer < 0 or layer >= 12 for layer in layers):
        raise argparse.ArgumentTypeError('Layer indices must be unique integers in [0, 11].')
    return sorted(layers)


def strip_module_prefix_if_needed(state_dict):
    return {key.removeprefix('module.'): value for key, value in state_dict.items()}


class AverageMeter:
    """Sample-weighted running average."""
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, value, n=1):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
