# mixa_scoring.py
import copy
import torch
import torch.nn as nn
import numpy as np

from mixa_attention import (
    ScoreableSoftmaxAttention,
    SteLLAAttention,
)


def build_loss(loss_name: str):
    if loss_name == "MSE":
        return nn.MSELoss()
    elif loss_name == "L1":
        return nn.L1Loss()
    elif loss_name == "SmoothL1":
        return nn.SmoothL1Loss(beta=1.0)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def make_scoreable_softmax_model(model: nn.Module) -> nn.Module:
    """
    Replace every blk.attn with ScoreableSoftmaxAttention
    so we can access attention matrix and its gradient.
    """
    model = copy.deepcopy(model)
    for blk in model.v.blocks:
        blk.attn = ScoreableSoftmaxAttention(blk.attn, store_attention=True)
    return model


def compute_importance_scores(
    baseline_model: nn.Module,
    calib_loader,
    device: torch.device,
    loss_name: str = "MSE",
    max_batches: int = 8,
):
    """
    Implements MixA-style importance scoring on an AST regressor.

    For each layer l:
      1) get original A_l and grad wrt A_l from baseline
      2) replace only layer l with SteLLA, get A'_l
      3) score_l += mean_{batch,head} | Tr( (grad A_l)^T (A'_l - A_l) ) |

    Returns:
      scores: np.ndarray shape [num_layers]
    """
    loss_fn = build_loss(loss_name)

    # baseline with scoreable softmax attentions
    base_model = make_scoreable_softmax_model(baseline_model).to(device)
    base_model.eval()

    num_layers = len(base_model.v.blocks)
    score_sum = np.zeros(num_layers, dtype=np.float64)
    score_count = np.zeros(num_layers, dtype=np.int64)

    for batch_idx, (audio_input, mos) in enumerate(calib_loader):
        if batch_idx >= max_batches:
            break

        audio_input = audio_input.to(device, non_blocking=True).float()
        mos = mos.to(device, non_blocking=True).float()

        # ----- original forward/backward -----
        base_model.zero_grad(set_to_none=True)
        pred = base_model(audio_input)
        loss = loss_fn(pred, mos)
        loss.backward()

        original_attn = []
        original_grad = []

        for l in range(num_layers):
            A = base_model.v.blocks[l].attn.last_attn
            G = base_model.v.blocks[l].attn.last_attn.grad

            if A is None or G is None:
                raise RuntimeError(f"Layer {l}: attention or grad was not captured.")

            original_attn.append(A.detach().cpu())
            original_grad.append(G.detach().cpu())

        # ----- one-layer replacement with SteLLA -----
        for l in range(num_layers):
            mod_model = copy.deepcopy(baseline_model).to(device)
            mod_model.eval()

            # Replace only layer l with SteLLA in scoring mode (materialize A')
            mod_model.v.blocks[l].attn = SteLLAAttention(mod_model.v.blocks[l].attn, store_attention=True)

            with torch.no_grad():
                _ = mod_model(audio_input)

            A_prime = mod_model.v.blocks[l].attn.last_attn
            if A_prime is None:
                raise RuntimeError(f"Layer {l}: modified attention A' not captured.")
            A_prime = A_prime.detach().cpu()

            delta = A_prime - original_attn[l]      # [B, H, N, N]
            grad = original_grad[l]                 # [B, H, N, N]

            # Tr(G^T Δ) == elementwise inner product sum_ij G_ij * Δ_ij
            layer_score = (grad * delta).sum(dim=(-2, -1)).abs().mean().item()

            score_sum[l] += layer_score
            score_count[l] += 1

            del mod_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    scores = score_sum / np.maximum(score_count, 1)
    return scores


def select_topk_layers(scores, k=6):
    """
    Returns zero-based layer indices of top-k most important layers.
    """
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 1 <= k <= len(scores):
        raise ValueError('Expected finite layer scores and 1 <= k <= number of layers.')
    order = np.argsort(scores)[::-1]   # descending
    topk = order[:k]
    return np.sort(topk), order
