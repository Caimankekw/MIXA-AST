"""Attention capture for calibration; training uses mixa_ast_models_regression."""
import math
import copy
import torch
import torch.nn as nn


class _BaseWrappedAttention(nn.Module):
    def __init__(self, orig_attn, store_attention=False):
        super().__init__()
        # copy weights / modules from timm attention
        self.num_heads = orig_attn.num_heads
        self.qkv = copy.deepcopy(orig_attn.qkv)
        self.attn_drop = copy.deepcopy(orig_attn.attn_drop)
        self.proj = copy.deepcopy(orig_attn.proj)
        self.proj_drop = copy.deepcopy(orig_attn.proj_drop)

        # timm usually has self.scale = head_dim ** -0.5
        self.embed_dim = orig_attn.qkv.in_features
        self.head_dim = self.embed_dim // self.num_heads

        self.store_attention = store_attention
        self.last_attn = None

    def _reshape_qkv(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v

    def _final_proj(self, out):
        # out: [B, H, N, D]
        B, H, N, D = out.shape
        out = out.transpose(1, 2).reshape(B, N, H * D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def _maybe_store_attention(self, attn):
        if self.store_attention:
            self.last_attn = attn
            if attn.requires_grad:
                attn.retain_grad()


class ScoreableSoftmaxAttention(_BaseWrappedAttention):
    """
    For importance scoring only.
    Reproduces original softmax attention, but stores A and grad(A).
    """
    def __init__(self, orig_attn, store_attention=True):
        super().__init__(orig_attn, store_attention=store_attention)
        self.scale = getattr(orig_attn, "scale", self.head_dim ** -0.5)

    def forward(self, x):
        q, k, v = self._reshape_qkv(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        self._maybe_store_attention(attn)

        out = attn @ v
        return self._final_proj(out)


class SteLLAAttention(_BaseWrappedAttention):
    """
    MixA non-critical-layer attention:
        A_L = ReLU(Q) ReLU(K)^T / beta_L
        O   = (A_L V) / alpha_L
    where beta_L = beta_l_prime * sqrt(d)
          alpha_L = alpha_l_prime * N

    For training/inference we use the reordered form:
        O = ReLU(Q) (ReLU(K)^T V) / (alpha_L * beta_L)

    For scoring only, set store_attention=True to materialize A_L.
    """
    def __init__(self, orig_attn, store_attention=False):
        super().__init__(orig_attn, store_attention=store_attention)
        self.alpha_l_prime = nn.Parameter(torch.tensor(1.0))
        self.beta_l_prime = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        q, k, v = self._reshape_qkv(x)
        B, H, N, D = q.shape

        phi_q = torch.relu(q)
        phi_k = torch.relu(k)

        beta_l = self.beta_l_prime * math.sqrt(D)
        alpha_l = self.alpha_l_prime * N

        if self.store_attention:
            # scoring mode: explicitly materialize A_L so we can compare A' - A
            attn = (phi_q @ phi_k.transpose(-2, -1)) / beta_l
            attn = self.attn_drop(attn)
            self._maybe_store_attention(attn)
            out = (attn @ v) / alpha_l
        else:
            # efficient mode: Q (K^T V)
            kv = phi_k.transpose(-2, -1) @ v          # [B, H, D, D]
            out = (phi_q @ kv) / (alpha_l * beta_l)  # [B, H, N, D]

        return self._final_proj(out)