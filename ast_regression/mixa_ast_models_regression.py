# mixa_ast_models_regression.py
import math
import copy
import torch
import torch.nn as nn

from ast_models_regression import ASTRegressor


class _BaseWrappedAttention(nn.Module):
    """
    Wrap timm DeiT attention while keeping qkv/proj parameter names compatible
    with the AST baseline checkpoint.
    """
    def __init__(self, orig_attn):
        super().__init__()
        self.num_heads = orig_attn.num_heads

        # keep the same submodule names so baseline checkpoint can load with strict=False
        self.qkv = copy.deepcopy(orig_attn.qkv)
        self.attn_drop = copy.deepcopy(orig_attn.attn_drop)
        self.proj = copy.deepcopy(orig_attn.proj)
        self.proj_drop = copy.deepcopy(orig_attn.proj_drop)

        self.embed_dim = orig_attn.qkv.in_features
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = getattr(orig_attn, "scale", self.head_dim ** -0.5)

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


class ReLUQuadraticAttention(_BaseWrappedAttention):
    """
    MixA targeted quadratic attention:
        A_Q = ReLU(QK^T / beta_Q)
        O   = (A_Q V) / alpha_Q

    beta_Q = beta_q_prime * sqrt(d)
    alpha_Q = alpha_q_prime * N
    """
    def __init__(self, orig_attn):
        super().__init__(orig_attn)
        self.alpha_q_raw = nn.Parameter(torch.tensor(0.5413))
        self.beta_q_raw = nn.Parameter(torch.tensor(0.5413))

    def forward(self, x):
        q, k, v = self._reshape_qkv(x)  # [B,H,N,D]
        B, H, N, D = q.shape

        q = q.float()
        k = k.float()
        v = v.float()

        alpha_q_prime = torch.nn.functional.softplus(self.alpha_q_raw) + 1e-4
        beta_q_prime = torch.nn.functional.softplus(self.beta_q_raw) + 1e-4

        beta_q = beta_q_prime * math.sqrt(D)
        alpha_q = alpha_q_prime * N

        attn = torch.relu((q @ k.transpose(-2, -1)) / beta_q)
        attn = self.attn_drop(attn)
        out = (attn @ v) / alpha_q

        return self._final_proj(out)


class SteLLAAttention(_BaseWrappedAttention):
    """
    MixA SteLLA linear attention:
        A_L = ReLU(Q) ReLU(K)^T / beta_L
        O   = (A_L V) / alpha_L

    Efficient reordered form:
        O = ReLU(Q) (ReLU(K)^T V) / (alpha_L * beta_L)

    beta_L = beta_l_prime * sqrt(d)
    alpha_L = alpha_l_prime * N
    """
    def __init__(self, orig_attn):
        super().__init__(orig_attn)
        self.alpha_l_raw = nn.Parameter(torch.tensor(0.5413))
        self.beta_l_raw = nn.Parameter(torch.tensor(0.5413))

    def forward(self, x):
        q, k, v = self._reshape_qkv(x)  # [B,H,N,D]
        B, H, N, D = q.shape

        q = q.float()
        k = k.float()
        v = v.float()

        alpha_l_prime = torch.nn.functional.softplus(self.alpha_l_raw) + 1e-4
        beta_l_prime = torch.nn.functional.softplus(self.beta_l_raw) + 1e-4

        beta_l = beta_l_prime * math.sqrt(D)
        alpha_l = alpha_l_prime * N

        phi_q = torch.relu(q)
        phi_k = torch.relu(k)

        # reordered linear attention: Q (K^T V)
        kv = phi_k.transpose(-2, -1) @ v           # [B,H,D,D]
        out = (phi_q @ kv) / (alpha_l * beta_l)   # [B,H,N,D]

        return self._final_proj(out)


class ASTMixARegressor(ASTRegressor):
    """
    AST regression backbone modified with MixA:
      - top-k important layers -> ReLUQuadraticAttention
      - remaining layers       -> SteLLAAttention
    """
    def __init__(
        self,
        topk_layers,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1024,
        imagenet_pretrain=True,
        model_size='base384',
        verbose=True,
    ):
        super().__init__(
            fstride=fstride,
            tstride=tstride,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain,
            model_size=model_size,
            verbose=verbose,
        )

        self.topk_layers = sorted([int(x) for x in topk_layers])
        if len(set(self.topk_layers)) != len(self.topk_layers) or any(
            i < 0 or i >= len(self.v.blocks) for i in self.topk_layers
        ):
            raise ValueError('topk_layers must contain unique, valid zero-based block indices.')
        self._apply_mixa()

        if verbose:
            print('---------------MixA AST Layer Layout---------------')
            print('Top-k quadratic layers (zero-based):', self.topk_layers)
            print('Top-k quadratic layers (one-based):', [x + 1 for x in self.topk_layers])
            for i, blk in enumerate(self.v.blocks):
                print(f'Layer {i+1:02d}: {type(blk.attn).__name__}')

    def _apply_mixa(self):
        topk_set = set(self.topk_layers)
        for i, blk in enumerate(self.v.blocks):
            orig_attn = blk.attn
            if i in topk_set:
                blk.attn = ReLUQuadraticAttention(orig_attn)
            else:
                blk.attn = SteLLAAttention(orig_attn)

    def get_topk_layers(self):
        return list(self.topk_layers)

class ASTSoftmaxSteLLARegressor(ASTRegressor):
    """
    Hybrid AST:
      - top-k important layers -> keep original softmax attention
      - remaining layers       -> replace with SteLLAAttention

    This is NOT the original MixA design.
    It is a task-specific hybrid ablation.
    """
    def __init__(
        self,
        topk_layers,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1024,
        imagenet_pretrain=True,
        model_size='base384',
        verbose=True,
    ):
        super().__init__(
            fstride=fstride,
            tstride=tstride,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain,
            model_size=model_size,
            verbose=verbose,
        )

        self.topk_layers = sorted([int(x) for x in topk_layers])
        if len(set(self.topk_layers)) != len(self.topk_layers) or any(
            i < 0 or i >= len(self.v.blocks) for i in self.topk_layers
        ):
            raise ValueError('topk_layers must contain unique, valid zero-based block indices.')
        self._apply_softmax_stella_hybrid()

        if verbose:
            print('---------------Softmax-SteLLA Hybrid AST Layer Layout---------------')
            print('Softmax layers kept (zero-based):', self.topk_layers)
            print('Softmax layers kept (one-based):', [x + 1 for x in self.topk_layers])
            for i, blk in enumerate(self.v.blocks):
                print(f'Layer {i+1:02d}: {type(blk.attn).__name__}')

    def _apply_softmax_stella_hybrid(self):
        topk_set = set(self.topk_layers)
        for i, blk in enumerate(self.v.blocks):
            if i in topk_set:
                # keep original attention unchanged
                continue
            else:
                blk.attn = SteLLAAttention(blk.attn)

    def get_topk_layers(self):
        return list(self.topk_layers)