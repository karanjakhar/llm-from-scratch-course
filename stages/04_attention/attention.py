"""Stage 04: attention, from one head up.

docs/04_attention.md is the lesson. Everything here is written out by hand
and checked against F.scaled_dot_product_attention and nn.MultiheadAttention
in test_attention.py.

Shape letters: B batch, T tokens, C model width, H heads, d = C // H head size.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def causal_mask(T: int, device=None) -> torch.Tensor:
    """True where attention is ALLOWED: key j at or before query i."""
    # EXERCISE (module 4): write this function.
    # A (T, T) boolean matrix, True where key j <= query i: the lower triangle.
    raise NotImplementedError("causal_mask: your code here")


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              causal: bool = True, scale: bool = True,
              return_weights: bool = False):
    """softmax(q kᵀ / √d) v, written out. docs §2-§4.

    q, k, v: (..., T, d). Any leading dims (batch, heads) are carried along.
    scale=False drops the √d, for the ablation in docs §9 only.
    """
    # EXERCISE (module 4): write this function.
    # scores = q @ k^T, divided by sqrt(d) if scale. If causal, set disallowed scores to
    # -inf (not 0). Softmax over the last dim, then weights @ v. docs §2-4.
    raise NotImplementedError("attention: your code here")


class SingleHeadAttention(nn.Module):
    """One head: three projections C -> d, then attention. No output projection,
    because with one head there is nothing to mix (docs §5)."""

    def __init__(self, dim: int, head_dim: int, causal: bool = True):
        super().__init__()
        self.causal = causal
        self.q = nn.Linear(dim, head_dim, bias=False)            # W_q (C, d)
        self.k = nn.Linear(dim, head_dim, bias=False)            # W_k
        self.v = nn.Linear(dim, head_dim, bias=False)            # W_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)                # (B, T, d) each
        return attention(q, k, v, causal=self.causal)            # (B, T, d)


class MultiHeadAttention(nn.Module):
    """H heads of size d = C / H, side by side, then an output projection.

    One (C, 3C) matmul makes q, k, v for every head at once; the heads are
    then split out with view + transpose. That reshape is where attention bugs
    hide: a wrong one keeps the shape and scrambles the contents.
    """

    def __init__(self, dim: int, n_head: int, causal: bool = True,
                 scale: bool = True, bias: bool = True):
        super().__init__()
        assert dim % n_head == 0, "heads must split the width evenly"
        self.n_head, self.causal, self.scale = n_head, causal, scale
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)            # [W_q | W_k | W_v], each (C, C)
        self.proj = nn.Linear(dim, dim, bias=bias)               # W_o (C, C): mixes the heads
        self.last_weights: torch.Tensor | None = None            # filled when keep_weights is on
        self.keep_weights = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EXERCISE (module 4): write this function.
        # qkv -> split into q, k, v (B, T, C) -> view as (B, T, H, d) and transpose to
        # (B, H, T, d). Attend (keep the weights in self.last_weights if
        # self.keep_weights), transpose back, reshape to (B, T, C), then self.proj. The
        # reshape is where bugs hide.
        raise NotImplementedError("MultiHeadAttention.forward: your code here")
