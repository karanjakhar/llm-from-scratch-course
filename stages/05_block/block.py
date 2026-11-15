"""Stage 05: the transformer block, taken apart.

docs/05_transformer_block.md is the lesson. Every piece here is written by
hand and checked in test_block.py against nn.LayerNorm, nn.RMSNorm, F.gelu,
Hugging Face's GPT2Block and LlamaMLP.

Shape letters: B batch, T tokens, C model width, H heads.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
# Stage 04's attention, unchanged. Appended, not inserted first, so this
# stage's own files win any name clash.
sys.path.append(str(HERE.parent / "04_attention"))

from attention import MultiHeadAttention  # noqa: E402


# ---------------------------------------------------------------------------
# Normalisation (docs §3)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """γ · (x − mean) / sqrt(var + ε) + β, over the last dim only.

    Each token is normalised on its own, across its C features. Never across
    tokens or the batch, so it behaves the same for any batch size.
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))                  # γ (C,): starts as "no rescale"
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None  # β (C,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EXERCISE (module 5): write this function.
        # Per token, over the last dim: subtract the mean, divide by sqrt(variance +
        # eps), scale by weight, add bias if there is one. The BIASED variance (divide
        # by C).
        raise NotImplementedError("LayerNorm.forward: your code here")


class RMSNorm(nn.Module):
    """γ · x / sqrt(mean(x²) + ε). No mean subtraction, no β (docs §3).

    ε = 1e-6, the value Llama uses. Tests pass the same ε to the reference.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))                  # γ (C,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EXERCISE (module 5): write this function.
        # x divided by sqrt(mean(x^2) + eps), times weight. No mean subtraction, no
        # bias.
        raise NotImplementedError("RMSNorm.forward: your code here")


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "layer":
        return LayerNorm(dim)
    if kind == "rms":
        return RMSNorm(dim)
    raise ValueError(f"unknown norm {kind!r}")


# ---------------------------------------------------------------------------
# Activations and MLPs (docs §5-§6)
# ---------------------------------------------------------------------------

def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """GPT-2's GELU: the tanh approximation of x·Φ(x) ("gelu_new" in HF).

    Differs from the exact GELU by < 5e-4. Matching GPT-2 weights to 1e-5
    needs this exact formula, not the erf one.
    """
    # EXERCISE (module 5): write this function.
    # 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). GPT-2 uses exactly this.
    raise NotImplementedError("gelu_tanh: your code here")


def gelu_exact(x: torch.Tensor) -> torch.Tensor:
    """x · Φ(x), with Φ the standard normal CDF written via erf."""
    # EXERCISE (module 5): write this function.
    # x times the standard normal CDF, written with erf: 0.5 x (1 + erf(x / sqrt 2)).
    raise NotImplementedError("gelu_exact: your code here")


def relu(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_min(0.0)


def silu(x: torch.Tensor) -> torch.Tensor:
    """x · sigmoid(x), also called swish. SwiGLU's gate."""
    # EXERCISE (module 5): write this function.
    # x times sigmoid(x).
    raise NotImplementedError("silu: your code here")


ACTIVATIONS = {"gelu": gelu_exact, "gelu_tanh": gelu_tanh, "relu": relu}


class MLP(nn.Module):
    """The classic FFN: C → 4C, bend, 4C → C. Each token on its own."""

    def __init__(self, dim: int, act: str = "gelu", hidden: int | None = None,
                 bias: bool = True):
        super().__init__()
        hidden = hidden or 4 * dim
        self.act = ACTIVATIONS[act]
        self.fc = nn.Linear(dim, hidden, bias=bias)                  # W₁ (C → 4C)
        self.proj = nn.Linear(hidden, dim, bias=bias)                # W₂ (4C → C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EXERCISE (module 5): write this function.
        # Up to the hidden width, the activation, back down. (B, T, C) -> (B, T, 4C) ->
        # (B, T, C).
        raise NotImplementedError("MLP.forward: your code here")


def swiglu_hidden(dim: int, multiple_of: int = 1) -> int:
    """8C/3, so three matrices cost what the GELU MLP's two cost (docs §6).
    Real models round up to a multiple of 64 or 256 for fast matmuls; we keep
    multiple_of=1 by default so the parameter match is as close as possible."""
    h = int(8 * dim / 3)
    return multiple_of * math.ceil(h / multiple_of)


class SwiGLU(nn.Module):
    """W₂ · (silu(W_gate·x) ⊙ W_up·x). Llama's MLP, no biases."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or swiglu_hidden(dim)
        self.gate = nn.Linear(dim, hidden, bias=False)               # W_gate (C → h)
        self.up = nn.Linear(dim, hidden, bias=False)                 # W_up   (C → h)
        self.proj = nn.Linear(hidden, dim, bias=False)               # W₂     (h → C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EXERCISE (module 5): write this function.
        # proj( silu(gate(x)) * up(x) ): the gate decides how much of `up` passes
        # through.
        raise NotImplementedError("SwiGLU.forward: your code here")


# ---------------------------------------------------------------------------
# The block (docs §1-§4, §7)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """One transformer block with every design choice exposed as a switch.

    norm:     "layer" | "rms"
    act:      "gelu" | "gelu_tanh" | "relu" | "swiglu" | "none" (no MLP at all)
    prenorm:  True  -> x = x + f(norm(x))      (GPT-2, Llama)
              False -> x = norm(x + f(x))      (2017 paper, BERT)
    residual: False removes both "x +" adds, for the ablation in docs §10c.
    dropout:  on each sublayer's output, before the add (docs §7).
    """

    def __init__(self, dim: int, n_head: int, norm: str = "layer", act: str = "gelu",
                 prenorm: bool = True, residual: bool = True, dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        self.prenorm, self.residual = prenorm, residual
        # Registration order matters for reproducibility: it fixes the order
        # in which random init consumes the RNG. It matches Stage 04's Block
        # (ln1, attn, ln2, mlp) so a seed gives the same starting weights.
        self.ln1 = make_norm(norm, dim)
        self.attn = MultiHeadAttention(dim, n_head, causal=True, bias=bias)
        self.ln2 = make_norm(norm, dim)
        if act == "swiglu":
            self.mlp = SwiGLU(dim)
        elif act == "none":
            self.mlp = None
        else:
            self.mlp = MLP(dim, act=act, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.probe = False                                       # record |f(x)| / |x| (run.py diagnose)
        self.ratios: list[float] = []

    def _sublayer(self, x: torch.Tensor, f, norm: nn.Module) -> torch.Tensor:
        if self.prenorm:
            out = self.drop(f(norm(x)))
            if self.probe:
                self.ratios.append((out.float().pow(2).mean() / x.float().pow(2).mean()).sqrt().item())                              # (B, T, C) the side road
            return x + out if self.residual else out                 # highway untouched
        out = self.drop(f(x))
        if self.probe:                                           # how big is the edit next to the highway?
            self.ratios.append((out.float().pow(2).mean() / x.float().pow(2).mean()).sqrt().item())
        return norm(x + out if self.residual else out)               # norm sits ON the highway

    def forward(self, x: torch.Tensor) -> torch.Tensor:             # x: (B, T, C)
        x = self._sublayer(x, self.attn, self.ln1)                   # tokens talk
        if self.mlp is not None:
            x = self._sublayer(x, self.mlp, self.ln2)                # each token thinks alone
        return x                                                     # (B, T, C)
