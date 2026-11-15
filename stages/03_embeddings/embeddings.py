"""Stage 03: token embeddings and three ways to give a model position.

docs/03_embeddings_position.md is the lesson; this file is the code it talks
about. Everything here is deliberately explicit. The compact versions that
later stages reuse get promoted into llmfs/ when a stage first needs them.

Shape letters, used on every tensor line:
    B batch   T sequence length   C model width   V vocab size
    H heads   d head size (C // H)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Token embedding: a lookup table (docs §1)
# ---------------------------------------------------------------------------

class TokenEmbedding(nn.Module):
    """One learned row of C numbers per token id.

    Written by hand instead of nn.Embedding so the mechanism is visible: the
    forward pass is a single indexing operation. The test checks it against
    nn.Embedding and against the one-hot matmul below.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        # std 0.02 is GPT-2's init. Small, so early logits are near-uniform and
        # the first loss is close to ln(V) instead of something huge.
        self.weight = nn.Parameter(torch.randn(vocab_size, dim) * 0.02)   # (V, C)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # EXERCISE (module 3): write this function.
        # One indexing operation: each id selects its row of self.weight. (B, T) -> (B,
        # T, C).
        raise NotImplementedError("TokenEmbedding.forward: your code here")


def embed_onehot(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The same lookup written as a matmul. Never use this for real: it builds
    a V-wide vector per token (8,196 floats to pick one row). It exists to show
    that an embedding is a bias-free linear layer on one-hot inputs, which is
    why only the rows of tokens present in a batch receive a gradient."""
    one_hot = F.one_hot(ids, num_classes=weight.shape[0]).to(weight.dtype)  # (B, T, V)
    return one_hot @ weight                                      # (B, T, V) @ (V, C) -> (B, T, C)


# ---------------------------------------------------------------------------
# 2. Learned absolute positions, GPT-2 style (docs §4)
# ---------------------------------------------------------------------------

class LearnedPosition(nn.Module):
    """A second lookup table, indexed by slot number instead of token id."""

    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.max_len = max_len
        self.weight = nn.Parameter(torch.randn(max_len, dim) * 0.02)  # (T_max, C)

    def forward(self, T: int) -> torch.Tensor:
        # The hard wall from docs §4: there is no row for position max_len.
        # Fail loudly rather than let an index error surface somewhere deep.
        # EXERCISE (module 3): write this function.
        # The first T rows of the table. Asking for more than max_len rows is a
        # ValueError: there is no row to return (docs §4, the hard wall).
        raise NotImplementedError("LearnedPosition.forward: your code here")


# ---------------------------------------------------------------------------
# 3. Sinusoidal positions, 2017 transformer (docs §5)
# ---------------------------------------------------------------------------

def frequencies(dim: int, base: float = 10000.0) -> torch.Tensor:
    """Angular speed of each 'clock': omega_i = base^(-2i/dim), i = 0..dim/2-1.

    Shared by sinusoidal positions (dim = C) and RoPE (dim = d, the head size).
    omega_0 = 1 radian per token; the last one is about 1/base.
    Computed in float64 then cast, so large positions do not lose precision
    in the multiply that follows.
    """
    # EXERCISE (module 3): write this function.
    # omega_i = base^(-2i/dim) for i = 0 .. dim/2 - 1, computed in float64. dim must be
    # even.
    raise NotImplementedError("frequencies: your code here")


def sinusoidal_table(T: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    """PE[p, 2i] = sin(p * omega_i), PE[p, 2i+1] = cos(p * omega_i)."""
    # EXERCISE (module 3): write this function.
    # angles = position * omega, shape (T, dim/2). Even columns get sin, odd columns
    # cos. Compute in float64, return float32.
    raise NotImplementedError("sinusoidal_table: your code here")


class SinusoidalPosition(nn.Module):
    """No parameters. Computed for whatever length is asked, so unlike the
    learned table there is no wall -- but see docs §5 for why 'can compute'
    is not the same as 'has learned what to do with'."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim, self.base = dim, base

    def forward(self, T: int, device=None) -> torch.Tensor:
        # A parameter-free module has no tensor to ask for its device, so the
        # caller says where the table should live.
        return sinusoidal_table(T, self.dim, self.base).to(device)   # (T, C)


# ---------------------------------------------------------------------------
# 4. RoPE: rotate q and k by position (docs §6)
# ---------------------------------------------------------------------------

def rope_angles(T: int, d: int, base: float = 10000.0, offset: int = 0,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """angle[m, i] = m * theta_i for positions m = offset..offset+T-1.

    Precision note: the fastest pair's angle equals the position in radians.
    float32 has ~7 significant digits, so at position 127 the angle is off by
    up to ~4e-6 rad, and at position 100,000 by ~4e-3. Long-context models
    care about this; at our 256-1024 token context it is noise. The test that
    checks "score depends only on offset" asks for float64 to see the exact math.
    """
    # EXERCISE (module 3): write this function.
    # Like the sinusoidal angles, for positions offset .. offset+T-1, using the head
    # size d. float64 for the multiply, then cast to `dtype`.
    raise NotImplementedError("rope_angles: your code here")


def rope_interleaved(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (0,1), (2,3), ... -- the paper's layout and Meta's.

    x: (..., T, d) queries or keys. angles: (T, d/2).
    Each pair (a, b) becomes (a cos - b sin, a sin + b cos): a 2-D rotation.
    """
    # EXERCISE (module 3): write this function.
    # Pairs are dims (0,1), (2,3), ... For each pair (a, b): (a cos - b sin, a sin + b
    # cos). Cast cos and sin to x's dtype.
    raise NotImplementedError("rope_interleaved: your code here")


def rope_half(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (0, d/2), (1, d/2+1), ... -- HuggingFace's `rotate_half`.

    Same rotation, different choice of which two dimensions form a pair.
    Weights trained with one layout are silently wrong under the other.
    """
    # EXERCISE (module 3): write this function.
    # The same rotation with pairs (i, i + d/2): first half is a, second half is b.
    raise NotImplementedError("rope_half: your code here")


def rope_complex(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """The complex-number view (docs §6): pair (a, b) is the number a + ib, and
    rotating by angle t is multiplying by e^{it}. This is how Meta's reference
    Llama code writes it. Must equal rope_interleaved exactly."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))   # (..., T, d/2) complex
    rot = torch.polar(torch.ones_like(angles), angles)           # (T, d/2) complex, |rot| = 1
    return torch.view_as_real(xc * rot).flatten(-2).to(x.dtype)  # (..., T, d)


def interleaved_to_half(x: torch.Tensor) -> torch.Tensor:
    """The dimension shuffle that turns one layout into the other: even dims
    first, then odd dims. rope_half(interleaved_to_half(x)) equals
    interleaved_to_half(rope_interleaved(x)) -- the test checks it."""
    return torch.cat([x[..., 0::2], x[..., 1::2]], dim=-1)


# ---------------------------------------------------------------------------
# 5. Output head and weight tying (docs §2)
# ---------------------------------------------------------------------------

class OutputHead(nn.Module):
    """hidden (B, T, C) -> logits (B, T, V).

    tie_to: pass the TokenEmbedding to reuse its table. The head then holds a
    reference to the *same* Parameter, so there is one tensor, one gradient
    (the sum of both uses), and V*C fewer parameters.
    """

    def __init__(self, dim: int, vocab_size: int, tie_to: TokenEmbedding | None = None):
        super().__init__()
        if tie_to is not None:
            self.weight = tie_to.weight                          # same storage, not a copy
        else:
            self.weight = nn.Parameter(torch.randn(vocab_size, dim) * 0.02)  # (V, C)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Score for token i = dot(hidden, row i). Stored as (V, C), as in
        # nn.Linear, so the matmul uses the transpose.
        return h @ self.weight.T                                 # (B, T, C) @ (C, V) -> (B, T, V)


def count_params(module: nn.Module) -> int:
    # .parameters() de-duplicates shared tensors, so a tied table counts once.
    return sum(p.numel() for p in module.parameters())


def attention_no_mask(x: torch.Tensor) -> torch.Tensor:
    """The smallest possible self-attention, with no mask and no learned
    projections, used only to show permutation equivariance (docs §3).
    Stage 04 builds real attention."""
    scores = x @ x.transpose(-2, -1) / math.sqrt(x.shape[-1])    # (B, T, T)
    return scores.softmax(dim=-1) @ x                            # (B, T, C)
