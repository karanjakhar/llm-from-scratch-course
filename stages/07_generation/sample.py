"""Stage 07: turning logits into text. docs/07_generation.md is the lesson.

Every knob is a separate function that takes logits (B, V) and returns
logits (B, V), so they compose in a fixed order and each can be tested alone:

    repetition penalty -> temperature -> top-k -> top-p -> softmax -> draw

The model is llmfs.model.gpt.GPT (Stage 06), whose forward takes an optional
KVCache (docs §6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmfs.model.gpt import GPT, KVCache  # noqa: E402

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# The knobs (docs §3-§5). All take and return logits of shape (B, V).
# ---------------------------------------------------------------------------

def apply_temperature(logits: torch.Tensor, tau: float) -> torch.Tensor:
    # Softmax only sees differences between logits; dividing by τ scales
    # every difference at once (docs §3). τ = 0 is handled as greedy upstream.
    # EXERCISE (module 7): write this function.
    # Divide the logits by tau.
    raise NotImplementedError("apply_temperature: your code here")


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    # EXERCISE (module 7): write this function.
    # Nothing to do if k <= 0 or k >= V. Otherwise every logit below the k-th largest
    # becomes NEG_INF (ties with the k-th survive).
    raise NotImplementedError("top_k_filter: your code here")


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest set of top tokens whose probability sum reaches p."""
    # EXERCISE (module 7): write this function.
    # Sort descending, softmax, cumulative sum. Drop a token when the tokens ABOVE it
    # already reach p, so the token that crosses p survives. Scatter the mask back to
    # vocabulary order before applying it. docs §4.
    raise NotImplementedError("top_p_filter: your code here")


def apply_repetition_penalty(logits: torch.Tensor, context: torch.Tensor,
                             penalty: float) -> torch.Tensor:
    """CTRL's penalty (docs §5). context: (B, T) ids already in the sequence."""
    # EXERCISE (module 7): write this function.
    # Gather the logits of tokens already in context; divide the positive ones by the
    # penalty and multiply the negative ones, so both go down; scatter back.
    raise NotImplementedError("apply_repetition_penalty: your code here")


def sample_next(logits: torch.Tensor, context: torch.Tensor | None = None,
                temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                repetition_penalty: float = 1.0,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """logits (B, V) for the next position -> ids (B, 1)."""
    # EXERCISE (module 7): write this function.
    # float() the logits; repetition penalty if context is given; temperature 0 means
    # argmax. Otherwise temperature, top-k, top-p, softmax, torch.multinomial with the
    # generator. Returns (B, 1).
    raise NotImplementedError("sample_next: your code here")


# ---------------------------------------------------------------------------
# The loop, two ways (docs §1, §6)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_naive(model: GPT, ids: torch.Tensor, max_new_tokens: int, eot: int | None = None,
                   seed: int | None = None, **knobs) -> torch.Tensor:
    """Re-run the whole sequence every step. O(N²) positions processed."""
    g = _gen(ids.device, seed)
    for _ in range(max_new_tokens):
        logits, _ = model(ids, last_only=True)                   # (B, 1, V)
        nxt = sample_next(logits[:, -1], ids, generator=g, **knobs)
        ids = torch.cat([ids, nxt], dim=1)                       # (B, T+1)
        if eot is not None and (nxt == eot).all():
            break
    return ids


@torch.no_grad()
def generate_cached(model: GPT, ids: torch.Tensor, max_new_tokens: int, eot: int | None = None,
                    seed: int | None = None, cache_dtype=None, **knobs) -> torch.Tensor:
    """Prefill the prompt into a KV cache, then feed one token per step."""
    # EXERCISE (module 7): write this function.
    # Make a cache with model.make_cache(B, T + max_new_tokens, device, dtype); run the
    # whole prompt once with it (last_only=True); then feed ONE new token per step. Stop
    # on eot or after max_new_tokens tokens. Compare generate_naive above.
    raise NotImplementedError("generate_cached: your code here")


def _gen(device, seed):
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g
