"""Stage 06: the complete GPT. docs/06_gpt_model.md is the lesson.

Everything from Stages 03-05 in one model, with GPT-2's module names so the
real GPT-2 weights load into it (docs §5). This is the explicit, commented
version. llmfs/model/gpt.py is the same model in the form later stages import.

Shape letters: B batch, T tokens, C width (n_embd), H heads, d = C/H,
L layers, V vocabulary.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "04_attention"))
sys.path.append(str(HERE.parent / "05_block"))

from attention import attention  # noqa: E402   Stage 04, hand-written softmax(qkᵀ/√d)v
from block import LayerNorm, gelu_exact, gelu_tanh  # noqa: E402   Stage 05


@dataclass
class GPTConfig:
    vocab_size: int = 8196        # our tokenizer: 8,192 merges + 4 specials
    block_size: int = 256         # T_max: rows in the position table
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    act: str = "gelu_tanh"        # GPT-2 uses the tanh approximation (Stage 05 §6)
    tie: bool = True              # lm_head shares wte's table (Stage 03 §2)
    scaled_init: bool = True      # c_proj std 0.02/√(2L) (docs §4b)
    flash: bool = False           # F.scaled_dot_product_attention instead of Stage 04's
    bias: bool = True             # GPT-2 has biases on every linear and norm


# GPT-2's four sizes, for counting and loading. vocab 50,257, T_max 1,024.
GPT2_SIZES = {
    "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),     # 124M
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),    # 350M
    "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),    # 774M
    "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),    # 1558M
}


class CausalSelfAttention(nn.Module):
    """Stage 04's MultiHeadAttention under GPT-2's names: c_attn makes q, k, v
    in one (C, 3C) matmul, c_proj mixes the heads and writes to the stream."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.flash = cfg.n_head, cfg.flash
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)   # [W_q | W_k | W_v]
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)       # writes to the residual stream
        self.c_proj.writes_residual = True                       # marker for the scaled init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EXERCISE (module 6): write this function.
        # Module 4's multi-head attention under GPT-2's names (c_attn, c_proj). If
        # self.flash, use F.scaled_dot_product_attention(q, k, v, is_causal=True), else
        # your attention().
        raise NotImplementedError("CausalSelfAttention.forward: your code here")


class MLP(nn.Module):
    """C -> 4C -> C, Stage 05 §5, under GPT-2's names."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.writes_residual = True
        self.act = {"gelu_tanh": gelu_tanh, "gelu": gelu_exact}[cfg.act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))               # (B, T, C) -> (B, T, 4C) -> (B, T, C)


class Block(nn.Module):
    """Pre-norm block (Stage 05 §4): x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, T, C)
        # EXERCISE (module 6): write this function.
        # Pre-norm: x + attn(ln_1(x)), then x + mlp(ln_2(x)).
        raise NotImplementedError("Block.forward: your code here")


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        # nn.ModuleDict named "transformer" so parameter names are exactly
        # GPT-2's ("transformer.h.3.attn.c_attn.weight"), docs §5.
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),       # (V, C)
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),       # (T_max, C)
            h=nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer)),
            ln_f=LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)    # (V, C) stored
        if cfg.tie:
            # One Parameter, two uses. Its gradient is the sum of both.
            self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init)
        self._scale_residual_init()

    # -- initialisation, docs §4 -------------------------------------------

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
        # LayerNorm keeps its own init: gain 1, shift 0.

    def _scale_residual_init(self):
        # EXERCISE (module 6): write this function.
        # Only if cfg.scaled_init: every module marked writes_residual gets normal init
        # with std 0.02 / sqrt(2 * n_layer). docs §4b.
        raise NotImplementedError("GPT._scale_residual_init: your code here")

    # -- forward -----------------------------------------------------------

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                keep_stream: bool = False):
        # EXERCISE (module 6): write this function.
        # Token embedding + position embedding, every block, ln_f, lm_head. With
        # targets, the cross-entropy over all positions in float32. keep_stream also
        # returns the list of x after the embedding and after each block.
        raise NotImplementedError("GPT.forward: your code here")

    # -- loading GPT-2, docs §5 ---------------------------------------------

    @classmethod
    def from_pretrained(cls, name: str = "gpt2", **overrides) -> "GPT":
        """Build GPT-2 `name` and copy OpenAI's weights in, reading the
        safetensors file directly (no transformers model is constructed)."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        cfg = GPTConfig(vocab_size=50257, block_size=1024, act="gelu_tanh", tie=True,
                        **GPT2_SIZES[name], **overrides)
        model = cls(cfg)
        path = hf_hub_download(f"openai-community/{name}", "model.safetensors")
        theirs = load_file(path)
        load_gpt2_state(model, theirs)
        return model


# Conv1D layers store (in, out); nn.Linear stores (out, in). docs §5 trap 1.
TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def load_gpt2_state(model: GPT, theirs: dict[str, torch.Tensor]) -> None:
    """Copy a GPT-2 state dict into `model`. Fails on anything unmatched."""
    # EXERCISE (module 6): write this function.
    # docs §5's three traps: skip the saved attention-mask buffers, skip lm_head (it is
    # tied), transpose the TRANSPOSED weights (Conv1D stores (in, out)). Add
    # 'transformer.' to names that lack it. Raise on any tensor with no home, any shape
    # mismatch, and any of our tensors left unfilled.
    raise NotImplementedError("load_gpt2_state: your code here")


# ---------------------------------------------------------------------------
# Counting by hand, docs §3
# ---------------------------------------------------------------------------

def count_params_by_hand(cfg: GPTConfig) -> dict[str, int]:
    # EXERCISE (module 6): write this function.
    # Per block: two LayerNorms (2C each), c_attn C*3C, attn c_proj C*C, c_fc C*4C, mlp
    # c_proj 4C*C, plus biases if cfg.bias. Then wte V*C, wpe T*C, ln_f 2C, and lm_head
    # only if untied. Return the parts, 'total' and 'non_embedding'.
    raise NotImplementedError("count_params_by_hand: your code here")


def count_params(model: nn.Module) -> int:
    # .parameters() yields a shared (tied) tensor once.
    return sum(p.numel() for p in model.parameters())
