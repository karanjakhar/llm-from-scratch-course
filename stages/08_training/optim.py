"""Stage 08: the optimizer and schedule, by hand. docs/08_training_loop.md §2-§6.

AdamW here is written out for reading; test_training.py checks it against
torch.optim.AdamW step for step. Real runs use torch's fused version
(llmfs/train/trainer.py), which is the same arithmetic in one kernel.
"""

from __future__ import annotations

import math

import torch


class SGD:
    """θ ← θ − lr·g. docs §2a. Kept for the comparison in run.py adamw."""

    def __init__(self, params, lr: float, momentum: float = 0.0):
        self.params = [p for p in params]
        self.lr, self.momentum = lr, momentum
        self.m = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        # EXERCISE (module 8): write this function.
        # For each parameter with a gradient: with momentum, m = momentum*m + g and step
        # along m; without, step along g. Step size lr, downhill. In place.
        raise NotImplementedError("SGD.step: your code here")


class AdamW:
    """Adam with decoupled weight decay (docs §2c-§2d).

    param_groups: list of {"params": [...], "weight_decay": float}.
    l2_in_loss=True gives the *old* coupled version: decay is added to the
    gradient and then divided by √v like everything else. Only for §8a.
    """

    def __init__(self, param_groups, lr: float = 1e-3, betas=(0.9, 0.95), eps: float = 1e-8,
                 l2_in_loss: bool = False):
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": list(param_groups), "weight_decay": 0.0}]
        self.param_groups = [dict(g, lr=lr) for g in param_groups]
        self.beta1, self.beta2 = betas
        self.eps, self.l2 = eps, l2_in_loss
        self.t = 0
        # The two state tensors per weight, same shape as the weight (§2b-c).
        self.state = {id(p): (torch.zeros_like(p), torch.zeros_like(p))
                      for g in self.param_groups for p in g["params"]}

    @torch.no_grad()
    def step(self):
        # EXERCISE (module 8): write this function.
        # t += 1. Per parameter: decoupled decay shrinks p by (1 - lr*wd) (or, if
        # self.l2, wd*p is added to the gradient instead); m and v are running averages
        # of g and g^2; bias-correct both with 1 - beta^t; p -= lr * m_hat /
        # (sqrt(v_hat) + eps). docs §2.
        raise NotImplementedError("AdamW.step: your code here")

    def zero_grad(self):
        for g in self.param_groups:
            for p in g["params"]:
                p.grad = None


def lr_schedule(step: int, total: int, peak: float, warmup: int, min_ratio: float = 0.1) -> float:
    """Linear warmup to peak, then cosine to min_ratio·peak (docs §4)."""
    # EXERCISE (module 8): write this function.
    # Linear warmup from peak/warmup to peak over `warmup` steps, then cosine from peak
    # down to min_ratio*peak at step `total`.
    raise NotImplementedError("lr_schedule: your code here")


def clip_grad_norm(params, max_norm: float) -> float:
    """Global-norm clipping (docs §5). Returns the norm BEFORE clipping."""
    # EXERCISE (module 8): write this function.
    # The norm of ALL gradients together (one vector). If it exceeds max_norm, scale
    # every gradient by max_norm / (norm + 1e-6). Return the norm before clipping.
    raise NotImplementedError("clip_grad_norm: your code here")
