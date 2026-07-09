"""
STDP-style Eligibility Traces — v6.
====================================

A decaying memory of recent (pre, post) co-activations, used to credit-
assign delayed rewards.  Supports 1-D (vector) and 2-D (matrix) traces.

When `E.ndim == 2`:  E[i, j] tracks how recently post-neuron i was active
   shortly after pre-neuron j (causal STDP+) minus how often they co-fire
   in the anti-causal direction (STDP-).

When `E.ndim == 1`:  collapsed scalar trace (used for value-vector
   credit assignment in the basal ganglia).
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


class EligibilityTrace:
    def __init__(self, shape: Tuple[int, ...], gamma: float = 0.95,
                 lam: float = 0.85, stdp_pos: float = 0.05,
                 stdp_neg: float = 0.025):
        self.shape = tuple(int(s) for s in shape)
        self.E = np.zeros(self.shape, dtype=np.float32)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.stdp_pos = float(stdp_pos)
        self.stdp_neg = float(stdp_neg)
        pre_dim = self.shape[1] if len(self.shape) >= 2 else self.shape[0]
        self._prev = np.zeros(pre_dim, dtype=np.float32)

    def step(self, post: np.ndarray, pre: np.ndarray) -> None:
        self.E *= self.gamma * self.lam
        if self.E.ndim == 2:
            causal = self.stdp_pos * np.outer(post, self._prev)
            anti = self.stdp_neg * np.outer(post, pre)
            self.E += (causal - 0.5 * anti).astype(np.float32)
        else:
            self.E += (self.stdp_pos * self._prev
                       - self.stdp_neg * pre).astype(np.float32)
            self.E += (0.5 * post).astype(np.float32)
        np.clip(self.E, -1.0, 1.0, out=self.E)
        self._prev = pre.astype(np.float32).copy()

    def reset(self) -> None:
        self.E.fill(0.0)
        self._prev.fill(0.0)

    def magnitude(self) -> float:
        return float(np.linalg.norm(self.E))
