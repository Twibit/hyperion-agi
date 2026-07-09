"""
Continual learning: Synaptic Tagging & Capture + Benna-Fusi cascade.
=====================================================================

`SynapticConsolidator` — Elastic-Weight-Consolidation-style importance
mask that grows under reward-modulated bursts (synaptic tagging) and
locks the most important weights against future drift (capture).

`BennaFusiCascade` — multi-timescale cascade of variables (fast, medium,
slow) approximating the Benna-Fusi (2016) optimal memory model.  Each
synapse stores N coupled scalars at different timescales; the visible
weight is a weighted combination, so transient learning lives in the
fast variable while long-term knowledge migrates to slow variables.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


class SynapticConsolidator:
    """Online EWC-lite: keep an importance mask over a weight tensor."""

    def __init__(self, shape: Tuple[int, ...], decay: float = 0.999):
        self.shape = tuple(int(s) for s in shape)
        self.importance = np.zeros(self.shape, dtype=np.float32)
        self.anchor = np.zeros(self.shape, dtype=np.float32)
        self.decay = float(decay)

    def tag(self, grad_like: np.ndarray, reward_burst: float = 1.0) -> None:
        """Increase importance on synapses with large reward-correlated grads."""
        self.importance = (self.decay * self.importance
                           + (1.0 - self.decay) * np.abs(grad_like) * float(reward_burst))

    def capture(self, weights: np.ndarray) -> None:
        """Move the anchor toward current weights, weighted by importance."""
        w = self.importance / (self.importance.max() + 1e-9)
        self.anchor = (1.0 - w) * self.anchor + w * weights

    def regularise(self, weights: np.ndarray, lam: float = 0.01) -> np.ndarray:
        """Pull weights back toward anchor on important synapses."""
        return lam * self.importance * (self.anchor - weights)


class BennaFusiCascade:
    """N coupled scalars per synapse, exchanging mass at geometrically
    increasing timescales.  Returns a single visible weight (sum)."""

    def __init__(self, shape: Tuple[int, ...], n_levels: int = 4,
                 base_tau: float = 1.0):
        self.shape = tuple(int(s) for s in shape)
        self.U = [np.zeros(self.shape, dtype=np.float32) for _ in range(int(n_levels))]
        # taus: 1, 4, 16, 64, ...
        self.taus = [float(base_tau) * (4.0 ** k) for k in range(int(n_levels))]

    def inject(self, delta: np.ndarray) -> None:
        """A learning step injects mass into the fastest variable."""
        self.U[0] += delta.astype(np.float32)

    def relax(self) -> None:
        """Flow mass downward through the cascade (decay)."""
        for k in range(len(self.U) - 1):
            tau = self.taus[k]
            flow = (self.U[k] - self.U[k + 1]) / tau
            self.U[k] -= flow
            self.U[k + 1] += flow / (self.taus[k + 1] / tau)

    def weight(self, weights: list[float] | None = None) -> np.ndarray:
        if weights is None:
            weights = [1.0 / (k + 1) for k in range(len(self.U))]
        out = np.zeros(self.shape, dtype=np.float32)
        for c, u in zip(weights, self.U):
            out = out + c * u
        return out.astype(np.float32)
