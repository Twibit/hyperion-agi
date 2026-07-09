"""
Neural Efficiency Engine — Gap 1: Genius Brain Architecture.
=============================================================

The Neural Efficiency Hypothesis (Haier et al.) shows that genius brains
use LESS activation for easy tasks and MORE for hard tasks. This module
dynamically adjusts the number of active experts (k) in the MoE Cortex
based on the difficulty of the input signal.

For easy inputs (low prediction error): k=1 (minimal compute, thrifty)
For hard inputs (high prediction error): k=n_experts (full recruitment)
"""
from __future__ import annotations
import numpy as np


class NeuralEfficiencyGate:
    """Dynamically gates how many cortical experts fire per input.

    Monitors a running estimate of prediction error. When the error is
    low (the cortex already understands the input), it reduces the
    number of active experts to save metabolic energy. When the error
    spikes (novel or complex input), it recruits the full bank.

    This mirrors the Neural Efficiency Hypothesis: gifted brains show
    *less* activation for routine tasks but *more* for genuinely hard ones.
    """

    def __init__(self, n_experts: int, k_min: int = 1, k_max: int | None = None,
                 easy_threshold: float = 0.05, hard_threshold: float = 0.3,
                 ema_alpha: float = 0.01):
        self.n_experts = n_experts
        self.k_min = max(1, k_min)
        self.k_max = k_max or n_experts
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold
        self._ema_err = 0.15  # Start at mid-range
        self._alpha = ema_alpha
        self._current_k = self.k_max  # Start fully active

    def update(self, prediction_error: float) -> int:
        """Update the running error estimate and return the optimal k."""
        self._ema_err = (1 - self._alpha) * self._ema_err + self._alpha * prediction_error

        if self._ema_err < self.easy_threshold:
            # Easy input — engage minimal experts (thrifty genius mode)
            self._current_k = self.k_min
        elif self._ema_err > self.hard_threshold:
            # Hard input — recruit everything (full engagement)
            self._current_k = self.k_max
        else:
            # Proportional scaling between easy and hard
            frac = (self._ema_err - self.easy_threshold) / (self.hard_threshold - self.easy_threshold)
            self._current_k = int(self.k_min + frac * (self.k_max - self.k_min))
            self._current_k = max(self.k_min, min(self.k_max, self._current_k))

        return self._current_k

    @property
    def efficiency_ratio(self) -> float:
        """How efficient the brain is being (1.0 = maximally thrifty)."""
        return 1.0 - (self._current_k - self.k_min) / max(1, self.k_max - self.k_min)

    def snapshot(self) -> dict:
        return {
            "current_k": self._current_k,
            "ema_err": float(self._ema_err),
            "efficiency": self.efficiency_ratio,
        }
