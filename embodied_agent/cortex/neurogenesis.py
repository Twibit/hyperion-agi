"""
Neurogenesis & lottery-ticket regrowth — v6.
=============================================

Maintains an active-mask over a population of "neurons" (e.g. MoE expert
slots, hippocampal slots, etc.).  Every `cycle` steps, the lowest-utility
neurons are pruned and replaced with fresh random initializations
sampled from a lottery-ticket distribution.

The result is a slowly-renewing reservoir of capacity: routinely-used
features get to live long, unused features get reborn.  This is the AI
analogue of adult neurogenesis in the dentate gyrus.
"""
from __future__ import annotations
import numpy as np
from collections import deque
from typing import Deque


class NeurogenesisController:
    def __init__(self, n_neurons: int, cycle: int = 200,
                 prune_frac: float = 0.05, seed: int = 0):
        self.n = int(n_neurons)
        self.cycle = int(cycle)
        self.prune_frac = float(prune_frac)
        self.mask = np.ones(self.n, dtype=np.float32)
        self.utility = np.zeros(self.n, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.births = 0
        self.deaths = 0
        self.steps = 0
        self.growth_history: Deque[int] = deque(maxlen=64)

    def report_usage(self, neuron_ids, signals=None) -> None:
        if signals is None:
            signals = np.ones(len(neuron_ids), dtype=np.float32)
        for i, s in zip(neuron_ids, signals):
            self.utility[int(i) % self.n] = (0.99 * self.utility[int(i) % self.n]
                                             + 0.01 * float(s))

    def maybe_grow(self) -> list[int]:
        self.steps += 1
        if self.steps % self.cycle != 0:
            return []
        # find lowest-utility neurons
        k = max(1, int(self.prune_frac * self.n))
        victims = np.argsort(self.utility)[:k]
        self.utility[victims] = 0.0
        self.mask[victims] = 1.0  # ensure they are active
        self.deaths += k
        self.births += k
        self.growth_history.append(k)
        return [int(v) for v in victims]

    def stats(self) -> dict:
        return dict(
            n=self.n, cycle=self.cycle,
            births=self.births, deaths=self.deaths,
            active_frac=float(self.mask.mean()),
            history=list(self.growth_history)[-16:],
        )
