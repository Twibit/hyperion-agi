"""
Astrocyte Metaplasticity Field — v6.
=====================================

Glial astrocytes modulate the per-region learning rate based on a slow
moving average of local prediction error.  Regions that are doing well
get their gain dialed down (consolidation); regions that are surprised
get a temporary gain boost (focused learning).

This is the AI analogue of attention-gated metaplasticity observed in
hippocampal CA1 and cortical L2/3.
"""
from __future__ import annotations
import numpy as np


class AstrocyteField:
    def __init__(self, n_regions: int, decay: float = 0.98,
                 gmin: float = 0.25, gmax: float = 2.5, seed: int = 0):
        self.n = int(n_regions)
        self.decay = float(decay)
        self.gmin = float(gmin)
        self.gmax = float(gmax)
        self.err_ema = np.full(self.n, 0.5, dtype=np.float32)
        self.g = np.ones(self.n, dtype=np.float32)

    def report_error(self, region_id: int, err: float) -> float:
        i = int(region_id) % self.n
        self.err_ema[i] = self.decay * self.err_ema[i] + (1.0 - self.decay) * float(err)
        # Higher error -> higher gain (focus).  Lower error -> lower gain
        # (let the region consolidate).
        target = 1.0 + 2.0 * (self.err_ema[i] - 0.5)
        self.g[i] = float(np.clip(0.9 * self.g[i] + 0.1 * target,
                                  self.gmin, self.gmax))
        return self.g[i]

    def gain(self, region_id: int) -> float:
        return float(self.g[int(region_id) % self.n])

    def stats(self) -> dict:
        return dict(
            regions=self.n,
            g_mean=float(self.g.mean()),
            g_min=float(self.g.min()),
            g_max=float(self.g.max()),
            err_mean=float(self.err_ema.mean()),
        )
