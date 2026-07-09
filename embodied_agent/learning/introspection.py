"""
Introspection — self-distillation & monotone-error guard (v6).
===============================================================

Tracks a baseline error EMA and, when a meaningful drop occurs, snapshots
the current cortex state into a "best-so-far" buffer.  If error regresses
substantially, the introspector recommends a revert.

This is the AI analogue of metacognitive monitoring — knowing when you
just got worse at something and pulling back to a previously-known good
state, instead of continuing to train into a regression.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Optional


class Introspector:
    def __init__(self, drop_thresh: float = 0.05, revert_thresh: float = 0.15):
        self.err_ema = 0.5
        self.best = 1.0
        self.snapshot: Optional[Any] = None
        self.distillations = 0
        self.reverts = 0
        self.last_error_drop = 0.0
        self.drop_thresh = float(drop_thresh)
        self.revert_thresh = float(revert_thresh)

    def observe(self, err: float, snapshot_fn=None) -> dict:
        prev = self.err_ema
        self.err_ema = 0.95 * self.err_ema + 0.05 * float(err)
        self.last_error_drop = prev - self.err_ema
        out = dict(distill=False, revert=False)
        # New best -> snapshot.
        if self.err_ema < self.best - self.drop_thresh:
            self.best = self.err_ema
            if snapshot_fn is not None:
                self.snapshot = snapshot_fn()
            self.distillations += 1
            out["distill"] = True
        # Regression -> recommend revert.
        elif self.err_ema > self.best + self.revert_thresh and self.snapshot is not None:
            self.reverts += 1
            out["revert"] = True
        return out

    def stats(self) -> dict:
        return dict(
            err_ema=self.err_ema, best=self.best,
            distillations=self.distillations, reverts=self.reverts,
            last_drop=self.last_error_drop,
        )
