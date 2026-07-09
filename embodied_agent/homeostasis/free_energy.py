"""
FreeEnergyMonitor — V20.

Aggregates the free-energy scalars already produced by the cortex into
a single time-series, and exposes helpers used by both the
AdaptiveTierController and the neurogenesis code path.

Audit item #12: "Free-Energy loss aggregation". The KL-like load-balance
penalty is already computed by `_Router.load_balance_penalty()`; each
`PredictiveBlock` already has `_err_var` from which it derives
`precision`. V20 does not add new math — it just wires those existing
scalars into one number: F = err² · precision + 0.5 · ln(var) +
β · load_imbalance.

Highest ROI-per-LOC item in the whole V20 roadmap.
"""
from __future__ import annotations
from collections import deque


class FreeEnergyMonitor:
    def __init__(self, window: int = 32, beta_kl: float = 0.1):
        self.window = int(window)
        self.beta_kl = float(beta_kl)
        self._history: deque[float] = deque(maxlen=self.window)

    def observe(self, cortex) -> float:
        """Compute F(cortex) and push it into the trailing window."""
        fe = float(cortex.free_energy())
        self._history.append(fe)
        return fe

    @property
    def latest(self) -> float:
        return self._history[-1] if self._history else 0.0

    @property
    def mean(self) -> float:
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    def slope(self) -> float:
        """Signed slope of the trailing FE window (negative = improving)."""
        n = len(self._history)
        if n < 2:
            return 0.0
        xs = list(range(n))
        ys = list(self._history)
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs) or 1.0
        return num / den
