"""
Metabolic budget (v8 — Aurora-Sapiens).
=======================================

The human brain runs on ~20 W and cannot activate more than ~2 % of its
synapses simultaneously without exceeding its glucose/O₂ supply.  This
is not decoration — it is *the* mechanism that enforces the sparsity the
v3 RATIONALE claims (§3.1: "~1 % of cortex fires in any given 100 ms
window").

In v7 nothing enforces this at runtime: every region proxy runs every
step at full lr.  The 100T-synapse census reports a beautiful sparsity
number (~1.3 %) — but that number comes from the *static* ``active_frac``
field, not from anything the simulator actually does.

`MetabolicBudget` fixes that.  Each step it:

1. Estimates the "cost" of activating each region (proportional to its
   ``synapses × active_frac``).
2. Given a total ATP budget, picks the top-K regions whose cost fits
   inside the budget.  Regions outside the top-K get their lr gain
   multiplied by ``rest_gain`` (default 0.05) — they still run, but with
   near-frozen weights, mimicking the resting-state metabolic floor.
3. Emits a report of realised sparsity so the user can verify the
   simulator is honouring the biological constraint.

Combined with `NeuromodulatorBus`, this reproduces the real biological
allocation logic: NE/ACh nominate salient regions, glucose limits how
many can burn hot at once.

Notes on the budget number
--------------------------
`budget_frac` is the fraction of total synapses allowed to fire this
step.  Default 0.02 (2 %) matches the neuroscience upper bound; the
`cerebrum` tier can dial it up if simulating altered-state activity
(e.g. seizure = 0.1, deep sleep = 0.005).
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np


class MetabolicBudget:
    """Per-step ATP/glucose-like allocator across brain regions.

    Parameters
    ----------
    atlas : BrainAtlas
        Provides the static ``synapses × active_frac`` cost per region.
    budget_frac : float
        Fraction of total synapses allowed to be "on" per step.  Default
        0.02 corresponds to the ~2 % cortical activation ceiling.
    rest_gain : float
        lr multiplier applied to regions that do *not* fit in the budget
        this step.  0.05 = "resting state" (near-frozen weights).
    priority_bias : float
        Weight on the incoming ``priority`` signal when ranking regions;
        higher = more responsive to salience/NE, lower = more inertial.
    """

    def __init__(self, atlas, budget_frac: float = 0.02,
                 rest_gain: float = 0.05, priority_bias: float = 1.0):
        self.atlas = atlas
        self.budget_frac = float(budget_frac)
        self.rest_gain = float(rest_gain)
        self.priority_bias = float(priority_bias)

        total = atlas.total_synapses()
        self.total_synapses = int(total)
        self.budget = int(total * self.budget_frac)

        # V14 Polyglot: Mixed-Substrate Energy Accounting
        # Rust SNN regions are 10x more energy efficient than NumPy matrix multiplications.
        # This naturally rewards the organism for migrating tissue to the efficient substrate.
        self._cost: Dict[str, int] = {}
        for r in atlas.regions:
            base_cost = int(r.synapses * max(0.001, r.active_frac))
            substrate_multiplier = 0.1 if r.parent_id in ("cerebrum_left", "cerebrum_right") else 1.0
            self._cost[r.name] = int(base_cost * substrate_multiplier)

        self._last_active: Dict[str, bool] = {}
        self._usage_stats: Dict[str, float] = {r.name: 0.0 for r in atlas.regions}
        self._last_used_frac: float = 0.0
        self._n_steps = 0

    # ------------------------------------------------------------------
    def allocate(self, priorities: Dict[str, float]) -> Dict[str, float]:
        """Return per-region lr multiplier under the metabolic constraint.

        Parameters
        ----------
        priorities : dict region -> float
            Salience score (from thalamus / NE / connectome).  Higher →
            more likely to be picked.

        Returns
        -------
        dict region -> gain (in [rest_gain, 1.0])
        """
        # Rank by priority (default 1.0 if a region isn't in the dict).
        ranked = sorted(
            self._cost.keys(),
            key=lambda n: -(self.priority_bias * priorities.get(n, 1.0)),
        )

        gains: Dict[str, float] = {}
        active: Dict[str, bool] = {}
        used = 0
        for name in ranked:
            c = self._cost[name]
            if used + c <= self.budget:
                gains[name] = 1.0
                active[name] = True
                used += c
            else:
                gains[name] = self.rest_gain
                active[name] = False
                
            # Update EMA usage tracking for V12 Mitosis
            current_usage = 1.0 if active[name] else 0.0
            self._usage_stats[name] = 0.99 * self._usage_stats.get(name, 0.0) + 0.01 * current_usage

        self._last_active = active
        self._last_used_frac = used / max(1, self.total_synapses)
        self._n_steps += 1
        return gains

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return dict(
            budget_frac=self.budget_frac,
            realised_active_frac=float(self._last_used_frac),
            regions_active=sum(1 for v in self._last_active.values() if v),
            regions_resting=sum(1 for v in self._last_active.values() if not v),
            steps=self._n_steps,
            total_synapses=self.total_synapses,
        )
