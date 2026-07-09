"""
Neuromodulator Bus (v8 — Aurora-Sapiens).
=========================================

In v7 the connectome models dopaminergic / noradrenergic / serotonergic /
cholinergic bundles as *structural* edges — they exist in the graph but
carry no runtime effect during ``Cerebrum.step()``: every region receives
exactly the same learning-rate scalar ``lr``.

That is anatomically incomplete.  Biologically, those bundles are the
*sole* mechanism by which arousal (NE), reward-prediction (DA), mood
(5-HT) and attention (ACh) modulate cortical plasticity on a
region-by-region basis.  A cortex column innervated by the locus
coeruleus experiences a different LTP curve from one that is not.

`NeuromodulatorBus` closes the loop.  It reads the four global
neuromodulator levels from ``NeuromodulatorState`` and multiplies them
into region-specific learning-rate gains along the paths that the
connectome actually contains.  It is completely additive — every region
still gets its baseline lr; the bus only *modulates* it.

The gains are exposed to `Cerebrum.step(..., modulator_gains=...)` which
the v7 code already accepts but never used.  v8 wires it up.

Design choices
--------------
* Gains are computed once per step from a single ``NeuromodulatorState``
  snapshot; no per-region state is stored.  This keeps the bus a *pure
  function* of the connectome plus modulator levels.
* Bundle weights from the connectome are re-used directly — we do *not*
  invent new numbers.
* The output is clipped to ``[gain_min, gain_max]`` so a pathological
  cortisol spike cannot zero out plasticity everywhere.

References
----------
* Aston-Jones & Cohen 2005 (LC-NE gain modulation)
* Yu & Dayan 2005 (ACh as expected uncertainty, NE as unexpected)
* Schultz 1998 (DA reward prediction error)
* Doya 2002 (four-modulator functional decomposition)
"""
from __future__ import annotations
from typing import Dict
import numpy as np


# Mapping from connectome bundle names to which modulator scalar drives them.
_BUNDLE_TO_MODULATOR = {
    "dopaminergic":  "dopamine",
    "noradrenergic": "norepinephrine",
    "serotonergic":  "serotonin",
    "cholinergic":   "acetylcholine",
}


class NeuromodulatorBus:
    """Turns the four global neuromodulators into per-region lr gains.

    Parameters
    ----------
    connectome : Connectome
        The v7 connectome; only the four neuromodulator bundles are read.
    baseline : float
        Multiplicative gain applied to every region even without any
        neuromodulator projection (usually 1.0).
    gain_min, gain_max : float
        Hard bounds on the returned gain per region.  Cortisol can pull
        gains toward ``gain_min`` but never below it.
    cortisol_suppression : float
        How strongly rising cortisol suppresses gains.  Matches the
        cortex's ``effective_lr(base, cortisol)`` semantic so that
        anatomy learning-rate suppression stays consistent with the v3
        formula ``lr · (1 − k·cortisol)``.
    """

    def __init__(self, connectome, baseline: float = 1.0,
                 gain_min: float = 0.1, gain_max: float = 2.0,
                 cortisol_suppression: float = 0.7):
        self.connectome = connectome
        self.baseline = float(baseline)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.cortisol_suppression = float(cortisol_suppression)

        # Pre-index: for each destination region, sum of |weight| for each
        # of the four neuromodulator bundles arriving at it.  Static.
        self._innervation: Dict[str, Dict[str, float]] = {}
        for e in self.connectome.edges:
            mod = _BUNDLE_TO_MODULATOR.get(e.bundle)
            if mod is None:
                continue
            row = self._innervation.setdefault(e.dst, {
                "dopamine": 0.0, "norepinephrine": 0.0,
                "serotonin": 0.0, "acetylcholine": 0.0,
            })
            row[mod] += float(abs(e.weight))

        # Book-keeping for `stats()`.
        self._last_gains: Dict[str, float] = {}
        self._n_computed = 0

    # ------------------------------------------------------------------
    def gains(self, modulators, cortisol: float = 0.0) -> Dict[str, float]:
        """Return dict region_name -> multiplicative lr gain.

        Regions with no neuromodulator input still get ``baseline``.
        Cortisol globally suppresses (mirrors ``effective_lr``).
        """
        da  = float(modulators.dopamine)
        ne  = float(modulators.norepinephrine)
        ht  = float(modulators.serotonin)
        ach = float(modulators.acetylcholine)

        # Global cortisol suppression, matching v3 effective_lr semantics.
        cort_scale = max(0.0, 1.0 - self.cortisol_suppression * float(cortisol))

        out: Dict[str, float] = {}
        for region, inn in self._innervation.items():
            # Sum of modulator level × innervation strength.
            #  - DA & ACh raise plasticity (excitatory on LTP)
            #  - 5-HT slightly damps (patience / avoid rash rewrites)
            #  - NE raises attention/plasticity acutely
            delta = (
                0.6 * da  * inn["dopamine"]      +
                0.5 * ne  * inn["norepinephrine"] +
                0.5 * ach * inn["acetylcholine"] -
                0.25 * ht * inn["serotonin"]
            )
            g = (self.baseline + delta) * cort_scale
            g = min(self.gain_max, max(self.gain_min, g))
            out[region] = float(g)

        self._last_gains = out
        self._n_computed += 1
        return out

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        if not self._last_gains:
            return dict(n_regions_innervated=len(self._innervation),
                        computed=self._n_computed,
                        gain_mean=1.0, gain_min=1.0, gain_max=1.0)
        arr = np.array(list(self._last_gains.values()), dtype=np.float32)
        return dict(
            n_regions_innervated=len(self._innervation),
            computed=self._n_computed,
            gain_mean=float(arr.mean()),
            gain_min=float(arr.min()),
            gain_max=float(arr.max()),
        )
