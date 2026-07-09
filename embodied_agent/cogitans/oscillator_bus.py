"""
Multi-timescale Oscillator Bus (v9 — Aurora-Cogitans).
=======================================================

Every step of a current LLM (and of Aurora v3–v8) ticks a single global
clock: one step per input, all modules updated at the same cadence.
Biological brains do the opposite.  They multiplex several bands of
oscillation simultaneously, and different processes phase-lock to
different bands:

  * **δ (delta, ~1-4 Hz)**   – slow-wave sleep, consolidation
  * **θ (theta, ~4-8 Hz)**   – episodic encoding & retrieval, hippocampal
                                indexing of items in working memory
  * **α (alpha, ~8-13 Hz)**  – inhibitory gating / attention suppression
  * **β (beta, ~13-30 Hz)**  – motor set / policy commitment
  * **γ (gamma, ~30-80 Hz)** – feature binding, local cortical
                                microcircuit activity

The functional value of this multiplexing is well-established: theta-
gamma phase-amplitude coupling is the leading candidate mechanism for
how the brain binds a *sequence* of ~4-7 items into a working-memory
episode (Lisman & Jensen 2013).  Without it, WM items would collide.

v8's step function does none of this.  Every module runs every step at
the same rate; the sleep cycle is a *phase* triggered periodically but
inside a phase, everything still ticks at 1 Hz.

The oscillator bus supplies phase state without forcing us to run at a
1 kHz sample rate.  It maintains phase for each band, advances phases
each step (with each band having a *cycle length in steps*), and exposes
two derived signals:

  * ``phase(band)`` — current phase ∈ [0, 2π)
  * ``gate(band)``  — cos(phase) in [-1, 1], convenient for direct use
                       as a plasticity gate.

Modules can then *conditionally* update:

  * WM slot rotation on theta gate crossings ("this is where a new
    chunk can be admitted")
  * hippocampal replay preference weighted by delta gate during sleep
  * cortex plasticity multiplied by (1 + 0.5·γ_gate) inside the top-K
    active regions — realising the "gamma binds features" story
  * policy commitment only when β gate > 0 (motor "go" window)

This does not require a real-time simulation — the number of "brain
seconds" per organism step is just a config number
(``steps_per_second``).  The bands' cycle lengths in steps are computed
from that.

The bus is a *pure kinematic* module: it holds phase state, does not
mutate any other module.  Coupling is arranged at the organism level.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional
import math
import numpy as np


# The user specifically requested "golden ratio" tuning. In computational
# neuroscience (e.g., Buzsáki's "Rhythms of the Brain"), cortical rhythms
# are observed to be separated by approximately the golden ratio, preventing
# integer-multiple phase locking and allowing multi-band multiplexing.
PHI = (1 + math.sqrt(5)) / 2

DEFAULT_BANDS_HZ: Dict[str, float] = {
    "delta": float(PHI**2),   # ~2.6 Hz
    "theta": float(PHI**4),   # ~6.8 Hz
    "alpha": float(PHI**5),   # ~11.1 Hz
    "beta":  float(PHI**6),   # ~17.9 Hz
    "gamma": float(PHI**8),   # ~47.0 Hz
}


@dataclass
class _Band:
    hz: float
    phase: float = 0.0
    delta_phase: float = 0.0
    coupling: float = 1.0     # optional slow modulation of amplitude


class OscillatorBus:
    """A multi-band phase clock the organism can query each step.

    Parameters
    ----------
    steps_per_second : float
        How many organism steps correspond to one biological second.
        The default 100.0 means each step is 10 ms of "brain time",
        which is roughly the fastest realistic Aurora tick.
    bands : dict[str, float] | None
        Band-name → centre-frequency-in-Hz.  Defaults to the standard
        five (δ θ α β γ).
    theta_gamma_coupling : bool
        If True, gamma amplitude is modulated by theta phase (Lisman-
        Jensen style PAC) — the biological substrate for chunking of
        ~4-7 items into a WM episode.
    """

    def __init__(self, steps_per_second: float = 100.0,
                 bands: Optional[Dict[str, float]] = None,
                 theta_gamma_coupling: bool = True,
                 seed: int = 1729):
        assert steps_per_second > 0
        self.steps_per_second = float(steps_per_second)
        self.theta_gamma_coupling = bool(theta_gamma_coupling)
        cfg = dict(bands or DEFAULT_BANDS_HZ)

        rng = np.random.default_rng(seed)
        self.bands: Dict[str, _Band] = {}
        for name, hz in cfg.items():
            b = _Band(hz=float(hz))
            # phase change per step = 2π · hz / steps_per_second
            b.delta_phase = 2.0 * math.pi * b.hz / self.steps_per_second
            b.phase = float(rng.uniform(0.0, 2.0 * math.pi))
            self.bands[name] = b

        self._n_steps = 0

    # ------------------------------------------------------------------
    def tick(self, adrenaline: float = 0.0) -> None:
        """Advance every band's phase by its per-step increment.
        
        Adrenaline > 0 induces 'Time Dilation' by speeding up the biological
        clock (increasing delta_phase), allowing the organism to process
        and gate high-frequency signals faster during danger/surprise.
        """
        dilation = 1.0 + float(adrenaline)
        for b in self.bands.values():
            b.phase = (b.phase + (b.delta_phase * dilation)) % (2.0 * math.pi)
        self._n_steps += 1

    # ------------------------------------------------------------------
    def phase(self, band: str) -> float:
        b = self.bands.get(band)
        return float(b.phase) if b is not None else 0.0

    def gate(self, band: str) -> float:
        """Return cos(phase(band)) — in [-1, 1].

        Callers typically use ``0.5 * (1 + gate)`` when they want a
        [0, 1] modulator (e.g. plasticity multiplier).  If theta-gamma
        coupling is on and ``band == 'gamma'``, the amplitude is scaled
        by ``0.5 * (1 + cos(θ_phase))`` — gamma bursts ride on theta
        troughs, so *bindings* only get amplitude when theta gates them
        in.
        """
        b = self.bands.get(band)
        if b is None:
            return 0.0
        g = math.cos(b.phase)
        if self.theta_gamma_coupling and band == "gamma":
            theta = self.bands.get("theta")
            if theta is not None:
                amp = 0.5 * (1.0 + math.cos(theta.phase))
                g = g * (0.35 + 0.65 * amp)   # coupled gamma
        return float(g)

    def all_gates(self) -> Dict[str, float]:
        return {name: self.gate(name) for name in self.bands}

    # ------------------------------------------------------------------
    def is_pac_binding_window(self, threshold: float = 0.7) -> bool:
        """True when we're in the "chunk this into WM" theta-gamma peak.

        The condition is that (a) we're near the peak of a theta cycle
        AND (b) coupled gamma is above threshold.  This is when
        biological WM binds ~4 items via nested gamma cycles.
        """
        theta = self.bands.get("theta")
        gamma = self.bands.get("gamma")
        if theta is None or gamma is None:
            return False
        theta_gate = 0.5 * (1.0 + math.cos(theta.phase))
        gamma_gate = 0.5 * (1.0 + self.gate("gamma"))
        return bool(theta_gate * gamma_gate > threshold)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return dict(
            steps=self._n_steps,
            steps_per_second=self.steps_per_second,
            theta_gamma_coupling=self.theta_gamma_coupling,
            bands={n: dict(hz=b.hz, phase=float(b.phase),
                            gate=self.gate(n))
                    for n, b in self.bands.items()},
        )
