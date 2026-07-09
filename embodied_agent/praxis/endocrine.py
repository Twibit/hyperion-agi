"""
Endocrine Expansion — V11 (Aurora-Praxis).
==========================================

The original v3 NeuromodulatorState models the four fast-acting synaptic
modulators (DA, 5HT, NE, ACh). But the biological brain also has a slower
*endocrine* (hormonal) layer that operates on a completely different timescale:

  * **Adrenaline (Epinephrine):** Secreted by the adrenal medulla during
    fight-or-flight. It causes 'Time Dilation' — subjectively slowing down
    time by accelerating internal clock speed, allowing faster reaction to
    threats. In Aurora, adrenaline speeds up the OscillatorBus tick rate.

  * **Oxytocin:** The 'bonding' hormone. In a multi-agent or social context,
    oxytocin modulates the MirrorSystem's empathy threshold and increases
    cooperative/imitative behaviour. In Aurora, it lowers the mirror match
    threshold so the organism is more willing to learn from observed actions.

  * **Melatonin:** The sleep hormone. It accumulates slowly during waking
    and triggers the sleep cycle when it crosses a threshold. In Aurora,
    melatonin provides a biological clock for sleep pressure, complementing
    the existing periodic `consolidation_every` counter.

All three hormones decay slowly (τ ~ 0.001–0.01) compared to the fast
neuromodulators (τ ~ 0.02–0.10), reflecting their minutes-to-hours timescale
in real biology.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class EndocrineState:
    """Slow-acting hormonal layer complementing NeuromodulatorState."""

    # Current levels (0.0 = baseline, 1.0 = maximum secretion)
    adrenaline: float = 0.0
    oxytocin: float = 0.0
    melatonin: float = 0.0

    # Decay rates (very slow — these are hormones, not neurotransmitters)
    tau_adrenaline: float = 0.008   # ~125 steps half-life
    tau_oxytocin: float = 0.003     # ~333 steps half-life
    tau_melatonin: float = 0.001    # ~1000 steps half-life

    # Accumulation rates
    melatonin_wake_accum: float = 0.0005  # slow buildup during waking

    # Thresholds
    sleep_pressure_threshold: float = 0.7  # melatonin level that triggers sleep

    def update(self, *,
               sensory_surprise: float = 0.0,
               cortisol: float = 0.0,
               social_signal: float = 0.0,
               is_sleeping: bool = False) -> None:
        """
        Tick the endocrine system.

        sensory_surprise : high values trigger adrenaline release (fight-or-flight)
        cortisol         : sustained stress amplifies adrenaline
        social_signal    : positive social interaction releases oxytocin
        is_sleeping      : melatonin drops rapidly during sleep
        """
        # --- Adrenaline ---
        # Spike on surprise/danger, amplified by existing cortisol
        adrenaline_input = sensory_surprise * (1.0 + 0.5 * cortisol)
        self.adrenaline += 0.3 * adrenaline_input
        # Decay toward 0
        self.adrenaline -= self.tau_adrenaline * self.adrenaline
        self.adrenaline = float(np.clip(self.adrenaline, 0.0, 1.0))

        # --- Oxytocin ---
        # Released during positive social interactions
        self.oxytocin += 0.2 * max(0.0, social_signal)
        # Decay toward 0
        self.oxytocin -= self.tau_oxytocin * self.oxytocin
        self.oxytocin = float(np.clip(self.oxytocin, 0.0, 1.0))

        # --- Melatonin ---
        if is_sleeping:
            # Melatonin drops rapidly during sleep (consumed)
            self.melatonin *= 0.5
        else:
            # Slowly accumulates during waking
            self.melatonin += self.melatonin_wake_accum
        # Decay toward 0 (very slow natural clearance)
        self.melatonin -= self.tau_melatonin * self.melatonin
        self.melatonin = float(np.clip(self.melatonin, 0.0, 1.0))

    @property
    def sleep_pressure(self) -> float:
        """How urgently the organism needs to sleep (0.0–1.0)."""
        return float(self.melatonin / self.sleep_pressure_threshold)

    @property
    def time_dilation(self) -> float:
        """
        Adrenaline-driven time dilation factor.
        0.0 = normal speed, up to ~2.0 = triple-speed internal clock.
        This is fed directly into OscillatorBus.tick(adrenaline=...).
        """
        return float(self.adrenaline * 2.0)

    @property
    def empathy_boost(self) -> float:
        """
        Oxytocin-driven empathy modifier.
        Lowers the MirrorSystem's match threshold, making the organism
        more willing to imitate and learn from observed actions.
        Returns a value in [0.0, 0.3] that is *subtracted* from the threshold.
        """
        return float(self.oxytocin * 0.3)

    def snapshot(self) -> Dict[str, float]:
        return {
            "adrenaline": self.adrenaline,
            "oxytocin": self.oxytocin,
            "melatonin": self.melatonin,
            "time_dilation": self.time_dilation,
            "sleep_pressure": self.sleep_pressure,
            "empathy_boost": self.empathy_boost,
        }
