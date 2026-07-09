"""
Adaptive Plasticity Schedule — Gap 3: Genius Brain Architecture.
=================================================================

Gifted children show a unique cortical development trajectory:
  Phase 1: Cortex starts thinner, learning rate is aggressive (rapid growth)
  Phase 2: Cortex thickens more than average (peak plasticity, refinement)
  Phase 3: Vigorous pruning — cortex thins again (maturation, precision)

This module tracks the organism's "developmental age" (total steps) and
dynamically adjusts the learning rate to mirror this biological trajectory.
A fixed lr=3e-4 forever is how you build an average brain. This schedule
is how you build a genius.
"""
from __future__ import annotations
import math


class AdaptivePlasticitySchedule:
    """Mimics the cortical development trajectory of gifted individuals.

    Uses a bell-curve learning rate that rises aggressively in youth,
    peaks at the "critical period," then gradually decays to precision
    refinement in maturity.
    """

    def __init__(self,
                 lr_initial: float = 1e-2,
                 lr_peak: float = 3e-3,
                 lr_mature: float = 1e-4,
                 growth_phase_steps: int = 100_000,
                 peak_phase_steps: int = 1_000_000,
                 warmup_steps: int = 1000):
        self.lr_initial = lr_initial
        self.lr_peak = lr_peak
        self.lr_mature = lr_mature
        self.growth_end = growth_phase_steps
        self.peak_end = growth_phase_steps + peak_phase_steps
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self) -> float:
        """Advance one step and return the current learning rate."""
        self._step += 1
        return self.current_lr

    @property
    def current_lr(self) -> float:
        s = self._step

        # Warmup: ramp from 0 to initial
        if s < self.warmup_steps:
            return self.lr_initial * (s / max(1, self.warmup_steps))

        # Phase 1: Growth (aggressive, rapid cortical thickening)
        if s < self.growth_end:
            # Cosine ramp from initial up to peak
            progress = (s - self.warmup_steps) / max(1, self.growth_end - self.warmup_steps)
            return self.lr_initial + (self.lr_peak - self.lr_initial) * (
                0.5 * (1 - math.cos(math.pi * progress))
            )

        # Phase 2: Peak plasticity (refinement, critical period)
        if s < self.peak_end:
            return self.lr_peak

        # Phase 3: Maturation (precision pruning, slow decay)
        # Exponential decay toward lr_mature
        overshoot = s - self.peak_end
        decay_rate = 1e-6  # Very slow decay
        return self.lr_mature + (self.lr_peak - self.lr_mature) * math.exp(-decay_rate * overshoot)

    @property
    def phase_name(self) -> str:
        if self._step < self.warmup_steps:
            return "warmup"
        elif self._step < self.growth_end:
            return "growth"
        elif self._step < self.peak_end:
            return "peak_plasticity"
        else:
            return "maturation"

    def set_step(self, step: int):
        """Resume from a saved step count."""
        self._step = step

    def snapshot(self) -> dict:
        return {
            "step": self._step,
            "phase": self.phase_name,
            "lr": self.current_lr,
        }
