"""
Glial Regulation System — Gap 4: Genius Brain Architecture.
=============================================================

Einstein's brain had a significantly higher glial-to-neuron ratio in the
inferior parietal lobule. Glial cells (astrocytes) actively regulate
synaptic transmission by:
  1. Monitoring synaptic activity variance
  2. Dampening noisy/unstable synapses
  3. Boosting the conductance of precise/reliable synapses
  4. Providing metabolic support to high-demand regions

This extends the existing AstrocyteField by adding per-synapse regulation
and calcium-wave-like asynchronous sweeps.
"""
from __future__ import annotations
import numpy as np


class GlialRegulationSystem:
    """Active synaptic regulation inspired by Einstein's glial architecture.

    Operates on a per-layer basis, monitoring the variance of recent weight
    updates and applying gain control. Noisy synapses get dampened; precise
    synapses get amplified. This creates a "self-tuning" network that
    converges to efficient representations.
    """

    def __init__(self, n_layers: int, latent_dim: int,
                 regulation_interval: int = 100,
                 dampen_factor: float = 0.95,
                 boost_factor: float = 1.02,
                 noise_threshold: float = 0.5,
                 precision_threshold: float = 0.1,
                 seed: int = 0):
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.regulation_interval = regulation_interval
        self.dampen_factor = dampen_factor
        self.boost_factor = boost_factor
        self.noise_threshold = noise_threshold
        self.precision_threshold = precision_threshold

        # Per-layer tracking of recent weight update magnitudes
        self._update_history = [[] for _ in range(n_layers)]
        self._step = 0

        # Per-layer gain multipliers (start at 1.0 = neutral)
        self.gain = np.ones(n_layers, dtype=np.float32)

        # Calcium wave state: tracks which layer is being swept
        self._wave_position = 0

    def record_update(self, layer_idx: int, update_magnitude: float):
        """Record the magnitude of a weight update for a specific layer."""
        if layer_idx < self.n_layers:
            history = self._update_history[layer_idx]
            history.append(update_magnitude)
            # Keep only the last regulation_interval entries
            if len(history) > self.regulation_interval * 2:
                self._update_history[layer_idx] = history[-self.regulation_interval:]

    def regulate(self) -> dict:
        """Perform a calcium-wave sweep across all layers.

        Called periodically (every regulation_interval steps).
        Returns a dict of gain adjustments per layer.
        """
        self._step += 1
        adjustments = {}

        for i in range(self.n_layers):
            history = self._update_history[i]
            if len(history) < 10:
                continue

            recent = np.array(history[-self.regulation_interval:], dtype=np.float32)
            variance = float(np.var(recent))
            mean_mag = float(np.mean(recent))

            if variance > self.noise_threshold:
                # Noisy synapse → astrocyte dampens conductance
                self.gain[i] *= self.dampen_factor
                adjustments[i] = {"action": "dampen", "variance": variance}
            elif variance < self.precision_threshold and mean_mag > 0.001:
                # Precise synapse → astrocyte boosts conductance
                self.gain[i] *= self.boost_factor
                adjustments[i] = {"action": "boost", "variance": variance}
            else:
                adjustments[i] = {"action": "maintain", "variance": variance}

            # Clamp gain to prevent runaway
            self.gain[i] = float(np.clip(self.gain[i], 0.1, 5.0))

        return adjustments

    def get_layer_gain(self, layer_idx: int) -> float:
        """Get the current glial gain multiplier for a layer."""
        if layer_idx < self.n_layers:
            return float(self.gain[layer_idx])
        return 1.0

    def should_regulate(self) -> bool:
        """Check if it's time for a regulation sweep."""
        return self._step > 0 and self._step % self.regulation_interval == 0

    def snapshot(self) -> dict:
        return {
            "step": self._step,
            "gains": self.gain.tolist(),
        }
