"""
Salience Network — Gap 6: Genius Brain Architecture.
=====================================================

The Salience Network (anterior insula + dorsal anterior cingulate) detects
what's important and acts as the master switch between:
  - The Default Mode Network (creative daydreaming, free association)
  - The Executive Control Network (focused, goal-directed processing)

In genius brains, the Salience Network is more finely tuned: it allows
the DMN to run productively even during focused tasks, rather than
suppressing it completely like in average brains.
"""
from __future__ import annotations
import numpy as np


class SalienceNetwork:
    """Detects input novelty/importance and routes processing mode.

    Scores each input on two axes:
      1. Novelty: How different is this from recent experience?
      2. Emotional weight: How strong is the neuromodulatory response?

    High salience → route to full cortex (Executive/focused mode)
    Low salience → route to DMN (background associative processing)
    """

    def __init__(self, latent_dim: int, history_size: int = 100,
                 novelty_threshold: float = 0.3,
                 emotional_threshold: float = 0.4,
                 seed: int = 0):
        self.latent_dim = latent_dim
        self._history = []
        self.history_size = history_size
        self.novelty_threshold = novelty_threshold
        self.emotional_threshold = emotional_threshold
        self._rng = np.random.default_rng(seed + 5555)

        # Running centroid of recent inputs (prototype of "normal")
        self._centroid = np.zeros(latent_dim, dtype=np.float32)
        self._centroid_count = 0

    def score(self, z_input: np.ndarray,
              dopamine: float = 0.5,
              cortisol: float = 0.0,
              prediction_error: float = 0.0) -> dict:
        """Score the salience of an input vector.

        Returns a dict with:
          - novelty: float [0, 1]
          - emotional_weight: float [0, 1]
          - salience: float [0, 1] (combined score)
          - route: "executive" or "dmn"
        """
        # --- Novelty: distance from running centroid ---
        if self._centroid_count > 0:
            diff = z_input.astype(np.float32) - self._centroid
            distance = float(np.sqrt(np.mean(diff ** 2)))
            # Normalize against typical distances
            novelty = min(1.0, distance / 2.0)
        else:
            novelty = 1.0  # Everything is novel at birth

        # Update centroid (exponential moving average)
        alpha = 1.0 / (self._centroid_count + 1) if self._centroid_count < 100 else 0.01
        self._centroid = (1 - alpha) * self._centroid + alpha * z_input.astype(np.float32)
        self._centroid_count += 1

        # --- Emotional weight: combine neuromodulatory signals ---
        emotional_weight = float(np.clip(
            0.3 * abs(dopamine - 0.5) +  # Dopamine deviation from baseline
            0.3 * cortisol +              # Stress amplifies salience
            0.4 * prediction_error,       # Surprise amplifies salience
            0.0, 1.0
        ))

        # --- Combined salience ---
        salience = 0.6 * novelty + 0.4 * emotional_weight

        # --- Routing decision ---
        if salience > max(self.novelty_threshold, self.emotional_threshold):
            route = "executive"
        else:
            route = "dmn"

        return {
            "novelty": novelty,
            "emotional_weight": emotional_weight,
            "salience": salience,
            "route": route,
        }

    def snapshot(self) -> dict:
        return {
            "centroid_count": self._centroid_count,
            "novelty_threshold": self.novelty_threshold,
            "emotional_threshold": self.emotional_threshold,
        }
