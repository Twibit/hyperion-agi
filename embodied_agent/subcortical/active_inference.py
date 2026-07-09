"""
Active-Inference action selector — v6.
=======================================

Arbitrates between two policies:

  * habitual  — the basal-ganglia softmax policy (cheap, automatic)
  * deliberative — short imagined rollouts that score expected free energy
                   (= epistemic value - extrinsic reward)

Switches to deliberation when the habit's uncertainty is high or
homeostatic state is dysregulated.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Optional


class ActiveInferenceSelector:
    def __init__(self, n_actions: int,
                 uncertainty_thresh: float = 0.5,
                 horizon: int = 3,
                 seed: int = 0):
        self.n_actions = int(n_actions)
        self.thresh = float(uncertainty_thresh)
        self.horizon = int(horizon)
        self.rng = np.random.default_rng(seed)
        self.deliberations = 0
        self.habits = 0

    def select(self, z: np.ndarray, habit_action: int, habit_probs: np.ndarray,
               uncertainty: float, world_forward: Callable, value_fn: Callable,
               cortisol: float = 0.0) -> tuple[int, str]:
        # Always deliberate under stress *or* high uncertainty.
        if uncertainty < self.thresh and cortisol < 0.5:
            self.habits += 1
            return int(habit_action), "habit"

        # Sample a few candidate actions and roll them out shortly.
        top_k = min(4, self.n_actions)
        candidates = np.argsort(-habit_probs)[:top_k]
        best_a = int(habit_action)
        best_score = -np.inf
        for a in candidates:
            z_sim = z.copy()
            G = 0.0  # expected free energy proxy
            for _ in range(self.horizon):
                z_next, r, u = world_forward(z_sim, int(a))
                # epistemic = -uncertainty (we prefer reducing uncertainty)
                # extrinsic = value of resulting state
                G += float(r) + 0.2 * value_fn(z_next) - 0.3 * float(u)
                z_sim = z_next
            if G > best_score:
                best_score = G
                best_a = int(a)
        self.deliberations += 1
        return best_a, "deliberation"

    def stats(self) -> dict:
        total = max(1, self.deliberations + self.habits)
        return dict(deliberations=self.deliberations, habits=self.habits,
                    deliberation_ratio=self.deliberations / total)
