"""
Prefrontal Cortex (PFC) — V11 (Aurora-Praxis).
===============================================

The Prefrontal Cortex manages executive function, long-term planning, and
hierarchical goal management. While the MotorCortex handles immediate step-by-step
tool execution (V10), the PFC holds a slowly decaying "Goal SDR" (an abstract
intent vector) that persists over hundreds or thousands of steps.

This Goal SDR provides a top-down bias to the ThalamicRouter and the MotorCortex,
ensuring the organism does not "lose the plot" during complex, multi-day
agentic workflows.

It uses the HTMSequenceMemory to predict expected state trajectories and evaluate
whether the current actions are moving the organism closer to the Goal SDR.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from .htm_predictor import HTMSequenceMemory


class PrefrontalCortex:
    """
    Manages long-horizon planning via a persistent Goal SDR and HTM sequence prediction.
    """
    def __init__(self, latent_dim: int, goal_decay: float = 0.99, seed: int = 1729):
        self.latent_dim = latent_dim
        # How slowly the Goal SDR decays (0.99 means it persists for ~100 steps)
        # Closer to 1.0 = longer-term planning
        self.goal_decay = float(goal_decay)
        self.rng = np.random.default_rng(seed)
        
        # The persistent goal vector
        self.goal_sdr = np.zeros(latent_dim, dtype=np.float32)
        
        # Sequence memory for trajectory prediction
        self.htm = HTMSequenceMemory(latent_dim=latent_dim, seed=seed)
        
        # Diagnostic tracking
        self._n_goals_set: int = 0
        self._last_anomaly_score: float = 0.0
        self._predicted_sdr = np.zeros(latent_dim, dtype=np.float32)

    def set_goal(self, abstract_intent: np.ndarray) -> None:
        """
        Inject a new high-level goal into the PFC. Overrides the current goal.
        """
        intent = np.asarray(abstract_intent, dtype=np.float32).reshape(-1)
        assert intent.shape[0] == self.latent_dim
        
        norm = np.linalg.norm(intent)
        if norm > 0:
            self.goal_sdr = intent / norm
            self._n_goals_set += 1

    def step(self, current_z_top: np.ndarray, learn: bool = True) -> Dict[str, Any]:
        """
        Tick the PFC.
        1. Process the current global state (z_top) through the HTM sequence memory.
        2. Calculate anomaly score (surprise).
        3. Decay the Goal SDR.
        4. Return top-down bias for the next step.
        """
        z_top = np.asarray(current_z_top, dtype=np.float32)
        
        # 1. Anomaly detection: how surprised are we by this state?
        self._last_anomaly_score = self.htm.get_anomaly_score(z_top, self._predicted_sdr)
        
        # 2. Update sequence memory and predict the NEXT state
        self._predicted_sdr = self.htm.step(z_top, learn=learn)
        
        # 3. Slowly decay the Goal SDR (unless it's actively reinforced by a new set_goal)
        self.goal_sdr *= self.goal_decay
        
        # If the goal has decayed to nothing, clear it
        if np.linalg.norm(self.goal_sdr) < 1e-4:
            self.goal_sdr = np.zeros(self.latent_dim, dtype=np.float32)
            
        return {
            "predicted_sdr": self._predicted_sdr,
            "anomaly_score": self._last_anomaly_score,
            "goal_bias": self.goal_sdr.copy(),
            "goal_intensity": float(np.linalg.norm(self.goal_sdr))
        }
        
    def get_top_down_bias(self) -> np.ndarray:
        """
        Returns the combined top-down bias (Goal SDR + HTM Prediction) to influence
        the thalamus and basal ganglia in the next step.
        """
        bias = self.goal_sdr + (0.5 * self._predicted_sdr)
        norm = np.linalg.norm(bias)
        if norm > 0:
            return (bias / norm).astype(np.float32)
        return np.zeros(self.latent_dim, dtype=np.float32)

    def stats(self) -> Dict[str, Any]:
        return {
            "n_goals_set": self._n_goals_set,
            "goal_intensity": float(np.linalg.norm(self.goal_sdr)),
            "last_anomaly_score": self._last_anomaly_score,
            "htm_sparsity": self.htm.sparsity
        }
