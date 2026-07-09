"""
Global Workspace + Default Mode Network — v6.
==============================================

Baars-style Global Workspace Theory: only the most coherent broadcast
(the one with the highest cross-region agreement) becomes "conscious"
and gets distributed to every cortical region for the next step.

The DefaultModeNetwork runs offline (during sleep / idle) doing
spontaneous self-simulation by sampling from the recent workspace
distribution.  This is the AI analogue of mind-wandering and supports
goal-free creative exploration.
"""
from __future__ import annotations
import numpy as np
from collections import deque
from typing import Deque, List, Optional


class GlobalWorkspace:
    def __init__(self, latent_dim: int, history: int = 64,
                 broadcast_temp: float = 0.5):
        self.dim = int(latent_dim)
        self.hist: Deque[np.ndarray] = deque(maxlen=int(history))
        self.broadcast_temp = float(broadcast_temp)
        self.current: np.ndarray = np.zeros(self.dim, dtype=np.float32)
        self.coherence: float = 0.0
        self.broadcasts = 0

    def integrate(self, regional_states: List[np.ndarray],
                  precisions: Optional[List[float]] = None) -> np.ndarray:
        """Precision-weighted union of regional states; emits the
        single broadcast vector that all regions get to read next step."""
        if not regional_states:
            return self.current
        precisions = precisions or [1.0] * len(regional_states)
        M = np.stack(regional_states, axis=0).astype(np.float32)
        w = np.asarray(precisions, dtype=np.float32)
        w = w / max(1e-9, w.sum())
        broadcast = (w[:, None] * M).sum(axis=0)
        n = np.linalg.norm(broadcast) + 1e-9
        broadcast = broadcast / n
        # Coherence = avg cosine of each region with the consensus.
        sims = (M @ broadcast) / (np.linalg.norm(M, axis=1) + 1e-9)
        self.coherence = float(sims.mean())
        self.current = broadcast.astype(np.float32)
        self.hist.append(self.current.copy())
        self.broadcasts += 1
        return self.current

    def stats(self) -> dict:
        return dict(broadcasts=self.broadcasts,
                    coherence=self.coherence,
                    history=len(self.hist))


class DefaultModeNetwork:
    """Off-task mind wandering: samples mixtures of past workspace
    broadcasts to seed dream rollouts."""

    def __init__(self, workspace: GlobalWorkspace, seed: int = 0):
        self.ws = workspace
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros(self.ws.dim, dtype=np.float32)
        self.activations = 0

    def sample(self, n_components: int = 3) -> np.ndarray:
        if not self.ws.hist:
            return np.zeros(self.ws.dim, dtype=np.float32)
        k = min(int(n_components), len(self.ws.hist))
        idx = self.rng.choice(len(self.ws.hist), size=k, replace=False)
        # Dirichlet weights -> smooth mixture
        w = self.rng.dirichlet(np.ones(k))
        z = sum(wi * self.ws.hist[i] for wi, i in zip(w, idx))
        n = np.linalg.norm(z) + 1e-9
        z = z / n
        self.mean = 0.95 * self.mean + 0.05 * z
        self.activations += 1
        return z.astype(np.float32)

    def stats(self) -> dict:
        return dict(activations=self.activations,
                    mean_norm=float(np.linalg.norm(self.mean)))
