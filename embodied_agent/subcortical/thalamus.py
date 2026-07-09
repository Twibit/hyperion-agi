"""
Precision-weighted Thalamic Gate.

v2 attended over modalities by ||z|| × salience-prior.  v3 also weighs by
the *precision* (= inverse residual variance) of the cortical block that
last consumed each modality.  This is the standard predictive-coding
formulation: Bayesian-optimal cue combination weighs each signal by its
reliability, not its loudness.

Also includes a v3 dissociation guard: attention is forbidden from
concentrating above `dissociation_max_attn` on a single source, which
prevents pathological lock-in (the AI analogue of dissociative tunnel
vision).  When the cap fires, attention is renormalised across the rest.
"""
from __future__ import annotations
import numpy as np
from typing import Dict


class PrecisionThalamicGate:
    def __init__(self, latent_dim: int, dissociation_cap: float = 0.97):
        self.latent_dim = latent_dim
        self.salience: Dict[str, float] = {}
        self.precision: Dict[str, float] = {}     # learned per source
        self.dissociation_cap = dissociation_cap
        self.dissociation_strikes = 0

    def _softmax(self, x, temp):
        x = x / max(temp, 1e-6); x = x - x.max()
        e = np.exp(x); return e / e.sum()

    def set_precision(self, src: str, value: float):
        self.precision[src] = max(1e-3, float(value))

    def broadcast(self, candidates: Dict[str, np.ndarray],
                  acetylcholine: float = 0.5,
                  norepinephrine: float = 0.3) -> tuple[np.ndarray, Dict[str, float]]:
        if not candidates:
            return np.zeros(self.latent_dim, dtype=np.float32), {}

        names = list(candidates.keys())
        mat = np.stack([candidates[n] for n in names], axis=0)
        norms = np.linalg.norm(mat, axis=1)
        prior = np.array([self.salience.get(n, 1.0) for n in names], dtype=np.float32)
        prec  = np.array([self.precision.get(n, 1.0) for n in names], dtype=np.float32)
        scores = norms * prior * prec
        scores = scores * (1.0 + 2.0 * norepinephrine)
        temp = max(0.05, 1.0 - 0.7 * acetylcholine)
        w = self._softmax(scores, temp)

        # Dissociation guard.
        if w.max() > self.dissociation_cap:
            self.dissociation_strikes += 1
            # Cap and renormalise the rest.
            cap = self.dissociation_cap
            top = int(np.argmax(w))
            remainder = 1.0 - cap
            others = np.delete(w, top)
            if others.sum() > 0:
                others = others * (remainder / others.sum())
            w = np.insert(others, top, cap)

        broadcast = (w[:, None] * mat).sum(axis=0)
        weights = {n: float(wi) for n, wi in zip(names, w)}
        return broadcast.astype(np.float32), weights

    def reinforce(self, source, signal, lr=0.01):
        cur = self.salience.get(source, 1.0)
        self.salience[source] = float(np.clip(cur + lr * signal, 0.05, 5.0))
