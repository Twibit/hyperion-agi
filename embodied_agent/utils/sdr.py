"""SDR utilities — identical to v2 (kept for compatibility)."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class SDR:
    active: np.ndarray
    size: int
    def to_dense(self) -> np.ndarray:
        out = np.zeros(self.size, dtype=np.float32); out[self.active] = 1.0; return out
    def overlap(self, other: "SDR") -> int:
        return int(np.intersect1d(self.active, other.active, assume_unique=True).size)


class SDREncoder:
    def __init__(self, latent_dim: int, sdr_size: int, k: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((sdr_size, latent_dim)).astype(np.float32) / np.sqrt(latent_dim)
        self.k = k; self.sdr_size = sdr_size

    def encode(self, z: np.ndarray) -> SDR:
        scores = self.W @ z if z.ndim == 1 else (z @ self.W.T).reshape(-1)
        active = np.argpartition(scores, -self.k)[-self.k:]
        active.sort()
        return SDR(active=active.astype(np.int64), size=self.sdr_size)
