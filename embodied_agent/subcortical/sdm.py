"""
Sparse Distributed Memory (Kanerva 1988) — v6.
==============================================

A content-addressable subconscious store.  Each "hard location" has a
random address; writes increment all addresses within Hamming radius;
reads sum all addresses within radius.  Gives O(log N) recall on a noisy
prompt — the AI analogue of the cocktail-party effect.
"""
from __future__ import annotations
import numpy as np


class SparseDistributedMemory:
    def __init__(self, dim: int, n_locations: int = 4096,
                 radius: float = 0.35, seed: int = 0):
        self.dim = int(dim)
        self.n = int(n_locations)
        self.radius = float(radius)
        rng = np.random.default_rng(seed)
        # Bipolar address vectors
        self._addr = rng.choice([-1.0, 1.0], size=(self.n, self.dim)).astype(np.float32)
        # Content counters
        self._data = np.zeros((self.n, self.dim), dtype=np.float32)
        self.writes = 0
        self.reads = 0

    def _active(self, query: np.ndarray) -> np.ndarray:
        # Normalised cosine similarity
        q = query.astype(np.float32)
        qn = np.linalg.norm(q) + 1e-9
        sims = (self._addr @ q) / (np.linalg.norm(self._addr, axis=1) * qn + 1e-9)
        # 1 - sim is a Hamming-like distance for bipolar vectors
        return (1.0 - sims) <= self.radius

    def write(self, addr_hint: np.ndarray, content: np.ndarray) -> int:
        mask = self._active(addr_hint)
        if not mask.any():
            # ensure at least 1 location activates: nearest
            sims = self._addr @ addr_hint
            mask = np.zeros(self.n, dtype=bool)
            mask[int(np.argmax(sims))] = True
        self._data[mask] += content.astype(np.float32)
        self.writes += 1
        return int(mask.sum())

    def read(self, addr_hint: np.ndarray) -> np.ndarray:
        mask = self._active(addr_hint)
        if not mask.any():
            sims = self._addr @ addr_hint
            mask = np.zeros(self.n, dtype=bool)
            mask[int(np.argmax(sims))] = True
        out = self._data[mask].mean(axis=0)
        self.reads += 1
        n = np.linalg.norm(out)
        if n > 1e-9:
            out = out / n
        return out.astype(np.float32)

    def stats(self) -> dict:
        norms = np.linalg.norm(self._data, axis=1)
        return dict(
            locations=self.n, radius=self.radius,
            writes=self.writes, reads=self.reads,
            active_locations=int((norms > 1e-3).sum()),
            mean_norm=float(norms.mean()),
        )
