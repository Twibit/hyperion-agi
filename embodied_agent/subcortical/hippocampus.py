"""
Hippocampus — episodic memory with prioritized replay, content-addressable
recall, and the v3 trauma-quarantine flag.

Episodes marked `quarantine=True` by the Nociceptor are STILL stored
(the organism needs to remember the event existed), but they are excluded
from prioritized replay until/unless an explicit therapy routine
(`replay_in_therapy_mode`) is called.  This implements graduated
re-exposure under a low-cortisol baseline — analogous to the empirical
treatment recipe for PTSD.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Episode:
    z: np.ndarray
    a: int
    r: float
    z_next: np.ndarray
    mod: dict = field(default_factory=dict)
    priority: float = 1.0
    tag: str = ""
    quarantine: bool = False          # v3 trauma flag
    affect: dict = field(default_factory=dict)   # v3 affect snapshot


class Hippocampus:
    def __init__(self, capacity: int, latent_dim: int,
                 priority_alpha: float = 0.6, seed: int = 0):
        self.capacity = capacity
        self.alpha = priority_alpha
        self.rng = np.random.default_rng(seed)
        self.episodes: List[Optional[Episode]] = []
        self._idx = 0
        # parallel array of L2-normalised z's for fast cosine recall
        self._key = np.zeros((0, latent_dim), dtype=np.float32)
        self.latent_dim = latent_dim

    def __len__(self): return sum(1 for e in self.episodes if e is not None)

    def store(self, ep: Episode):
        kz = ep.z / (np.linalg.norm(ep.z) + 1e-8)
        if len(self.episodes) < self.capacity:
            self.episodes.append(ep)
            self._key = np.vstack([self._key, kz[None, :]])
        else:
            self.episodes[self._idx] = ep
            self._key[self._idx] = kz
            self._idx = (self._idx + 1) % self.capacity

    def recall(self, query_z: np.ndarray, k: int = 5) -> List[Episode]:
        if not self.episodes: return []
        q = query_z / (np.linalg.norm(query_z) + 1e-8)
        sims = self._key @ q
        # mask quarantined (don't surface trauma on free recall)
        for i, e in enumerate(self.episodes):
            if e is None or e.quarantine: sims[i] = -1.0
        k = min(k, len(sims))
        idx = np.argpartition(sims, -k)[-k:]
        return [self.episodes[int(i)] for i in idx if self.episodes[int(i)] is not None]

    def _priorities(self, allow_trauma: bool = False) -> np.ndarray:
        ps = np.array([
            (e.priority if (e is not None and (allow_trauma or not e.quarantine)) else 0.0)
            for e in self.episodes
        ], dtype=np.float32)
        ps = ps ** self.alpha
        s = ps.sum()
        return ps / s if s > 0 else None

    def sample_replay(self, batch: int, allow_trauma: bool = False) -> List[Episode]:
        ps = self._priorities(allow_trauma=allow_trauma)
        if ps is None: return []
        valid_count = np.count_nonzero(ps)
        if valid_count == 0: return []
        idx = self.rng.choice(len(self.episodes), size=min(batch, valid_count),
                              p=ps, replace=False)
        return [self.episodes[int(i)] for i in idx]

    def replay_in_therapy_mode(self, batch: int) -> List[Episode]:
        """Graduated re-exposure: include quarantined episodes, lower priority."""
        return self.sample_replay(batch, allow_trauma=True)

    def update_priorities(self, batch: List[Episode], tds: List[float]):
        for ep, td in zip(batch, tds):
            ep.priority = abs(td) + 1e-3
