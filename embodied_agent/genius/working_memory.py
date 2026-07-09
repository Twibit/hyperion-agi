"""
WorkingMemoryBuffer — V20.

A tiny fp16 ring buffer of recent latent states. Written by the main
loop after each cortex.forward(); read by ConsolidationScheduler
during forced rest.

Audit item #8 decision: raw latent vectors, fp16, 4 slots. VQ codebook
deferred to V21.

Storage footprint at latent_dim=1024, 4 slots, fp16:
  4 · 1024 · 2 B = 8 KB. Negligible.
"""
from __future__ import annotations
import numpy as np


class WorkingMemoryBuffer:
    def __init__(self, latent_dim: int, n_slots: int = 4):
        self.latent_dim = int(latent_dim)
        self.n_slots = int(n_slots)
        self._buf = np.zeros((self.n_slots, self.latent_dim), dtype=np.float16)
        # Parallel arrays: error at the time of writing, and a monotone
        # sequence number so consumers can tell newest from stale slots.
        self._err = np.zeros(self.n_slots, dtype=np.float32)
        self._seq = np.zeros(self.n_slots, dtype=np.int64)
        self._next = 0
        self._counter = 0

    def write(self, latent: np.ndarray, err: float = 0.0) -> None:
        v = latent.astype(np.float16, copy=False)
        if v.shape != (self.latent_dim,):
            # Best-effort resize — pad or truncate. This should not
            # normally happen; keeps the buffer robust to caller drift.
            if v.shape[0] < self.latent_dim:
                pad = np.zeros(self.latent_dim - v.shape[0], dtype=np.float16)
                v = np.concatenate([v, pad])
            else:
                v = v[: self.latent_dim]
        self._buf[self._next] = v
        self._err[self._next] = float(err)
        self._counter += 1
        self._seq[self._next] = self._counter
        self._next = (self._next + 1) % self.n_slots

    def replay(self, prioritise_by_error: bool = True) -> list[np.ndarray]:
        """Return the buffer contents as fp32 vectors, newest first.

        If `prioritise_by_error=True`, slots with higher stored error
        come first — that's the "surprising memory" replay strategy.
        """
        idxs = list(range(self.n_slots))
        if prioritise_by_error:
            idxs.sort(key=lambda i: (-self._err[i], -self._seq[i]))
        else:
            idxs.sort(key=lambda i: -self._seq[i])
        return [self._buf[i].astype(np.float32) for i in idxs
                if self._seq[i] > 0]

    def n_written(self) -> int:
        return int((self._seq > 0).sum())
