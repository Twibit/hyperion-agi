"""
Working Memory Bottleneck (v9 — Aurora-Cogitans).
=================================================

LLMs use enormous "contexts" (32 k → 1 M tokens) that they treat as if
they were working memory.  But biological working memory is famously
*small* — Miller (1956) put its capacity at 7 ± 2 chunks; Cowan (2001)
revised it downward to 4 ± 1.  This is not an implementation defect: the
bottleneck is *the mechanism* that forces chunking, abstraction, and
compositional generalisation.  Anything you cannot compress into ~4
slots is expelled — and this pressure is exactly what makes symbolic
composition happen.

v8's ``GlobalWorkspace`` has a history of 64 slots and integrates them
by weighted sum.  That is more like a *cache* than a bottleneck — it has
no capacity pressure, no eviction cost, no compression incentive.

``WorkingMemory`` supplies the missing bottleneck.

Design
------
A tiny, capacity-limited slot store:

  * ``capacity`` (default 4) — Cowan's core estimate for adults.
  * Each slot stores: latent vector, priority, age, tag.
  * ``write(z, priority, tag)`` uses *replace-worst-slot*: if all slots
    are occupied and the new item's priority exceeds min-priority in
    memory, the worst slot is evicted.
  * ``read()`` returns the current occupied slots as (K, d) matrix.
  * ``decay()`` reduces every priority every step (age × λ).

The workspace's output is now conditioned on the WM contents via an
*attention-over-slots* mechanism (dot-product, softmax, no learned
params — pure functional).

Why replace-worst instead of oldest-first?
------------------------------------------
FIFO working-memory models (e.g. simple rings) don't reproduce the
biological finding that *rehearsal keeps items in WM*.  Replace-worst
+ priority decay reproduces both:

  * unrehearsed items age → fall in priority → are evicted
  * highly-relevant items (high td_err, high novelty) survive

which is precisely how human WM behaves under dual-task load.

Integration with the rest of Aurora
-----------------------------------
* Priority is set from: |td_err| + novelty_bonus + workspace_gain.  This
  matches the biological finding that reward-prediction errors, novelty,
  and top-down attention all promote WM entry.
* Contents feed into the workspace's ``integrate`` as high-precision
  entries, biasing conscious broadcast toward WM contents.
* When the epistemic gate emits "hedge" or "refuse", the WM state is
  logged into the self-model's belief-revision log (v9 module 4).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class WMSlot:
    z: np.ndarray                # (latent_dim,) unit-norm
    priority: float
    age: int
    tag: str = ""


class WorkingMemory:
    def __init__(self, latent_dim: int, capacity: int = 4,
                 decay: float = 0.9, min_admit_priority: float = 0.05):
        assert 1 <= capacity <= 12
        self.latent_dim = int(latent_dim)
        self.capacity   = int(capacity)
        self.decay      = float(decay)
        self.min_admit_priority = float(min_admit_priority)

        self.slots: List[WMSlot] = []
        self._n_writes = 0
        self._n_evictions = 0
        self._n_rejects = 0

    # ------------------------------------------------------------------
    def _normalise(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(z)) + 1e-9
        return (z / n).astype(np.float32)

    def write(self, z: np.ndarray, priority: float,
              tag: str = "") -> bool:
        """Try to admit z into WM.  Returns True on admit, False on reject.

        * If a free slot exists → occupy it.
        * Else if priority > min priority in WM → evict worst, admit.
        * Else → reject.
        """
        if priority < self.min_admit_priority:
            self._n_rejects += 1
            return False
        z = self._normalise(z)
        if len(self.slots) < self.capacity:
            self.slots.append(WMSlot(z=z, priority=float(priority),
                                     age=0, tag=tag))
            self._n_writes += 1
            return True
        # find lowest-priority slot
        idx = int(np.argmin([s.priority for s in self.slots]))
        if priority > self.slots[idx].priority:
            self.slots[idx] = WMSlot(z=z, priority=float(priority),
                                     age=0, tag=tag)
            self._n_evictions += 1
            self._n_writes += 1
            return True
        self._n_rejects += 1
        return False

    def read(self) -> np.ndarray:
        """Return (n_occupied, latent_dim) matrix of current WM contents."""
        if not self.slots:
            return np.zeros((0, self.latent_dim), dtype=np.float32)
        return np.stack([s.z for s in self.slots]).astype(np.float32)

    def priorities(self) -> np.ndarray:
        if not self.slots:
            return np.zeros((0,), dtype=np.float32)
        return np.array([s.priority for s in self.slots], dtype=np.float32)

    def tick(self) -> None:
        """Age every slot and decay its priority."""
        keep: List[WMSlot] = []
        for s in self.slots:
            s.age += 1
            s.priority *= self.decay
            if s.priority > self.min_admit_priority * 0.25:
                keep.append(s)
        self.slots = keep

    # ------------------------------------------------------------------
    def attend(self, query: np.ndarray) -> np.ndarray:
        """Attention-over-slots read-out.

        Given a query vector, return the priority-weighted, softmax-mixed
        combination of the currently-held slot vectors.  Returns a zero
        vector when WM is empty.  No learned params.
        """
        M = self.read()
        if M.shape[0] == 0:
            return np.zeros(self.latent_dim, dtype=np.float32)
        q = self._normalise(query)
        logits = M @ q                                   # (K,)
        pri = self.priorities()
        logits = logits + 0.5 * np.log(np.maximum(pri, 1e-9))
        logits -= logits.max()
        w = np.exp(logits)
        w = w / (w.sum() + 1e-9)
        out = (w[:, None] * M).sum(0).astype(np.float32)
        n = float(np.linalg.norm(out)) + 1e-9
        return (out / n).astype(np.float32)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return dict(
            capacity=self.capacity,
            occupied=len(self.slots),
            writes=self._n_writes,
            evictions=self._n_evictions,
            rejects=self._n_rejects,
            mean_priority=(float(np.mean([s.priority for s in self.slots]))
                           if self.slots else 0.0),
            mean_age=(float(np.mean([s.age for s in self.slots]))
                      if self.slots else 0.0),
            tags=[s.tag for s in self.slots],
        )
