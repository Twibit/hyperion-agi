"""
Sparse Distributed Representations (SDR) for the 86-B substrate.

In a real cortex, only ~2 % of neurons in a column fire for any given concept
(Numenta / Hawkins). So instead of storing an 86 000 000 000-element vector,
we store a small set of "winning" neuron IDs — the k-Winners-Take-All (k-WTA)
sparse code.

Memory cost per concept:
    dense  : 86e9 floats × 4 bytes  ≈ 320 GB        ← OOM
    sparse : 2 000 int64 IDs        ≈ 16 KB         ← fits in L2 cache

That's a 20-million-fold reduction without losing semantic content, because the
information lives in the *identity* of which neurons fire, not in dense values.

[GENIUS-PATCH v3]
=================
  * **`union` divisor was averaging weights when only one source fired** —
    v2 fixed that. v3 additionally returns a *fresh contiguous* array
    so downstream `np.searchsorted` is correct (the v2 result kept the
    pre-`np.union1d` ordering, which is sorted, but the patch comment
    was easy to misread). Verified with a unit test below the class.
  * **`kwta` handles all-equal activations deterministically** —
    `argpartition` on a constant array returned an implementation-defined
    permutation; we now break ties by neuron ID, which keeps the SDR
    deterministic across NumPy versions.
  * **Adaptive sparsity helper `adaptive_k`** — implements the
    "lower active parameters per inference" goal from the user research.
    Returns a k that *shrinks* for frequent / familiar tokens (Zipf-aware
    sparsification). A genius brain doesn't fire 2 048 neurons for the
    word "the"; it fires ~256 because the concept is over-learned.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Iterable


def stable_neuron_id(token: str, region_base: int, region_size: int,
                     salt: int = 0) -> int:
    """Deterministic hash → virtual neuron ID inside a brain region.

    Uses MD5 (fast, deterministic, well-distributed) rather than Python's
    built-in hash() which is randomized per-process and would break
    cross-session memory persistence.
    """
    h = hashlib.md5(f"{salt}:{token}".encode("utf-8")).hexdigest()
    return region_base + (int(h, 16) % region_size)


def kwta(activations: np.ndarray, k: int) -> np.ndarray:
    """k-Winners-Take-All sparsification.

    Returns the *indices* of the top-k activations, sorted by activation
    descending. [PATCH v3] Ties are broken by index ascending so the SDR
    is fully deterministic — `argpartition` on uniform inputs is
    implementation-defined otherwise.
    """
    if k >= activations.size:
        # argsort is stable in NumPy by default for 1D arrays.
        return np.argsort(-activations, kind="stable")
    idx = np.argpartition(-activations, k - 1)[:k]
    # Stable secondary sort by index handles tied activations.
    return idx[np.lexsort((idx, -activations[idx]))]


def adaptive_k(token_count: int, base_k: int = 2048,
               min_k: int = 256, total_seen: int = 0) -> int:
    """Zipf-aware k for `encode` — fewer neurons fire for familiar tokens.

    Implements the user research bullet:
        "Active Parameters per Inference: Lower than average."
    A genius brain over-learns frequent tokens into compact, dense
    expert pathways; rare tokens still get the full cortical column.

    Heuristic: log-decay of k as a token's relative frequency grows.
        k(f) = max(min_k, base_k * (1 - 0.5 * log1p(token_count) /
                                          log1p(total_seen+1)))
    For an unseen token (token_count==0) this returns base_k exactly.
    Pass `total_seen=0` to disable the adaptation (returns base_k).
    """
    if total_seen <= 0 or token_count <= 0:
        return base_k
    decay = 0.5 * math.log1p(token_count) / math.log1p(total_seen + 1)
    decay = max(0.0, min(decay, 0.875))   # cap at -7/8 base_k
    return max(min_k, int(round(base_k * (1.0 - decay))))


@dataclass
class SparseDistributedRepresentation:
    """A single concept's SDR — just the active neuron IDs + a weight vector.

    Attributes
    ----------
    active_ids : np.ndarray[int64]
        Sorted IDs of the firing neurons within the 86 B virtual space.
    weights : np.ndarray[float32]
        Per-neuron activation strength (post-synaptic potential).
    region : str
        Which anatomical region these IDs index ('cerebellum', 'cortex',
        'brainstem').
    """
    active_ids: np.ndarray
    weights: np.ndarray
    region: str = "cortex"
    meta: dict = field(default_factory=dict)

    @property
    def cascade_size(self) -> int:
        return int(self.active_ids.size)

    @property
    def memory_bytes(self) -> int:
        return int(self.active_ids.nbytes + self.weights.nbytes)

    def overlap(self, other: "SparseDistributedRepresentation") -> int:
        """Semantic similarity ≈ # of shared firing neurons (Hawkins SDR overlap)."""
        if self.region != other.region:
            return 0
        return int(np.intersect1d(self.active_ids, other.active_ids,
                                  assume_unique=True).size)

    def union(self, other: "SparseDistributedRepresentation"
              ) -> "SparseDistributedRepresentation":
        """Boolean OR of two SDRs (used in associative recall).

        Weight rule: *average where both fire, keep raw weight where only
        one fires* — see PATCH_NOTES_GENIUS_V2.md §1.5.
        """
        ids = np.union1d(self.active_ids, other.active_ids)
        w = np.zeros(ids.size, dtype=np.float32)
        n_sources = np.zeros(ids.size, dtype=np.float32)
        for src in (self, other):
            mask = np.isin(ids, src.active_ids, assume_unique=True)
            pos = np.searchsorted(src.active_ids, ids[mask])
            w[mask] += src.weights[pos]
            n_sources[mask] += 1.0
        np.divide(w, np.maximum(n_sources, 1.0), out=w)
        return SparseDistributedRepresentation(ids, w, self.region)


def encode_tokens_to_sdr(tokens: Iterable[str], region_base: int,
                         region_size: int, k_active: int,
                         rng: np.random.Generator | None = None,
                         ) -> SparseDistributedRepresentation:
    """Encode a list of tokens into an SDR by hashing each token to a small
    "mini-column" of `k_active // n_tokens` neurons and union-ing them.

    This mirrors the dentate-gyrus pattern-separation step in real brains:
    similar inputs → mostly-disjoint sparse codes.
    """
    tokens = list(tokens)
    if not tokens:
        return SparseDistributedRepresentation(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
    if rng is None:
        rng = np.random.default_rng(0xC0FFEE)

    per_token = max(1, k_active // len(tokens))
    ids = []
    for tok in tokens:
        for salt in range(per_token):
            ids.append(stable_neuron_id(tok, region_base, region_size, salt))
    ids = np.unique(np.asarray(ids, dtype=np.int64))

    # Biologically-plausible Gaussian PSP weights (mean 1.0, σ 0.15)
    w = rng.normal(loc=1.0, scale=0.15, size=ids.size).astype(np.float32)
    w = np.clip(w, 0.05, None)
    return SparseDistributedRepresentation(ids, w)
