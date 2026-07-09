"""
Stigmergic Concept Graph — v6.
==============================

A semantic graph that grows like an ant colony.  Each cortical broadcast
deposits "pheromone" on the edge between the previous concept node and
the current one; unused edges evaporate; high-traffic edges thicken.

This emergent topology gives the organism a slowly-crystallising
non-parametric memory of recurring co-activation patterns — the
substrate for analogical reasoning and one-shot recall.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional, List


class StigmergicGraph:
    def __init__(self, latent_dim: int, max_nodes: int = 4096,
                 evap: float = 0.005, deposit: float = 1.0,
                 merge_thresh: float = 0.92, seed: int = 0):
        self.latent_dim = int(latent_dim)
        self.max_nodes = int(max_nodes)
        self.evap = float(evap)
        self.deposit = float(deposit)
        self.merge_thresh = float(merge_thresh)
        self.rng = np.random.default_rng(seed)
        # node centroids (allocated lazily)
        self._centroids = np.zeros((0, latent_dim), dtype=np.float32)
        self._counts = np.zeros(0, dtype=np.float32)
        # sparse edge dict {(u,v): weight}
        self._edges: Dict[Tuple[int, int], float] = {}
        self._last_node: Optional[int] = None
        self.evaporations = 0

    # ----- node management -----
    def _nearest(self, z: np.ndarray) -> Tuple[int, float]:
        if len(self._counts) == 0:
            return -1, -1.0
        cn = self._centroids
        sims = cn @ z / (np.linalg.norm(cn, axis=1) * np.linalg.norm(z) + 1e-9)
        i = int(np.argmax(sims))
        return i, float(sims[i])

    def _add_node(self, z: np.ndarray) -> int:
        if len(self._counts) >= self.max_nodes:
            # evict lowest-count node
            victim = int(np.argmin(self._counts))
            self._centroids[victim] = z
            self._counts[victim] = 1.0
            # drop its edges
            self._edges = {e: w for e, w in self._edges.items()
                           if victim not in e}
            return victim
        self._centroids = np.vstack([self._centroids, z[None, :]])
        self._counts = np.append(self._counts, 1.0)
        return len(self._counts) - 1

    # ----- ingestion -----
    def deposit_step(self, z: np.ndarray) -> int:
        """Snap `z` to the nearest concept node (or birth a new one),
        deposit pheromone on the edge from the last node, and evaporate."""
        z = z.astype(np.float32)
        i, sim = self._nearest(z)
        if i < 0 or sim < self.merge_thresh:
            i = self._add_node(z)
        else:
            # online centroid update
            self._counts[i] += 1.0
            alpha = 1.0 / self._counts[i]
            self._centroids[i] = (1.0 - alpha) * self._centroids[i] + alpha * z

        if self._last_node is not None and self._last_node != i:
            key = (self._last_node, i)
            self._edges[key] = self._edges.get(key, 0.0) + self.deposit

        # Evaporate (every step, on all edges, lightly).
        if self._edges:
            self.evaporations += 1
            decayed: Dict[Tuple[int, int], float] = {}
            for k, w in self._edges.items():
                nw = w * (1.0 - self.evap)
                if nw > 0.01:
                    decayed[k] = nw
            self._edges = decayed

        self._last_node = i
        return i

    def neighbours(self, node_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        outs = [(v, w) for (u, v), w in self._edges.items() if u == node_id]
        outs.sort(key=lambda t: -t[1])
        return outs[:top_k]

    def reset_trail(self) -> None:
        self._last_node = None

    def stats(self) -> dict:
        all_w = list(self._edges.values())
        max_w = max(all_w) if all_w else 0.0
        return dict(
            nodes=int(len(self._counts)),
            edges=len(self._edges),
            evaporations=self.evaporations,
            max_weight=max_w,
            mean_weight=float(np.mean(all_w)) if all_w else 0.0,
        )
