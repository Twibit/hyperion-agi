"""
Synaptic Pruning Engine — Gap 8: Genius Brain Architecture.
=============================================================

Gifted brains undergo VIGOROUS synaptic pruning during maturation.
Connections that are rarely used or carry weak signals are eliminated,
leaving behind fewer but STRONGER synapses. This creates the "neural
efficiency" observed in genius individuals.

This module runs during sleep consolidation and prunes weak synapses
from the cortex weight matrices.
"""
from __future__ import annotations
import numpy as np


class SynapticPruner:
    """Eliminates weak synapses during sleep consolidation.

    Mimics the aggressive cortical thinning observed in gifted adolescents.
    Weak connections (weight magnitude below threshold) are zeroed out,
    creating sparser, more efficient weight matrices.
    """

    def __init__(self,
                 prune_percentile: float = 10.0,
                 min_developmental_age: int = 10000,
                 prune_interval: int = 5000,
                 max_prune_fraction: float = 0.15):
        """
        Args:
            prune_percentile: Bottom X% of weights by magnitude get pruned.
            min_developmental_age: Don't prune until this many steps (let it grow first).
            prune_interval: Prune every N steps during sleep.
            max_prune_fraction: Safety cap — never prune more than this fraction per sweep.
        """
        self.prune_percentile = prune_percentile
        self.min_age = min_developmental_age
        self.prune_interval = prune_interval
        self.max_prune_fraction = max_prune_fraction
        self._last_prune_step = 0
        self._total_pruned = 0
        self._total_synapses = 0

    def should_prune(self, current_step: int) -> bool:
        """Check if pruning should occur at this step."""
        if current_step < self.min_age:
            return False
        return (current_step - self._last_prune_step) >= self.prune_interval

    def prune_weights(self, W: np.ndarray, current_step: int) -> dict:
        """Prune the weakest synapses from a weight matrix.

        Args:
            W: The weight matrix to prune (modified in-place).
            current_step: Current developmental age.

        Returns:
            Dict with pruning statistics.
        """
        abs_W = np.abs(W)
        total = W.size
        self._total_synapses = total

        # Find the threshold for the bottom percentile
        threshold = np.percentile(abs_W, self.prune_percentile)

        # Identify synapses below threshold
        mask = abs_W < threshold
        n_candidates = int(np.sum(mask))

        # Safety: cap the number pruned per sweep
        max_allowed = int(total * self.max_prune_fraction)
        if n_candidates > max_allowed:
            # Only prune the weakest of the weak
            flat_indices = np.where(mask.ravel())[0]
            flat_magnitudes = abs_W.ravel()[flat_indices]
            keep_count = n_candidates - max_allowed
            # Sort and keep the "strongest of the weak"
            sorted_idx = np.argsort(flat_magnitudes)
            prune_idx = flat_indices[sorted_idx[:max_allowed]]
            mask = np.zeros_like(W, dtype=bool)
            mask.ravel()[prune_idx] = True

        # Execute the prune
        n_pruned = int(np.sum(mask))
        W[mask] = 0.0

        self._total_pruned += n_pruned
        self._last_prune_step = current_step

        return {
            "pruned": n_pruned,
            "total_synapses": total,
            "prune_fraction": n_pruned / max(1, total),
            "threshold": float(threshold),
            "cumulative_pruned": self._total_pruned,
        }

    def prune_cortex_stack(self, cortex, current_step: int) -> list[dict]:
        """Prune all layers in a CortexStack or MoECortexStack."""
        results = []
        layers = cortex.layers if hasattr(cortex, 'layers') else []

        for i, layer in enumerate(layers):
            # Handle MoE layers (which have multiple experts)
            if hasattr(layer, 'experts'):
                for j, expert in enumerate(layer.experts):
                    r1 = self.prune_weights(expert.W_gen, current_step)
                    r2 = self.prune_weights(expert.W_rec, current_step)
                    results.append({
                        "layer": i, "expert": j,
                        "W_gen": r1, "W_rec": r2,
                    })
            else:
                # Dense CortexStack layer
                r1 = self.prune_weights(layer.W_gen, current_step)
                r2 = self.prune_weights(layer.W_rec, current_step)
                results.append({
                    "layer": i,
                    "W_gen": r1, "W_rec": r2,
                })

        return results

    def snapshot(self) -> dict:
        return {
            "total_pruned": self._total_pruned,
            "total_synapses": self._total_synapses,
            "sparsity": self._total_pruned / max(1, self._total_synapses),
            "last_prune_step": self._last_prune_step,
        }
