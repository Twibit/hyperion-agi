"""
P-FIT Cross-Region Highways — Gap 2: Genius Brain Architecture.
================================================================

Parieto-Frontal Integration Theory (Jung & Haier, 2007):
Intelligence correlates with the SPEED and BANDWIDTH of white matter
tracts connecting frontal, parietal, temporal, and occipital regions.

This module adds lateral skip-connections (arcuate fasciculi) between
non-adjacent cortex layers, creating "information highways" that allow
deep layers to directly influence early perception and vice versa.

In a genius brain, these highways are thicker (more myelinated) and
transmit signals faster than in an average brain.
"""
from __future__ import annotations
import numpy as np


class PFITHighway:
    """Cross-region lateral connections between non-adjacent cortex layers.

    Instead of signals flowing strictly bottom-up then top-down through
    the cortex stack, a PFITHighway creates direct lanes between
    arbitrary layer pairs. Each highway has a learnable gain (simulating
    myelination thickness) and a projection matrix.
    """

    def __init__(self, latent_dim: int, n_layers: int,
                 highway_pairs: list[tuple[int, int]] | None = None,
                 seed: int = 0):
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        rng = np.random.default_rng(seed + 7777)

        # Default highways: connect layer 0 ↔ last, and 1 ↔ second-to-last
        if highway_pairs is None:
            highway_pairs = []
            for i in range(min(3, n_layers // 2)):
                highway_pairs.append((i, n_layers - 1 - i))

        self.highways = []
        for src, dst in highway_pairs:
            W = rng.standard_normal((latent_dim, latent_dim)).astype(np.float32) / np.sqrt(latent_dim)
            self.highways.append({
                "src": src, "dst": dst,
                "W": W,
                "myelination": np.float32(0.1),  # Starts low, grows with use
            })

    def transmit(self, layer_states: list[np.ndarray],
                 lr: float = 1e-4) -> list[np.ndarray]:
        """Inject highway signals into target layer states.

        For each highway, project the source layer's state through the
        tract and add it (scaled by myelination) to the destination.
        Hebbian learning strengthens frequently-used tracts.
        """
        augmented = [s.copy() for s in layer_states]

        for hw in self.highways:
            src_idx, dst_idx = hw["src"], hw["dst"]
            if src_idx >= len(layer_states) or dst_idx >= len(layer_states):
                continue

            src_state = layer_states[src_idx]
            projected = hw["W"] @ src_state
            # Scale by myelination (thicker = stronger signal)
            augmented[dst_idx] += hw["myelination"] * projected

            # Hebbian myelination: if both ends are active, strengthen tract
            src_energy = float(np.mean(src_state ** 2))
            dst_energy = float(np.mean(layer_states[dst_idx] ** 2))
            coactivation = src_energy * dst_energy

            # Myelination grows slowly with use, caps at 1.0
            hw["myelination"] = min(1.0, hw["myelination"] + lr * coactivation)

            # Hebbian update to the projection matrix
            hw["W"] += lr * hw["myelination"] * np.outer(
                layer_states[dst_idx], src_state
            ).astype(np.float32)

        return augmented

    def myelination_report(self) -> list[dict]:
        """Return myelination levels for diagnostics."""
        return [
            {"src": hw["src"], "dst": hw["dst"],
             "myelination": float(hw["myelination"])}
            for hw in self.highways
        ]
