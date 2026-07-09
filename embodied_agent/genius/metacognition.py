"""
MetacognitiveMonitor — V20.

Replaces V19's *global* neurogenesis pass (which iterates every layer
and spawns an expert on all of them whenever error trips) with a
*targeted* pass that spawns on exactly one (layer, expert) location:
the one whose `precision × error` is highest.

The intuition (audit item #10): high precision × high error = "the
model is confidently wrong on this layer". That's the correct place to
allocate a new expert — everywhere else the existing capacity is
sufficient.

Uses `PredictiveBlock.precision` and `PredictiveBlock.last_err` — no
new confidence scalar (audit item #10 explicitly rejected a new one).
"""
from __future__ import annotations
import warnings
import numpy as np

from ..cortex.predictive_block import PredictiveBlock


class MetacognitiveMonitor:
    def __init__(self,
                 dissonance_threshold: float = 0.5,
                 max_experts_per_layer: int = 128):
        self.tau = float(dissonance_threshold)
        self.max_per_layer = int(max_experts_per_layer)
        self.n_targeted_spawns = 0

    def _worst_layer(self, cortex) -> tuple[int, float] | None:
        """(layer_idx, precision×err score) of the most dissonant layer."""
        best = None
        best_score = -1.0
        for i, layer in enumerate(getattr(cortex, "layers", [])):
            # Layer-level dissonance = max over its experts of
            # precision × err. Cached scalars — cheap.
            for exp in layer.experts:
                score = exp.precision * exp.last_err
                if score > best_score:
                    best_score = score
                    best = i
        if best is None or best_score < self.tau:
            return None
        return best, best_score

    def maybe_spawn(self, cortex, cfg, rng, step: int) -> bool:
        """If any layer's dissonance exceeds tau, spawn ONE expert
        there. Returns True iff a spawn happened.
        """
        target = self._worst_layer(cortex)
        if target is None:
            return False
        l_idx, score = target

        layer = cortex.layers[l_idx]
        if len(layer.experts) >= self.max_per_layer:
            return False

        # Same atomic-expand contract as v19_autopoiesis._spawn_expert,
        # but scoped to a single layer.
        n_before = len(layer.experts)
        latent_dim = getattr(cfg, "latent_dim", layer.router.W.shape[1])
        new_expert = PredictiveBlock(latent_dim, latent_dim,
                                     seed=step * 100 + l_idx + n_before)
        new_expert.id = (f"expert_meta{step}_L{l_idx}_e{n_before}")

        old_W = layer.router.W
        scale = np.float32(1.0 / np.sqrt(latent_dim))
        new_W_row = (rng.standard_normal((1, latent_dim)).astype(np.float32)
                     * scale)
        layer.router.W = np.vstack([old_W, new_W_row])
        # Bootstrap the newborn EMA usage at the layer mean so it isn't
        # instantly a prune target.
        mean_usage = float(layer.router.usage.mean()) if layer.router.n_experts else 0.0
        layer.router.usage = np.append(layer.router.usage, mean_usage)
        layer.router.lifetime_hits = np.append(layer.router.lifetime_hits, 0)
        layer.router.n_experts = old_W.shape[0] + 1
        layer.experts.append(new_expert)

        pager = getattr(layer, "pager", None)
        if pager is not None:
            pager.touch(new_expert.id)
        else:
            warnings.warn(
                f"[MetacognitiveMonitor] layer {l_idx} has no pager; "
                f"new expert {new_expert.id} unpaged.",
                RuntimeWarning,
            )

        # Flush cached routing so the next forward re-routes with the
        # new expert count.
        layer._last_idx = None
        layer._last_p = None
        self.n_targeted_spawns += 1
        return True
