"""
ConsolidationScheduler — V20.

During forced rest (BUG-6 in V19: forced rest was already free ATP-wise
but did zero cognitive work), replay recent working-memory contents
through the TOP-1 routed expert at 3× LR. This is targeted skill
consolidation — cheap because forced rest already returns compute we're
paying for.

Audit item #9. Key design choices:

  * Route through TOP-1 (not the mixture) — replay is meant to strengthen
    the already-preferred expert, not blur across the mixture.
  * `reuse_routing=True` on the layer so we don't corrupt the EMA
    usage counters with replay-only activations.
  * 3× LR is high but bounded — the same lr clipping in
    PredictiveBlock.step still applies.
"""
from __future__ import annotations
import numpy as np


class ConsolidationScheduler:
    def __init__(self, working_memory, lr_multiplier: float = 3.0):
        self.wm = working_memory
        self.lr_multiplier = float(lr_multiplier)
        self.n_consolidated = 0

    def replay_during_rest(self, cortex, base_lr: float = 3e-4) -> int:
        """Run one round of replay. Returns the number of replay steps
        actually performed (0 if working memory is empty).
        """
        if self.wm.n_written() == 0:
            return 0

        replays = self.wm.replay(prioritise_by_error=True)
        lr = base_lr * self.lr_multiplier
        n = 0

        layers = getattr(cortex, "layers", None)
        if not layers:
            return 0

        for latent in replays:
            x = latent
            for layer in layers:
                # 1. Route ONCE per replay latent, pick TOP-1.
                idx, p = layer.router.route(x)
                # Zero out router.usage EMA drift caused by this replay
                # step — replays should not influence normal routing.
                # We do this by rolling back the EMA update we just
                # applied. Cheap: a single subtraction + rescale.
                a = layer.router.USAGE_DECAY_ALPHA
                # Undo the decay+increment applied by route().
                layer.router.usage[idx] -= a * p
                layer.router.usage /= max(1.0 - a, 1e-9)
                layer.router.lifetime_hits[idx] -= 1

                # 2. Now do a *targeted* top-1 step. We deliberately do
                # NOT set `_last_idx` here — replay must not corrupt the
                # cached routing decision from the last real forward.
                top1 = int(idx[int(np.argmax(p))])
                expert = layer.experts[top1]
                pager = getattr(layer, "pager", None)
                if pager is not None:
                    expert.wake(pager)
                s, _ = expert.step(x, top=None, lr=lr)
                x = s
            n += 1

        self.n_consolidated += n
        return n
