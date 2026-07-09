"""
ComplementPruner — C1q / C3 complement-mediated synaptic pruning.

The current `CellularMatrix.synaptic_pruning` only drops *cells* with low ATP.
Real developmental pruning tags individual *synapses* (weights) with complement
proteins (C1q binds weak / underused synapses, C3 marks them for engulfment by
microglia). This module mimics that cascade on any 2-D weight matrix.

Key biological invariants we mirror:

    1. Activity tag:   high-activity synapses get sialic-acid "don't-eat-me"
                       signal (immune from pruning).
    2. C1q binding:    proportional to inactivity (low |Δw| over a window).
    3. C3 conversion:  thresholded — only sufficiently tagged synapses are
                       opsonised, the rest survive.
    4. Microglia engulf: opsonised weights → 0 (structural pruning).

Returns the new mask so the caller can keep weights sparse at inference time.
"""
import numpy as np


class ComplementPruner:
    def __init__(self,
                 c1q_threshold: float = 0.02,
                 c3_threshold: float = 0.5,
                 activity_decay: float = 0.95,
                 protect_quantile: float = 0.9):
        """
        Args:
            c1q_threshold:    |w| below which a synapse starts collecting C1q
            c3_threshold:     opsonisation level required for engulfment
            activity_decay:   EMA decay of recent activity tag
            protect_quantile: synapses in the top quantile of activity are
                              fully protected ("don't-eat-me" signal)
        """
        self.c1q_thr = c1q_threshold
        self.c3_thr  = c3_threshold
        self.alpha   = activity_decay
        self.protect_q = protect_quantile

        self.activity = None
        self.c1q      = None

    def _ensure_state(self, W):
        if self.activity is None or self.activity.shape != W.shape:
            self.activity = np.zeros_like(W)
            self.c1q      = np.zeros_like(W)

    def observe(self, W, pre_post_corr=None):
        """
        Update activity tag. `pre_post_corr` (optional) is a same-shape matrix
        of recent pre·post correlation — if given, that's the cleanest tag.
        Otherwise we fall back to using |W| as a proxy.
        """
        self._ensure_state(W)
        signal = np.abs(pre_post_corr) if pre_post_corr is not None else np.abs(W)
        self.activity = self.alpha * self.activity + (1 - self.alpha) * signal

    def step(self, W):
        """Run one C1q→C3→engulf cycle on weight matrix W. Returns pruned W."""
        self._ensure_state(W)

        # protected set: top quantile of recent activity
        protect_cut = np.quantile(self.activity, self.protect_q)
        protected = self.activity >= protect_cut

        # C1q accumulates on weak / inactive synapses
        weak = (np.abs(W) < self.c1q_thr) & (~protected)
        self.c1q = np.where(weak, self.c1q + 0.1, self.c1q * 0.9)

        # C3 cascade: c1q above threshold → opsonise and prune
        engulf = (self.c1q > self.c3_thr) & (~protected)
        new_W = np.where(engulf, 0.0, W)
        # reset c1q where pruning happened
        self.c1q = np.where(engulf, 0.0, self.c1q)
        return new_W, engulf

    def sparsity(self, W):
        return float(np.mean(W == 0.0))
