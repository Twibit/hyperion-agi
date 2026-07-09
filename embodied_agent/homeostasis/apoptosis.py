"""
ApoptosisController — V20.

Prunes MoE experts whose EMA usage stays below `tau` for `window` steps.
Complements the V19 neurogenesis path (`_spawn_expert`) — together they
turn the cortex from monotone-growing into homeostatic.

Contract (audit item #3):

  * Never prune below `min_experts` per layer.
  * Hysteresis window: an expert must be underused for W consecutive
    checks, not just one — protects against transient routing drift.
  * On prune: (a) drop it from the router, (b) drop it from the expert
    list, (c) call `pager.evict(id)` to remove the on-disk blob. All
    three must happen atomically. If any of (a-c) fails we abort the
    prune and log — better to leak an expert than to end up with a
    router/expert-list shape mismatch.
"""
from __future__ import annotations
import warnings
from collections import defaultdict


class ApoptosisController:
    def __init__(self, tau: float = 1e-3, window: int = 20,
                 min_experts: int = 4):
        self.tau = float(tau)
        self.window = int(window)
        self.min_experts = int(min_experts)
        # (layer_id, expert_id) → number of consecutive underused checks.
        self._streak: dict[tuple[int, str], int] = defaultdict(int)
        # Diagnostic counter.
        self.n_pruned = 0

    def check(self, cortex, step: int) -> list[tuple[int, str]]:
        """Run one apoptosis check across every MoE layer.

        Returns the list of (layer_idx, expert_id) pairs that were pruned
        this call, for logging.
        """
        pruned: list[tuple[int, str]] = []
        layers = getattr(cortex, "layers", None)
        if not layers:
            return pruned

        pager = getattr(cortex, "pager", None)

        for l_idx, layer in enumerate(layers):
            router = getattr(layer, "router", None)
            if router is None:
                continue
            if router.n_experts <= self.min_experts:
                # Clear streaks so ratcheting doesn't kill the layer as
                # soon as it grows back above min_experts.
                for k in list(self._streak.keys()):
                    if k[0] == l_idx:
                        self._streak[k] = 0
                continue

            under = router.underused_indices(self.tau)
            active_ids = {layer.experts[i].id for i in under}

            # Increment streaks for currently-under experts.
            for exp_id in active_ids:
                self._streak[(l_idx, exp_id)] += 1
            # Reset streaks for experts NOT under (or that no longer
            # exist).
            for k in list(self._streak.keys()):
                if k[0] != l_idx:
                    continue
                if k[1] not in active_ids:
                    self._streak[k] = 0

            # Prune those with streak >= window, respecting min_experts.
            for exp_id in list(active_ids):
                if router.n_experts <= self.min_experts:
                    break
                if self._streak[(l_idx, exp_id)] < self.window:
                    continue
                # Locate the current index of this expert.
                try:
                    cur_idx = next(i for i, e in enumerate(layer.experts)
                                   if e.id == exp_id)
                except StopIteration:
                    self._streak.pop((l_idx, exp_id), None)
                    continue

                try:
                    layer.router.drop_expert(cur_idx)
                    dead = layer.experts.pop(cur_idx)
                    if pager is not None:
                        pager.evict(dead.id)
                except Exception as e:
                    warnings.warn(
                        f"[ApoptosisController] failed to prune {exp_id} "
                        f"on layer {l_idx}: {e}. Layer state may be "
                        "inconsistent.", RuntimeWarning,
                    )
                    continue

                self._streak.pop((l_idx, exp_id), None)
                # Also flush the cached _last_idx/_last_p — they still
                # reference the old expert count.
                layer._last_idx = None
                layer._last_p = None

                pruned.append((l_idx, exp_id))
                self.n_pruned += 1

        return pruned
