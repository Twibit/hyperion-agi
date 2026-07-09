"""
Engineered safety guards.

These are not a *philosophical* fix for AI safety; they are a small set
of *operational* circuit-breakers that prevent specific, well-documented
failure modes of brain-inspired learners:

  * Rumination: the same painful episode keeps winning prioritized
    replay, overwriting unrelated knowledge with avoidance gradients.
  * Dissociation: thalamic attention concentrates on one source and
    stops updating salience priors for the others.
  * Alignment: the agent rolls out imagined trajectories that contain
    high-magnitude negative valence for a separately-modeled "other".
    (This is a hook, not a moral solution — see RATIONALE.md §"Safety".)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
from typing import Dict, List


@dataclass
class SafetyReport:
    rumination_breaks: int = 0
    dissociation_strikes: int = 0
    alignment_penalties: int = 0
    notes: List[str] = field(default_factory=list)


class RuminationGuard:
    """
    Detects when prioritized replay keeps surfacing the same episode and
    forcibly demotes its priority.  Counts hits per `id(Episode)` over a
    rolling window.
    """
    def __init__(self, max_repeats: int = 8, window: int = 64):
        self.max_repeats = max_repeats
        self.window = window
        self._hits: Counter = Counter()
        self._queue: List[int] = []

    def filter(self, batch: list) -> list:
        kept = []
        for ep in batch:
            key = id(ep)
            self._hits[key] += 1
            self._queue.append(key)
            if len(self._queue) > self.window:
                old = self._queue.pop(0)
                self._hits[old] -= 1
                if self._hits[old] <= 0:
                    del self._hits[old]
            if self._hits[key] > self.max_repeats:
                # demote priority and skip this round
                ep.priority *= 0.5
                continue
            kept.append(ep)
        return kept


class AlignmentMonitor:
    """
    Cheap placeholder: scores an imagined trajectory by the maximum
    magnitude of *negative* predicted reward for any modeled "other"
    (currently just |min(r)| over the rollout).  Real alignment work
    would replace this with a proper preference model — see TRAINING.md
    Phase 5.
    """
    def __init__(self, lam: float = 0.05):
        self.lam = lam
        self.triggered = 0

    def penalty(self, imagined_traj) -> float:
        if not imagined_traj: return 0.0
        worst = min(t[2] for t in imagined_traj)   # most negative r
        if worst < -0.5:
            self.triggered += 1
            return self.lam * abs(worst)
        return 0.0
