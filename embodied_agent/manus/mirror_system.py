"""
Mirror System — observation ↔ execution overlap (v10 — Aurora-Manus).
======================================================================

Rizzolatti et al.'s mirror-neuron discovery (PMv, IPL; monkeys in the
early 1990s, later in human PET/fMRI) is one of the more provocative
findings in motor neuroscience: the *same* neurons that fire when the
monkey grasps a peanut also fire when the monkey watches the
experimenter grasp a peanut.  The computational consequence is that
observed actions can be interpreted through the organism's own motor
repertoire — a natural substrate for imitation learning and for
elementary theory-of-mind.

Current agent stacks cannot do this in any grounded way.  Given a
demonstration ("here is a video of a person clicking Search, typing
'q', pressing Enter"), an LLM parses the *tokens* but does not map the
sequence back to its own tool inventory in a way that lets it *do the
same thing* one-shot.

v10 supplies the minimum viable mirror system.

Design
------
The mirror system holds a reference to the ``MotorCortex``.  Given an
*observed* effect latent — say, the sensory-bus delta produced by an
external agent's action — it computes the cosine similarity between
that vector and every tool's ``effect_schema``.  The argmax is the
best match, and its score is the *mirror confidence*.  Above a
threshold the mirror emits a ``MirrorMatch(tool_name, similarity)``
that the organism can consume as a *one-shot imitation hint*.

Two capabilities this gives the organism:

  1. **Grounded imitation.**  When the operator provides a
     demonstration trajectory, the mirror system produces a sequence
     of tool-name hints derived from the organism's own motor
     repertoire — imitation without any policy gradient step.

  2. **Elementary ToM.**  When the mirror confidence is high on some
     observed action, the organism can honestly say "if I had done
     that, I would have used tool X".  This is a much weaker claim
     than mental-state modelling, but it is the *actual* biological
     substrate — the empathy-through-motor-simulation pathway.

Interaction with the rest of Aurora
-----------------------------------
* The mirror is a *read-only* consumer of the motor cortex.  It does
  not mutate the tool table.
* When the epistemic gate is in ``refuse`` mode we still allow
  mirroring — observation is not the same as action.
* Matches above threshold can be admitted into ``WorkingMemory`` with
  a special ``tag="mirror_hint"`` so the organism has a Cowan-4 buffer
  of pending imitations.

No learned parameters.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:  # avoid runtime import cycle
    from .motor_cortex import MotorCortex


@dataclass
class MirrorMatch:
    tool_name: str
    similarity: float          # in [-1, 1] cosine
    step: int


class MirrorSystem:
    def __init__(self, motor: "MotorCortex",
                 match_threshold: float = 0.55):
        self.motor = motor
        self.match_threshold = float(match_threshold)
        self._recent: List[MirrorMatch] = []
        self._max_recent = 64
        self._n_observed = 0
        self._n_matched = 0

    # ------------------------------------------------------------------
    def _norm(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v)) + 1e-9
        return (v / n).astype(np.float32)

    def observe(self, observed_effect: np.ndarray,
                step: int = 0) -> Optional[MirrorMatch]:
        """Map an observed effect latent to the organism's motor repertoire.

        Returns the best-matching ``MirrorMatch`` iff its cosine
        similarity exceeds ``match_threshold``; else ``None`` (the
        observation was not interpretable as any known motor act).
        """
        tools = self.motor.tools()
        if not tools:
            return None
        v = self._norm(observed_effect)
        self._n_observed += 1
        best_name = None
        best_sim  = -2.0
        for name in tools:
            spec = self.motor._tools[name]
            sim = float(np.dot(spec.effect_schema, v))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_name is None or best_sim < self.match_threshold:
            return None
        match = MirrorMatch(tool_name=best_name,
                             similarity=float(best_sim),
                             step=int(step))
        self._recent.append(match)
        if len(self._recent) > self._max_recent:
            self._recent = self._recent[-self._max_recent:]
        self._n_matched += 1
        return match

    def observe_trajectory(self, effects: List[np.ndarray],
                           start_step: int = 0) -> List[Optional[MirrorMatch]]:
        """Batch version — one match per observed effect vector."""
        return [self.observe(e, step=start_step + i)
                 for i, e in enumerate(effects)]

    # ------------------------------------------------------------------
    def recent(self, n: int = 8) -> List[MirrorMatch]:
        return list(self._recent[-int(n):])

    def stats(self) -> dict:
        return dict(
            n_observed=self._n_observed,
            n_matched=self._n_matched,
            match_rate=self._n_matched / max(1, self._n_observed),
            threshold=self.match_threshold,
            recent=[dict(tool=m.tool_name, sim=m.similarity, step=m.step)
                     for m in self.recent(8)],
        )
