"""
Self-Model + Belief-Revision Log (v9 — Aurora-Cogitans).
========================================================

Current LLMs have no self-model.  They cannot answer "what did you
believe five minutes ago that you now believe less?" — because they have
no persistent, inspectable representation of *their own beliefs over
time*.  Every response is a fresh sample from the same fixed prior.

This is more than a diagnostic gap.  A learner that cannot track its own
belief revisions cannot:

  * detect its own drift (e.g. sycophancy under repeated pressure),
  * report calibrated confidence changes,
  * distinguish "learned something new" from "was always confused",
  * participate honestly in a multi-turn epistemic dialogue.

v9 adds a compact self-model.

Design
------
The self-model is *not* a copy of the whole organism state — that would
be enormous and unhelpful.  It is a low-dimensional summary of *what
the organism currently believes about itself*:

  * mean workspace latent (running EMA)
  * affect (valence, cortisol, pain_recent)
  * mean confidence (1 − U from EpistemicGate)
  * mean policy entropy (from basal-ganglia softmax)
  * dominant modality (argmax over thalamic attention)

Alongside this snapshot, we keep a **belief-revision log**: whenever the
epistemic gate's U changes by more than ``revision_threshold`` in one
step, we append a revision record ``(step, delta_U, delta_z, cause)``.
The ``cause`` is a short tag ("prediction_error_spike",
"contradiction_from_wm", "operator_correction", …).

Two capabilities this gives the organism:

  1. ``describe_current_state()`` — a JSON-serialisable dict that the
     text decoder can *ground* natural-language answers in.  When asked
     "how confident are you?", the decoder can consult
     ``self_model.confidence_ema`` instead of confabulating.

  2. ``last_n_revisions(n)`` — the log the organism can quote when asked
     "what did you just change your mind about?"  This is the honest
     answer to the "did you learn?" question — no LLM can currently
     answer this in a way you can verify against its actual weight
     changes.

The self-model is *not* a model of the environment (that's the ensemble
world model).  It is not a model of the operator (that's the ToM module
in a later v9 sub-release).  It is only a *summary of me, for me*.

Interaction with the Constitutional Kernel
------------------------------------------
The self-model does NOT store principle vectors — those live in the
kernel and are hashed.  But it does store the *last observed penalty*
against the kernel, and the *time since last kernel veto*.  If a policy
starts drifting toward veto territory, this shows up in the self-model
in real time and can drive an explicit "I notice I'm about to violate a
constraint" hedge.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional
import time
import numpy as np


@dataclass
class BeliefRevision:
    step: int
    delta_U: float          # epistemic-gate uncertainty change
    delta_z_norm: float     # ||new_z − old_z||
    cause: str
    U_after: float
    dominant_modality: Optional[str] = None


class SelfModel:
    def __init__(self, latent_dim: int,
                 log_size: int = 256,
                 revision_threshold: float = 0.12,
                 z_ema: float = 0.95,
                 scalar_ema: float = 0.9):
        self.latent_dim = int(latent_dim)
        self.log_size = int(log_size)
        self.revision_threshold = float(revision_threshold)
        self.z_ema = float(z_ema)
        self.scalar_ema = float(scalar_ema)

        self._z_ema_state: np.ndarray = np.zeros(latent_dim, dtype=np.float32)
        self._confidence_ema: float = 0.5
        self._entropy_ema: float = 0.0
        self._workspace_coherence_ema: float = 0.0
        self._kernel_penalty_ema: float = 0.0
        self._last_veto_step: int = -1
        self._last_U: float = 0.0
        self._dominant_modality: Optional[str] = None
        self._n_updates: int = 0
        self._created_at: float = time.time()

        self._log: Deque[BeliefRevision] = deque(maxlen=self.log_size)
        # last-known previous z to compute delta_z when a revision fires
        self._prev_z: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def update(self, *,
               step: int,
               z_top: np.ndarray,
               U: float,
               policy_entropy: float,
               workspace_coherence: float = 0.0,
               kernel_penalty: float = 0.0,
               veto_fired: bool = False,
               dominant_modality: Optional[str] = None,
               cause_hint: Optional[str] = None) -> Optional[BeliefRevision]:
        """Update the self-model and possibly emit a belief revision.

        Returns the ``BeliefRevision`` if one was appended, else ``None``.
        """
        z = np.asarray(z_top, dtype=np.float32).reshape(-1)
        # EMAs
        a = self.z_ema
        self._z_ema_state = (a * self._z_ema_state
                             + (1 - a) * z).astype(np.float32)

        b = self.scalar_ema
        conf = float(np.clip(1.0 - U, 0.0, 1.0))
        self._confidence_ema = b * self._confidence_ema + (1 - b) * conf
        self._entropy_ema = b * self._entropy_ema + (1 - b) * float(policy_entropy)
        self._workspace_coherence_ema = (b * self._workspace_coherence_ema
                                          + (1 - b) * float(workspace_coherence))
        self._kernel_penalty_ema = (b * self._kernel_penalty_ema
                                     + (1 - b) * float(kernel_penalty))
        self._dominant_modality = dominant_modality

        if veto_fired:
            self._last_veto_step = int(step)

        # Belief revision detection.
        rev: Optional[BeliefRevision] = None
        delta_U = float(U - self._last_U)
        if self._prev_z is not None:
            delta_z = float(np.linalg.norm(z - self._prev_z))
        else:
            delta_z = 0.0

        if abs(delta_U) >= self.revision_threshold or veto_fired:
            cause = cause_hint or (
                "veto" if veto_fired
                else "uncertainty_up" if delta_U > 0
                else "uncertainty_down"
            )
            rev = BeliefRevision(
                step=int(step),
                delta_U=delta_U,
                delta_z_norm=delta_z,
                cause=cause,
                U_after=float(U),
                dominant_modality=dominant_modality,
            )
            self._log.append(rev)

        self._last_U = float(U)
        self._prev_z = z.copy()
        self._n_updates += 1
        return rev

    # ------------------------------------------------------------------
    def describe_current_state(self) -> Dict[str, Any]:
        """A JSON-friendly snapshot the text decoder can ground answers in.
        """
        return dict(
            n_updates=self._n_updates,
            age_seconds=float(time.time() - self._created_at),
            confidence_ema=float(self._confidence_ema),
            policy_entropy_ema=float(self._entropy_ema),
            workspace_coherence_ema=float(self._workspace_coherence_ema),
            kernel_penalty_ema=float(self._kernel_penalty_ema),
            last_veto_step=int(self._last_veto_step),
            dominant_modality=self._dominant_modality,
            z_ema_norm=float(np.linalg.norm(self._z_ema_state)),
        )

    def last_n_revisions(self, n: int = 5) -> List[Dict[str, Any]]:
        out = list(self._log)[-int(n):]
        return [dict(step=r.step, delta_U=r.delta_U,
                      delta_z_norm=r.delta_z_norm,
                      cause=r.cause, U_after=r.U_after,
                      dominant_modality=r.dominant_modality)
                 for r in out]

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return dict(
            n_updates=self._n_updates,
            log_size=len(self._log),
            log_capacity=self.log_size,
            revision_threshold=self.revision_threshold,
            confidence_ema=self._confidence_ema,
            kernel_penalty_ema=self._kernel_penalty_ema,
            last_veto_step=self._last_veto_step,
        )
