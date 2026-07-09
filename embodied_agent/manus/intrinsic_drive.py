"""
Intrinsic Drive — curiosity, mastery, novelty (v10 — Aurora-Manus).
====================================================================

Aurora v3-v9 has an extrinsic reward channel (``reward``) and a stress
channel (``cortisol``), but no principled *intrinsic* reward.  Without
one, the organism has no reason to reach for the unknown when the
extrinsic signal is silent — a well-documented failure mode in both
biology (dopamine-deficient animals become behaviourally inert even in
rich environments) and RL (sparse-reward tasks stall).

The neuroscience answer is a *three-factor* intrinsic reward:

  * **prediction-gain** (mastery)    Kakade & Dayan 2002, Schmidhuber
                                     "compression progress"
  * **novelty**                       Kakade-Dayan, hippocampal-novelty
                                     dopamine bursts (Lisman-Grace 2005)
  * **competence-gain** (control)    Chentanez-Barto 2005 / Oudeyer 2007
                                     progress in effect-schema prediction

The intrinsic reward is *not* added to the extrinsic reward
unconditionally.  It is gated by the epistemic gate: if the gate is in
``refuse`` mode the organism should not train itself on curiosity about
things it cannot even model.  This is the exact analogue of the v3
policy that dreams stop when ensemble σ² is high (§1.2 RATIONALE) —
we do not train the learner on its own hallucinations.

Interaction with the safety machinery
-------------------------------------
* Cortisol suppression of extrinsic reward extends to intrinsic
  reward: under high stress the drive is dampened to prevent
  novelty-seeking-in-danger.
* Nociception clip caps intrinsic reward too — an intrinsically-driven
  agent must not be able to bypass pain gating.

No learned parameters.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class IntrinsicSample:
    mastery: float          # prediction-error improvement
    novelty: float          # stigmergic or z-space novelty
    competence: float       # sense-of-agency improvement
    r_int: float            # final gated intrinsic reward
    gated_off: bool         # True if the gate refused / cortisol clamped


class IntrinsicDrive:
    def __init__(self,
                 alpha_mastery: float = 0.4,
                 beta_novelty: float = 0.4,
                 gamma_competence: float = 0.2,
                 clip: float = 1.0,
                 cortisol_suppression: float = 0.7):
        self.alpha = float(alpha_mastery)
        self.beta  = float(beta_novelty)
        self.gamma = float(gamma_competence)
        self.clip  = float(clip)
        self.cortisol_suppression = float(cortisol_suppression)

        self._prev_sensory_err: Optional[float] = None
        self._prev_soa: Optional[float] = None
        self._n = 0
        self._sum_r_int = 0.0
        self._sum_r_int_gated = 0.0

    # ------------------------------------------------------------------
    def compute(self, *,
                sensory_err: float,
                novelty: float,
                sense_of_agency: Optional[float],
                cortisol: float,
                epistemic_mode: Optional[str] = None) -> IntrinsicSample:
        # Mastery = *improvement* in prediction error since last step.
        if self._prev_sensory_err is None:
            mastery = 0.0
        else:
            mastery = float(self._prev_sensory_err - sensory_err)
        self._prev_sensory_err = float(sensory_err)

        # Competence = improvement in sense-of-agency since last step.
        if sense_of_agency is None:
            competence = 0.0
        else:
            if self._prev_soa is None:
                competence = 0.0
            else:
                competence = float(sense_of_agency - self._prev_soa)
            self._prev_soa = float(sense_of_agency)

        # Raw intrinsic signal.
        r = (self.alpha * mastery
             + self.beta  * float(novelty)
             + self.gamma * competence)

        gated_off = False
        # Refuse-mode short-circuit — see docstring.
        if epistemic_mode == "refuse":
            r = 0.0
            gated_off = True

        # Cortisol suppression of the intrinsic drive (matches the v3
        # cortisol-lr-suppression semantic).
        r *= (1.0 - self.cortisol_suppression * float(np.clip(cortisol, 0.0, 1.0)))

        # Clip to keep the loss landscape sane.
        r = float(np.clip(r, -self.clip, self.clip))

        self._n += 1
        self._sum_r_int += r
        if not gated_off:
            self._sum_r_int_gated += r

        return IntrinsicSample(
            mastery=float(mastery), novelty=float(novelty),
            competence=float(competence), r_int=r, gated_off=bool(gated_off),
        )

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        n = max(1, self._n)
        return dict(
            n=self._n,
            mean_r_int=self._sum_r_int / n,
            mean_r_int_active=self._sum_r_int_gated / n,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
            clip=self.clip,
        )
