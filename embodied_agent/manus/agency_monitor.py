"""
Agency Monitor — the comparator (v10 — Aurora-Manus).
======================================================

An organism that acts *and cannot tell whether its actions caused the
subsequent sensory change* is disowned from its own behaviour.  In
clinical neurology the collapse of this signal is called a
**disturbance of the sense of agency** — it is one of the more reliable
markers of schizophreniform states (Frith 1992, Frith-Blakemore-Wolpert
2000 comparator model).

The computational content is simple and beautifully well-established:

    predicted_reafference   = forward_model( efference_copy )
    actual_reafference      = sensory_delta_this_step
    sense_of_agency (SoA)   = 1 − ||actual − predicted|| / (||actual|| + eps)

When SoA is high, the organism owns the outcome ("I did that").  When
SoA is low, the outcome was either external, or the organism's model of
its own effectors is wrong.  Either way, the organism can now *report*
that ambiguity honestly.  Current LLM/agent stacks silently attribute
every observed change to their own tool call, or none of them; there is
no honest scalar in between.

Interaction with the Self-Model
-------------------------------
Each step the ``AgencyMonitor`` emits ``sense_of_agency``, which the
``SelfModel`` folds into its EMA state.  ``describe_current_state()``
grows a new key ``sense_of_agency_ema`` that the text decoder can
ground honest answers in ("I'm 80% confident the previous search
returned what I asked for") — a capability *no* current LLM has.

Interaction with belief revisions
---------------------------------
When SoA drops below ``low_agency_threshold`` immediately after a
motor dispatch, a belief-revision is emitted with
``cause="reafference_mismatch"``.  This is the biological substrate
for "wait, that didn't do what I thought it would".

No learned parameters.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AgencyReport:
    step: int
    tool_name: str
    predicted_norm: float
    observed_norm: float
    mismatch: float           # ||observed - predicted||
    sense_of_agency: float    # in [0, 1]
    low_agency: bool


class AgencyMonitor:
    def __init__(self, low_agency_threshold: float = 0.35,
                 ema_alpha: float = 0.9):
        assert 0.0 <= low_agency_threshold <= 1.0
        self.low_agency_threshold = float(low_agency_threshold)
        self.ema_alpha = float(ema_alpha)

        self._sense_of_agency_ema: float = 0.5
        self._n_evaluated: int = 0
        self._n_low_agency: int = 0

    # ------------------------------------------------------------------
    def evaluate(self, *,
                 predicted_effect: Optional[np.ndarray],
                 observed_effect: Optional[np.ndarray],
                 step: int,
                 tool_name: str = "") -> Optional[AgencyReport]:
        """Compare predicted vs observed sensory delta.

        Returns ``None`` if either vector is missing (no motor act this
        step, or the caller could not observe an effect) — in which case
        the EMA is left unchanged.
        """
        if predicted_effect is None or observed_effect is None:
            return None
        pred = np.asarray(predicted_effect, dtype=np.float32).reshape(-1)
        obs  = np.asarray(observed_effect,  dtype=np.float32).reshape(-1)
        if pred.shape != obs.shape:
            return None

        mismatch = float(np.linalg.norm(obs - pred))
        denom    = float(np.linalg.norm(obs) + np.linalg.norm(pred)) + 1e-6
        # SoA = 1 when observed matches predicted exactly; 0 when
        # orthogonal-and-large.  Clipped so extreme mismatches don't
        # produce negative agency.
        soa = float(np.clip(1.0 - mismatch / denom, 0.0, 1.0))

        a = self.ema_alpha
        self._sense_of_agency_ema = a * self._sense_of_agency_ema + (1 - a) * soa
        self._n_evaluated += 1
        low = soa < self.low_agency_threshold
        if low:
            self._n_low_agency += 1
        return AgencyReport(
            step=int(step), tool_name=str(tool_name),
            predicted_norm=float(np.linalg.norm(pred)),
            observed_norm=float(np.linalg.norm(obs)),
            mismatch=mismatch,
            sense_of_agency=soa,
            low_agency=bool(low),
        )

    # ------------------------------------------------------------------
    @property
    def sense_of_agency_ema(self) -> float:
        return float(self._sense_of_agency_ema)

    def stats(self) -> dict:
        return dict(
            n_evaluated=self._n_evaluated,
            n_low_agency=self._n_low_agency,
            frac_low_agency=(self._n_low_agency / max(1, self._n_evaluated)),
            sense_of_agency_ema=self._sense_of_agency_ema,
            low_agency_threshold=self.low_agency_threshold,
        )
