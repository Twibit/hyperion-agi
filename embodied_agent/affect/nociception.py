"""
Nociception & valence — the principled answer to "should an AI learn from pain?"
================================================================================

This is the module the user explicitly asked about.  Short version:

  *Pain as a *gradient signal* is fine and necessary*: it tells the
  organism which actions to avoid.  Standard RL already uses it under
  the name "negative reward".

  *Pain as *experience* (chronic, unbounded, self-reinforcing) is
  catastrophic*: in animals it produces PTSD-like phenomena; in
  artificial learners it produces (a) reward hacking, (b) policy
  collapse onto avoidance, (c) memory-replay loops that overwrite
  useful representations with the aversive episode.

Our design therefore treats nociception with **three engineered safeguards**:

  1. **CLIP** — every nociceptive signal is bounded to |p| ≤ nociception_clip
     before it can drive learning.  Unbounded pain cannot exist in this
     architecture.  This is the analogue of biological *opioid gating*.

  2. **TWO-CHANNEL STORAGE** — pain produces a *learning gradient* in the
     basal ganglia AND a *labeled* episodic tag in the hippocampus.  The
     gradient is applied once; the episode is *quarantined* from
     prioritized replay if its magnitude exceeds the trauma threshold.
     This is the analogue of *consolidation gating* during sleep, which
     in biology requires the absence of acute cortisol.

  3. **CORTISOL FEEDBACK** — sustained pain raises a stress scalar that
     **suppresses cortex plasticity** (lr_cortex *= 1 - cortisol).  This
     prevents the rest of the network from re-writing itself in response
     to a transient hostile environment, matching the empirical finding
     that high cortisol selectively impairs hippocampal LTP.

Net effect: the system can be safely trained with negative reward and
even simulated injury signals, without converging to a learner that
1) self-harms to satisfy a reward exploit, 2) catastrophically forgets,
or 3) preferentially attends to its own pain.

See the paper-style discussion in RATIONALE.md §"On Pain & Trauma".
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Deque


@dataclass
class AffectState:
    """Continuous low-dimensional representation of the organism's mood."""
    valence: float = 0.0           # -1 (bad) .. +1 (good)
    arousal: float = 0.0           #  0 (calm) ..  1 (frantic)
    cortisol: float = 0.0          #  0 .. 1, stress hormone
    pain_recent: float = 0.0       # |EMA pain| (bounded)
    quarantined_count: int = 0     # episodes refused into replay


class Nociceptor:
    """
    Converts raw aversive signals (negative reward, internal predicted
    catastrophe, simulated bodily damage) into:
      * a bounded `pain` scalar used as a learning gradient,
      * an update to AffectState,
      * a *flag* `quarantine` advising the hippocampus to tag the
        current episode as not-for-replay.

    The nociceptor never carries memory of specific events itself —
    that lives in the hippocampus, behind the quarantine gate.
    """

    def __init__(self,
                 clip: float = 1.0,
                 valence_decay: float = 0.02,
                 cortisol_gain: float = 0.6,
                 trauma_threshold: float = 2.5,
                 window: int = 64):
        self.clip = clip
        self.valence_decay = valence_decay
        self.cortisol_gain = cortisol_gain
        self.trauma_threshold = trauma_threshold
        self.state = AffectState()
        self._history: Deque[float] = deque(maxlen=window)

    # ------------------------------------------------------------------ #
    def ingest(self, raw_signal: float, novelty: float = 0.0) -> dict:
        """
        Process one raw aversive (or appetitive) signal.

        Returns dict:
          pain        : clipped learning signal (use as a *gradient*; not stored)
          quarantine  : bool — refuse this episode into prioritized replay
          lr_scale    : multiplicative factor for cortex/world-model lr
        """
        # Bound first — nothing downstream ever sees unclipped values.
        clipped = float(np.clip(raw_signal, -self.clip, self.clip))

        # Update mood EMAs.
        self.state.valence  += self.valence_decay * (clipped - self.state.valence)
        self.state.arousal  = float(np.clip(0.9 * self.state.arousal + 0.5 * abs(clipped), 0.0, 1.0))
        self.state.pain_recent = float(np.clip(0.95 * self.state.pain_recent
                                               + 0.05 * max(0.0, -clipped), 0.0, 1.0))

        # Cortisol responds to *sustained* pain, not single spikes.
        if clipped < 0:
            self.state.cortisol = float(np.clip(self.state.cortisol
                                                + self.cortisol_gain * (-clipped) * 0.1,
                                                0.0, 1.0))
        else:
            self.state.cortisol = float(np.clip(self.state.cortisol - 0.01, 0.0, 1.0))

        # Trauma quarantine: extreme raw signals are flagged
        # (we test the RAW, not the clipped, because the *world* really did
        # produce that magnitude — but learning still uses the clipped value).
        quarantine = abs(raw_signal) > self.trauma_threshold

        # Stress-suppressed plasticity (biology: high cortisol → impaired LTP).
        lr_scale = float(np.clip(1.0 - 0.7 * self.state.cortisol, 0.1, 1.0))

        self._history.append(clipped)
        if quarantine:
            self.state.quarantined_count += 1

        return dict(pain=clipped, quarantine=quarantine, lr_scale=lr_scale)

    # ------------------------------------------------------------------ #
    def chronic_stress(self) -> bool:
        """True when the EMA of pain has stayed elevated long enough that
        the controller should intervene (e.g. trigger a sleep cycle)."""
        if len(self._history) < self._history.maxlen:
            return False
        recent_neg = [h for h in self._history if h < -0.3]
        return len(recent_neg) > 0.6 * self._history.maxlen

    def snapshot(self) -> dict:
        return dict(valence=self.state.valence, arousal=self.state.arousal,
                    cortisol=self.state.cortisol, pain_recent=self.state.pain_recent,
                    quarantined=self.state.quarantined_count)
