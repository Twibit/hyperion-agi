"""
Somatosensory Field — proprioception & haptics (v10 — Aurora-Manus).
=====================================================================

Language models learn "hot" as a co-occurrence pattern between the
token "hot" and other tokens.  They do not learn it as a *signal*.
This is the ungrounded-symbol problem in one line.

Real cortex grounds haptic and proprioceptive terms in the S1 /
posterior-parietal body schema: a continuously-updated latent map of
"where my effectors are, what they last did, and what the world felt
like when they did it".  The map is not the same as the world model —
it is *the model of the self as a physical thing among things*.

v10's ``SomatosensoryField`` is the minimal viable version of that:

  * ingests each motor return from ``MotorCortex`` (predicted_effect,
    observed_effect, cost, tool_name)
  * maintains a rolling latent ``body_schema`` (EMA of recent efference
    copies) and a rolling ``touch_trace`` (EMA of recent reafference)
  * exposes both as a new modality named ``"somatic"`` on the shared
    bus, so the cortex, HVSA and workspace can bind against them the
    same way they bind against text/vision/audio/interoception

Why an EMA and not a plain buffer?
----------------------------------
An EMA gives us the correct biological property that body schema
degrades slowly under sensory deprivation (the classic "phantom limb"
literature is *exactly* the failure mode of a static schema starved of
reafference).  Slow decay under absent input, fast update under new
input.

Interaction with existing modules
---------------------------------
* The organism's per-step observation dict optionally gains a
  ``"somatic"`` entry equal to ``field.readout()``.  Because the config
  already lists "proprio" in ``modalities`` (see v3 config), no
  breaking change is needed; but a deployer can *also* register a
  full ``"somatic"`` modality if they want richer haptic grounding.
* The thalamic salience router (v8) can boost regions declaring the
  ``"proprio"`` modality when the body schema changes rapidly — this
  is the "flinch" mechanism.

No learned parameters.
"""
from __future__ import annotations
from typing import Optional
import numpy as np


class SomatosensoryField:
    def __init__(self, latent_dim: int,
                 schema_decay: float = 0.92,
                 touch_decay: float = 0.85):
        self.latent_dim = int(latent_dim)
        self.schema_decay = float(schema_decay)
        self.touch_decay  = float(touch_decay)

        self._body_schema = np.zeros(self.latent_dim, dtype=np.float32)
        self._touch_trace = np.zeros(self.latent_dim, dtype=np.float32)
        self._n_updates = 0
        self._last_cost = 0.0

    # ------------------------------------------------------------------
    def _norm(self, v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v)) + 1e-9
        return (v / n).astype(np.float32)

    def ingest(self, *,
               predicted_effect: Optional[np.ndarray] = None,
               observed_effect: Optional[np.ndarray] = None,
               cost: float = 0.0) -> None:
        """Fold this step's motor evidence into the schema.

        Either half is optional — if the organism performed no motor
        act, both decay toward zero at their configured rates.
        """
        # Schema always decays a little each tick (S1 receptive-field
        # habituation).
        self._body_schema *= self.schema_decay
        self._touch_trace *= self.touch_decay

        if predicted_effect is not None:
            pe = np.asarray(predicted_effect, dtype=np.float32).reshape(-1)
            if pe.shape[0] == self.latent_dim:
                self._body_schema = (self._body_schema
                                      + (1.0 - self.schema_decay) * pe)
        if observed_effect is not None:
            oe = np.asarray(observed_effect, dtype=np.float32).reshape(-1)
            if oe.shape[0] == self.latent_dim:
                self._touch_trace = (self._touch_trace
                                      + (1.0 - self.touch_decay) * oe)

        self._last_cost = float(cost)
        self._n_updates += 1

    # ------------------------------------------------------------------
    def readout(self) -> np.ndarray:
        """Unit-norm somatic latent = 0.6·body_schema ⊕ 0.4·touch_trace."""
        v = 0.6 * self._body_schema + 0.4 * self._touch_trace
        if float(np.linalg.norm(v)) < 1e-9:
            return np.zeros(self.latent_dim, dtype=np.float32)
        return self._norm(v)

    def flinch(self) -> float:
        """A scalar in [0, ∞) — magnitude of instantaneous somatic change.

        This is what the thalamic router should treat as salience-of-
        body-schema-change.  Sharp reafference / sudden cost pushes it
        up.
        """
        return float(np.linalg.norm(self._touch_trace)
                     + 0.5 * abs(self._last_cost))

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return dict(
            n_updates=self._n_updates,
            schema_norm=float(np.linalg.norm(self._body_schema)),
            touch_norm=float(np.linalg.norm(self._touch_trace)),
            last_cost=self._last_cost,
            flinch=self.flinch(),
        )
