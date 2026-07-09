"""
Neuromodulators — DA / 5HT / NE / ACh.  v3 adds explicit *coupling* to the
affect system: cortisol (from the Nociceptor) clamps `effective_lr`, so
that learning rates do not climb during chronic stress even if local
prediction error suggests they should.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class NeuromodulatorState:
    dopamine: float = 0.5
    serotonin: float = 0.5
    norepinephrine: float = 0.3
    acetylcholine: float = 0.4

    tau_da: float = 0.05
    tau_5ht: float = 0.005
    tau_ne: float = 0.10
    tau_ach: float = 0.02

    base_da: float = 0.5
    base_5ht: float = 0.5
    base_ne: float = 0.3
    base_ach: float = 0.4

    def update(self, reward_prediction_error=0.0, sensory_surprise=0.0,
               task_progress=0.0, attention_load=0.0):
        self.dopamine       += self.tau_da  * (self.base_da  - self.dopamine)
        self.serotonin      += self.tau_5ht * (self.base_5ht - self.serotonin)
        self.norepinephrine += self.tau_ne  * (self.base_ne  - self.norepinephrine)
        self.acetylcholine  += self.tau_ach * (self.base_ach - self.acetylcholine)
        self.dopamine       = float(np.clip(self.dopamine       + 0.5 * reward_prediction_error, 0.0, 1.0))
        self.norepinephrine = float(np.clip(self.norepinephrine + 0.5 * sensory_surprise,        0.0, 1.0))
        self.serotonin      = float(np.clip(self.serotonin      + 0.1 * task_progress,           0.0, 1.0))
        self.acetylcholine  = float(np.clip(self.acetylcholine  + 0.3 * attention_load,          0.0, 1.0))

    def effective_lr(self, base_lr: float, cortisol: float = 0.0) -> float:
        # v3: cortisol from the Nociceptor clamps plasticity (max ~70% suppression).
        plasticity = (self.acetylcholine + 0.5 * self.norepinephrine) * (1.0 - 0.7 * cortisol)
        return base_lr * max(0.1, plasticity)

    def td_gain(self):    return self.dopamine
    def discount(self):   return 0.9 + 0.09 * self.serotonin
    def exploration_temp(self, base=1.0):
        return base * (0.5 + self.norepinephrine - 0.3 * self.dopamine)
    def snapshot(self):
        return dict(da=self.dopamine, ht=self.serotonin,
                    ne=self.norepinephrine, ach=self.acetylcholine)
