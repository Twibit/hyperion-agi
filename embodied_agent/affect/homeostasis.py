"""
Homeostatic controller — uses the AffectState to *act on* the rest of the
organism: nudge learning rates, trigger sleep when stress accumulates,
choose between exploration / exploitation, and inject "self-care" actions
when the chronic stress flag has been set.

Note: this is NOT a reward-hacking module.  It cannot fabricate reward,
only modulate *learning* and *attention*.  Reward still comes from the
environment.
"""
from __future__ import annotations
from .nociception import Nociceptor


class HomeostaticController:
    def __init__(self, nociceptor: Nociceptor, setpoint: float = 0.0):
        self.nocic = nociceptor
        self.setpoint = setpoint

    def request_sleep(self) -> bool:
        """Returns True when the controller wants an immediate sleep cycle —
        e.g. after a burst of pain it's better to consolidate before acting."""
        return self.nocic.chronic_stress() or self.nocic.state.cortisol > 0.85

    def lr_scale(self, base_lr: float) -> float:
        """Cortisol clamps cortical plasticity; mild positive valence boosts it."""
        s = 1.0 - 0.7 * self.nocic.state.cortisol
        s += 0.2 * max(0.0, self.nocic.state.valence - self.setpoint)
        return base_lr * max(0.1, s)

    def exploration_bias(self) -> float:
        """High arousal + chronic stress => narrow attention (exploit),
        low arousal + positive valence => wide attention (explore)."""
        return float(0.5 + 0.5 * self.nocic.state.valence
                     - 0.3 * self.nocic.state.cortisol)
