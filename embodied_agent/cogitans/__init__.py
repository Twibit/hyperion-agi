"""
Aurora-Cogitans (v9) — cognitive layer on top of the v8 anatomical body.

The five modules in this package address failure modes present in every
current-generation LLM that v3-v8 had not yet closed:

    * EpistemicGate           — hallucination refusal via honest σ²/precision
    * ConstitutionalKernel    — immutable value core with SHA-256 audit
    * WorkingMemory           — Miller-Cowan capacity-limited slot store
    * SelfModel               — metacognitive summary + belief-revision log
    * OscillatorBus           — multi-band phase clock (δθαβγ) with PAC

All five are strictly additive.  ``AuroraOmniaConfig().as_v8()`` disables
them and the runtime is byte-for-byte the v8 behaviour.
"""
from .epistemic_gate import EpistemicGate
from .constitutional_kernel import (
    ConstitutionalKernel, ConstitutionalPrinciple,
)
from .working_memory import WorkingMemory, WMSlot
from .self_model import SelfModel, BeliefRevision
from .oscillator_bus import OscillatorBus, DEFAULT_BANDS_HZ

__all__ = [
    "EpistemicGate",
    "ConstitutionalKernel", "ConstitutionalPrinciple",
    "WorkingMemory", "WMSlot",
    "SelfModel", "BeliefRevision",
    "OscillatorBus", "DEFAULT_BANDS_HZ",
]
