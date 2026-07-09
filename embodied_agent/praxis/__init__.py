"""
Aurora-Praxis (V11) — The Long-Horizon Planning & Execution Layer.

This package provides the modules necessary for long-horizon agentic workflows,
recursive reasoning, and advanced neurobiological control loops.

Modules:
- PrefrontalCortex: Executive function, long-term Goal SDR management.
- HTMSequenceMemory: Sequence prediction for trajectory simulation.
- Microglia: Immune pruning — garbage collection during sleep.
- EndocrineState: Hormonal layer (Adrenaline/Oxytocin/Melatonin).
- WernickesArea: Language comprehension (surface form → semantic SDR).
- BrocasArea: Language production (semantic SDR → token sequence).
- ArcuateFasciculus: Bidirectional bridge + internal monologue.
"""
from .htm_predictor import HTMSequenceMemory
from .prefrontal_cortex import PrefrontalCortex
from .microglia import Microglia
from .endocrine import EndocrineState
from .language_grounding import WernickesArea, BrocasArea, ArcuateFasciculus

__all__ = [
    "HTMSequenceMemory",
    "PrefrontalCortex",
    "Microglia",
    "EndocrineState",
    "WernickesArea",
    "BrocasArea",
    "ArcuateFasciculus",
]
