"""
Hyperion v10 — Aurora-Manus.

A brain-inspired generalist learner with:

  * v3 baseline: predictive cortex, MoE, ensemble world model, episodic
    hippocampus, basal-ganglia actor-critic, affect & homeostat, sleep
    consolidation, safety guards.

  * v6 Aurora-Omnia layer: HVSA omniscient eye, stigmergic concept graph,
    sparse distributed memory, astrocyte metaplasticity, global workspace
    + DMN, cerebellum forward/inverse model, STDP eligibility traces,
    synaptic tagging & capture + Benna-Fusi cascade, introspection,
    neurogenesis, active inference.

  * v7 Aurora-Cerebrum layer: explicit 1e14-synapse atlas distributed
    across left/right cerebral hemispheres (each with 4 lobes + insula +
    cingulate), cerebellum (L/R/vermis), diencephalon (thalamus +
    hypothalamus), limbic (hippocampus, amygdala, striatum, NAc, GP),
    brainstem (midbrain/pons/medulla + basal forebrain), and the corpus
    callosum.  A `Connectome` wires these regions with thalamic,
    callosal, cortico-cerebellar, dopaminergic, noradrenergic,
    serotonergic, cholinergic, and limbic bundles.

  * v8 Aurora-Sapiens layer: the anatomical wiring is *dynamically*
    biological.  The four aminergic bundles modulate per-region
    plasticity (`NeuromodulatorBus`); a `MetabolicBudget` enforces the
    ~2 % activation ceiling; a thalamic salience router boosts regions
    whose declared modalities appear in the current observation dict;
    interhemispheric coherence is exposed as a diagnostic.

  * v9 Aurora-Cogitans layer: the cognitive additions that address the
    specific failure modes of contemporary LLMs.
        - EpistemicGate         : honest 'I don't know' via σ²/precision
        - ConstitutionalKernel  : SHA-256-audited immutable value core
        - WorkingMemory         : Cowan-4 capacity bottleneck with decay
        - SelfModel             : metacognitive summary + revision log
        - OscillatorBus         : δ/θ/α/β/γ phase clock w/ PAC coupling

  * v10 Aurora-Manus layer: the efferent / agentic additions that close
    the last big gap left by v3–v9 — the organism could reason and
    refuse but could not act on the world in a way its self-model
    could verify.
        - MotorCortex           : schema-checked tool dispatch + efference copy
        - AgencyMonitor         : Frith-Blakemore comparator; sense of agency
        - SomatosensoryField    : body schema + haptic touch trace
        - IntrinsicDrive        : mastery + novelty + competence reward
        - MirrorSystem          : observation-to-execution mapping
"""
from .config import AuroraOmniaConfig, AuroraPrimeConfig, SCALE_TIERS
from .organism import AuroraOmnia, AuroraPrime
from .anatomy import (
    BrainAtlas, build_human_atlas, Connectome, Cerebrum,
    NeuromodulatorBus, MetabolicBudget,
)
from .cogitans import (
    EpistemicGate,
    ConstitutionalKernel, ConstitutionalPrinciple,
    WorkingMemory, WMSlot,
    SelfModel, BeliefRevision,
    OscillatorBus, DEFAULT_BANDS_HZ,
)
from .manus import (
    MotorCortex, ToolSpec, EfferenceCopy, MotorReturn,
    AgencyMonitor, AgencyReport,
    SomatosensoryField,
    IntrinsicDrive, IntrinsicSample,
    MirrorSystem, MirrorMatch,
)

__all__ = [
    "AuroraOmnia", "AuroraOmniaConfig",
    "AuroraPrime", "AuroraPrimeConfig",
    "SCALE_TIERS",
    "BrainAtlas", "build_human_atlas", "Connectome", "Cerebrum",
    "NeuromodulatorBus", "MetabolicBudget",
    # v9
    "EpistemicGate",
    "ConstitutionalKernel", "ConstitutionalPrinciple",
    "WorkingMemory", "WMSlot",
    "SelfModel", "BeliefRevision",
    "OscillatorBus", "DEFAULT_BANDS_HZ",
    # v10
    "MotorCortex", "ToolSpec", "EfferenceCopy", "MotorReturn",
    "AgencyMonitor", "AgencyReport",
    "SomatosensoryField",
    "IntrinsicDrive", "IntrinsicSample",
    "MirrorSystem", "MirrorMatch",
]
__version__ = "10.0.0"
