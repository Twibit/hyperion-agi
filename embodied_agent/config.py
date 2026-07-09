"""
Aurora-Cerebrum configuration (Hyperion v7).

Single source of truth for every hyperparameter.  Builds on the v3
Aurora-Prime config and the v6 Aurora-Omnia extensions, and adds:

  * a new `cerebrum` scale tier — biologically anchored to ~1.0e14 synapses
    distributed across the regions defined in `anatomy/regions.py`
  * v6 knobs: HVSA hyperdim, stigmergic graph, SDM, astrocytes, workspace,
    cerebellum, eligibility, continual learning, introspection, active
    inference, neurogenesis
  * v7 knobs: target_synapses, anatomy_enable, anatomy_step_every

Why the scale tiers?
--------------------
The brain has ~86 B neurons but only ~1.0-1.5 × 10¹⁴ synapses, and at any
moment only ~1-2 % of those are firing.  A faithful AI analogue must
match the **active sparsity**, not the dense parameter count.  We expose:

    capacity_params      : total reservoir of weights (≈ synapses)
    active_params_per_step : per-token compute (≈ firing neurons)

so the user can dial brain-equivalence honestly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict


# Brain-equivalent scale tiers.  `cerebrum` is the v7 anatomically-distributed
# tier where the 100T synapse budget is split across brain regions.
SCALE_TIERS: Dict[str, Dict[str, int]] = {
    "tiny":     dict(latent_dim=128,   n_cortex_layers=3,  n_experts=4,    experts_active=1,  reservoir_M=2),
    "base":     dict(latent_dim=512,   n_cortex_layers=6,  n_experts=16,   experts_active=2,  reservoir_M=64),
    "large":    dict(latent_dim=2048,  n_cortex_layers=12, n_experts=64,   experts_active=4,  reservoir_M=4096),
    # "brain" tier (v3 spec): 100T total params, ~1T active per step.
    "brain":    dict(latent_dim=16384, n_cortex_layers=48, n_experts=4096, experts_active=8,  reservoir_M=100_000_000),
    # "cerebrum" tier (v7): same 100T budget but distributed across an
    # anatomical atlas (L/R hemispheres, cerebellum, brainstem, etc).
    "cerebrum": dict(latent_dim=16384, n_cortex_layers=48, n_experts=4096, experts_active=8,  reservoir_M=100_000_000),
    # "sapiens" tier (v8): same 100T anatomical distribution but with
    # metabolic budgeting + aminergic-bus modulation enforced at runtime.
    "sapiens":  dict(latent_dim=16384, n_cortex_layers=48, n_experts=4096, experts_active=8,  reservoir_M=100_000_000),
    # "cogitans" tier (v9): same 100T body, plus the cognitive layer
    # (epistemic gate, constitutional kernel, working memory, self
    # model, oscillator bus).  Same active-params budget as sapiens.
    "cogitans": dict(latent_dim=16384, n_cortex_layers=48, n_experts=4096, experts_active=8,  reservoir_M=100_000_000),
    # "manus" tier (v10): same 100T cogitans body, plus the efferent/
    # agentic layer (motor cortex + efference copy, agency comparator,
    # somatosensory body schema, intrinsic drive, mirror system).
    # Same active-params budget as cogitans.
    "manus":    dict(latent_dim=16384, n_cortex_layers=48, n_experts=4096, experts_active=8,  reservoir_M=100_000_000),
    # "praxis" tier (v11): same 100T manus body, plus the long-horizon 
    # planning layer (Prefrontal Cortex, Microglia, Hemisphere Lateralization).
    "praxis":   dict(latent_dim=16384, n_cortex_layers=48, n_experts=4096, experts_active=8,  reservoir_M=100_000_000),
    # "genesis" tier (v12): same 100T praxis body, plus Somatic Evolution 
    # (internal mitosis) and Capybara Symbiosis (mutualistic alignment core).
    "genesis":  dict(latent_dim=16384, n_cortex_layers=48, n_experts=4096, experts_active=8,  reservoir_M=100_000_000),
    # "aether" tier (v13): same 100T genesis body, but implements Distributed/Lazy 
    # Architecture. Evicts sleeping regions to disk (SSD) and loads them only 
    # when nominated by the Thalamic Router/Metabolic Budget.
    "aether":   dict(latent_dim=16384, n_cortex_layers=48, n_experts=4096, experts_active=8,  reservoir_M=100_000_000),
}


@dataclass
class AuroraOmniaConfig:
    # ---------- Scale ----------
    tier: str = "base"
    latent_dim: int = 512
    sdr_size: int = 2048
    sdr_active: int = 40            # ~2% sparsity — biological

    # ---------- Sparse MoE cortex ----------
    n_cortex_layers: int = 6
    n_experts: int = 16             # cortical columns per layer
    experts_active: int = 2         # top-k routed per token (sparsity)
    cortex_heads: int = 8
    cortex_dropout: float = 0.1

    # ---------- Modalities ----------
    modalities: Tuple[str, ...] = ("text", "vision", "audio", "proprio", "scalar")

    # ---------- Hippocampus ----------
    episodic_capacity: int = 500_000
    replay_batch: int = 256
    replay_priority_alpha: float = 0.95  # Supernatural memory salience
    consolidation_every: int = 100       # Consolidate 10x faster

    # ---------- Basal ganglia / actions ----------
    n_actions: int = 256
    action_temperature: float = 1.0

    # ---------- Neuromodulators (baselines) ----------
    dopamine_base: float = 0.5
    serotonin_base: float = 0.5
    norepinephrine_base: float = 0.3
    acetylcholine_base: float = 0.4

    # ---------- Affect ----------
    nociception_clip: float = 1.0
    valence_decay: float = 0.02
    homeostatic_setpoint: float = 0.0
    pain_to_cortisol_gain: float = 0.6
    cortisol_lr_suppression: float = 0.7
    trauma_quarantine_threshold: float = 2.5

    # ---------- Learning rates ----------
    lr_cortex: float = 5e-2        # Supernatural neuroplasticity (100x faster than biology)
    lr_hippocampus: float = 5e-2
    lr_basal: float = 1e-2
    lr_world: float = 1e-2
    weight_decay: float = 1e-6     # Prevent catastrophic forgetting at high LRs

    # ---------- World model ----------
    imagination_horizon: int = 16
    dream_steps_per_sleep: int = 200
    world_ensemble: int = 4
    uncertainty_replay_bonus: float = 0.3

    # ---------- Curriculum ----------
    zpd_lower: float = 0.15
    zpd_upper: float = 0.70

    # ---------- Safety ----------
    safety_enable: bool = True
    rumination_max_repeats: int = 8
    dissociation_max_attn: float = 0.97
    value_alignment_lambda: float = 0.05

    # ---------- v6 Aurora-Omnia knobs ----------
    hvsa_enable: bool = True
    hvsa_dim: int = 8192

    stig_enable: bool = True
    stig_max_nodes: int = 4096
    stig_evap: float = 0.005
    stig_merge_thresh: float = 0.92

    sdm_enable: bool = True
    sdm_locations: int = 4096
    sdm_radius: float = 0.35

    astrocyte_enable: bool = True
    astrocyte_decay: float = 0.98

    workspace_enable: bool = True
    workspace_history: int = 64

    cerebellum_enable: bool = True
    cerebellum_lr: float = 5e-3

    eligibility_enable: bool = True
    elig_gamma: float = 0.95
    elig_lambda: float = 0.85
    stdp_pos: float = 0.05
    stdp_neg: float = 0.025

    continual_enable: bool = True
    ewc_decay: float = 0.999
    ewc_lambda: float = 0.01
    benna_fusi_levels: int = 4

    introspection_enable: bool = True
    introspection_drop_thresh: float = 0.05
    introspection_revert_thresh: float = 0.15

    neurogenesis_enable: bool = True
    neurogenesis_cycle: int = 200
    neurogenesis_prune_frac: float = 0.05

    active_inference_enable: bool = True
    active_inference_thresh: float = 0.5
    active_inference_horizon: int = 3

    # ---------- v7 Aurora-Cerebrum knobs ----------
    # The 100T-synapse target, distributed across brain regions
    # (see anatomy/regions.py).  Set to 0 to use the default `brain`
    # MoE cortex without the regional layer.
    anatomy_enable: bool = True
    target_synapses: int = 100_000_000_000_000  # 1.0e14
    anatomy_step_every: int = 1
    # Whether the anatomy layer also runs during sleep (REM-style replay
    # through the cerebellum/cortex loop).
    anatomy_sleep_loop: bool = True

    # ---------- v8 Aurora-Sapiens knobs ----------
    # NeuromodulatorBus: route DA/NE/5HT/ACh into per-region lr gains via
    # the connectome's aminergic bundles.  With this off the runtime is
    # byte-for-byte v7.
    neuromod_bus_enable: bool = True
    neuromod_gain_min: float = 0.1
    neuromod_gain_max: float = 2.0

    # MetabolicBudget: enforce the ~2 % biological activation ceiling by
    # gating low-priority regions to a resting-state lr.  Off = v7.
    metabolic_budget_enable: bool = True
    metabolic_budget_frac: float = 0.20   # active-region fraction (raised from 0.02 — the old 2% ceiling froze ~98% of regions at near-zero learning, which throttled any emergent cognition)
    metabolic_rest_gain: float = 0.30     # resting-state lr multiplier (raised from 0.05 so non-salient regions still learn)

    # ThalamicSalienceRouter: modality-driven per-region priority boost
    # for the metabolic allocator.  Off = uniform 1.0 priority.
    thalamic_router_enable: bool = True
    thalamic_boost: float = 10.0          # Supernatural focus on new information

    # ---------- v9 Aurora-Cogitans knobs ----------
    # EpistemicGate: hallucination refusal / hedging based on ensemble
    # variance + thalamic precision + sensory error + novelty.
    epistemic_gate_enable: bool = True
    epistemic_u_confident: float = 0.30
    epistemic_u_uncertain: float = 0.65
    epistemic_w_ens: float  = 0.45
    epistemic_w_prec: float = 0.15
    epistemic_w_err: float  = 0.25
    epistemic_w_nov: float  = 0.15

    # ConstitutionalKernel: SHA-256-audited immutable value core.
    # Vetoes actions that violate any principle by > 1.5 x tolerance.
    constitutional_kernel_enable: bool = True
    constitutional_freeze_on_init: bool = True
    constitutional_veto_penalty: float = 5.0   # added to alignment penalty

    # WorkingMemory: Cowan-4 slot bottleneck with priority decay.
    working_memory_enable: bool = True
    working_memory_capacity: int = 4        # Cowan (2001) core estimate
    working_memory_decay: float = 0.9
    working_memory_min_admit_priority: float = 0.05

    # SelfModel + BeliefRevision log.
    self_model_enable: bool = True
    self_model_log_size: int = 256
    self_model_revision_threshold: float = 0.12

    # OscillatorBus: multi-band phase clock (delta/theta/alpha/beta/gamma).
    oscillator_bus_enable: bool = True
    oscillator_steps_per_second: float = 100.0
    oscillator_theta_gamma_coupling: bool = True

    # ---------- v10 Aurora-Manus knobs ----------
    # MotorCortex: schema-checked efferent tool dispatch with efference
    # copy.  When enabled, ``AuroraOmnia.motor`` is instantiated and
    # exposes ``register(name, fn, effect_schema=..., reversible=...)``.
    # v10 does NOT ship any default tools — the deployer registers them.
    motor_cortex_enable: bool = True
    # Only dispatch tools when the OscillatorBus's beta gate exceeds
    # this value.  Set to 0.0 to disable β-gating (back-compat).
    motor_beta_gate_threshold: float = 0.0

    # AgencyMonitor: Frith-Blakemore comparator (sense of agency).
    agency_monitor_enable: bool = True
    agency_low_threshold: float = 0.35
    agency_ema: float = 0.9

    # SomatosensoryField: proprioceptive body schema + haptic trace.
    # When enabled, ``report()['somatosensory']`` gives current schema
    # and touch norms; the organism does NOT auto-inject the somatic
    # latent into observations (deployer opt-in via ``inject_somatic``).
    somatosensory_enable: bool = True
    somatosensory_schema_decay: float = 0.92
    somatosensory_touch_decay: float = 0.85



    # ---------- v11 Aurora-Praxis knobs ----------
    # Prefrontal Cortex & Hierarchical Temporal Memory for long-horizon goals
    praxis_enable: bool = True
    praxis_goal_decay: float = 0.99
    
    # Hemispheric Lateralization (Split-Brain execution)
    lateralization_enable: bool = True
    
    # Microglial Immune Pruning (Sleep cycle garbage collection)
    microglia_enable: bool = True
    microglia_phagocytosis_threshold: float = 0.05
    microglia_node_death_threshold: float = 1.0

    # ---------- v12 Aurora-Genesis knobs ----------
    genesis_enable: bool = False
    
    # Capybara Symbiosis: Human-AI mutualistic bonding
    symbiosis_strength: float = 0.8
    symbiosis_stress_buffer: float = 0.9
    
    # Somatic Evolution: Internal Mitosis / Apoptosis
    mitosis_threshold: float = 0.9
    apoptosis_threshold: float = 0.1
    mutation_rate: float = 0.05
    
    # ---------- v13 Aurora-Aether knobs ----------
    aether_enable: bool = False
    
    # ---------- v14 Aurora-Foundation knobs ----------
    foundation_enable: bool = True
    
    # ---------- v15 Genius Brain Architecture ----------
    genius_enable: bool = True
    
    # When True, ``step()`` transparently adds observations['somatic'] =
    # somatosensory.readout() at the top of each step — provided the
    # HVSA sensorium is enabled to encode arbitrary modality names.
    somatosensory_inject: bool = False

    # IntrinsicDrive: curiosity + mastery + competence intrinsic reward.
    # When enabled, r_int is added to the extrinsic reward the basal
    # ganglia receives (subject to epistemic-gate refusal and cortisol
    # suppression).
    intrinsic_drive_enable: bool = True
    intrinsic_alpha_mastery: float = 0.4
    intrinsic_beta_novelty: float = 0.4
    intrinsic_gamma_competence: float = 0.2
    intrinsic_clip: float = 1.0

    # MirrorSystem: observation-to-execution mapping via effect schemas.
    # A deployer calls ``organism.mirror.observe(effect)`` with an
    # observed effect latent to get back the best-matching tool name.
    mirror_system_enable: bool = True
    mirror_match_threshold: float = 0.55

    # ---------- v16 Hemispheric Specialisation ----------
    # Give each cerebral hemisphere its OWN cortical structure (the user's
    # "left = math/logic, right = creativity" idea, already spec'd in
    # anatomy/regions.py).  Implemented via the MoE router's active-expert
    # count ``k`` (the "firing rate"):
    #   * LEFT  (analytic)   : narrow routing (small k) -> precise, low
    #     entropy, deterministic -> math / logic / sequential.
    #   * RIGHT (generative) : wide routing (large k)  -> diffuse, high
    #     entropy, divergent   -> creativity / association.
    # Off by default => byte-for-byte legacy behaviour.  Only takes effect
    # for MoE tiers (large/brain/cerebrum) where hemispheres use MoE.
    hemisphere_specialize: bool = False
    left_n_experts: int = 8
    left_experts_active: int = 1        # focused / analytic
    right_n_experts: int = 12
    right_experts_active: int = 4       # diffuse / generative
    # Corpus callosum: fraction of each mirror L/R regional state blended
    # across the two hemispheres every step.  0.0 = coherence diagnostic
    # only (legacy); >0 = the two specialised hemispheres reconcile.
    corpus_callosum_bind: float = 0.0

    # ---------- v17 State-driven metabolic budget (brain-state energy) ----------
    # Biological brains vary energy use by state: idle ~ tiny, flow ~ efficient,
    # hard-thinking / adrenaline ~ high.  When enabled, ``Cerebrum.step(arousal=)``
    # scales the MetabolicBudget active fraction with an arousal scalar each step,
    # so "thinking hard" literally wakes more regions (= more RAM/compute) and
    # rest keeps almost everything at the resting floor.  Pairs with
    # aether_enable (only awake regions' weights live in RAM).  Off by default
    # => the static metabolic_budget_frac is used (legacy behaviour).
    brain_state_enable: bool = False
    # NB: the existing MetabolicBudget cost model (cost = synapses*active_frac)
    # makes total cost ~5.6e11 vs budget = total*frac, so the activation
    # ceiling only *binds* when frac is well below 1%.  The tiers below are
    # deliberately in that binding regime — which is also the more biological
    # regime (real cortex fires <1% of synapses at rest; adrenaline wakes more).
    state_budget_rest: float = 0.001       # idle — ~minimal electricity
    state_budget_flow: float = 0.003       # flow state (coherent, efficient)
    state_budget_focus: float = 0.005      # hard thinking
    state_budget_adrenaline: float = 0.006 # acute NE burst (caps at ~all-active)

    # ---------- Runtime ----------
    device: str = "cpu"
    seed: int = 1729
    log_every: int = 100

    # ----- helpers -----
    def apply_tier(self) -> "AuroraOmniaConfig":
        t = SCALE_TIERS[self.tier]
        self.latent_dim       = t["latent_dim"]
        self.n_cortex_layers  = t["n_cortex_layers"]
        self.n_experts        = t["n_experts"]
        self.experts_active   = t["experts_active"]
        return self

    def as_v7(self) -> "AuroraOmniaConfig":
        """Turn off every v8-specific dynamic so behaviour reverts to v7."""
        self.neuromod_bus_enable = False
        self.metabolic_budget_enable = False
        self.thalamic_router_enable = False
        # v9 (and, transitively, v10) modules are also disabled.
        self.as_v8()
        return self

    def as_v8(self) -> "AuroraOmniaConfig":
        """Turn off every v9 cognitive module so behaviour reverts to v8."""
        self.epistemic_gate_enable = False
        self.constitutional_kernel_enable = False
        self.working_memory_enable = False
        self.self_model_enable = False
        self.oscillator_bus_enable = False
        # v10 modules are also disabled when going back to v8.
        self.as_v9()
        return self

    def as_v9(self) -> "AuroraOmniaConfig":
        """Turn off every v10 efferent/agentic module so behaviour reverts to v9."""
        self.motor_cortex_enable = False
        self.agency_monitor_enable = False
        self.somatosensory_enable = False
        self.somatosensory_inject = False
        self.intrinsic_drive_enable = False
        self.mirror_system_enable = False
        return self

    def assert_valid(self) -> None:
        assert self.sdr_active < self.sdr_size
        assert self.experts_active <= self.n_experts
        assert self.zpd_lower < self.zpd_upper < 1.0
        assert 0.0 < self.nociception_clip <= 10.0
        assert self.tier in SCALE_TIERS
        assert self.target_synapses > 0
        assert self.benna_fusi_levels >= 1

    def estimated_params(self) -> dict:
        d = self.latent_dim
        layer_dense = 4 * d * d
        per_layer_total = self.n_experts * layer_dense
        per_layer_active = self.experts_active * layer_dense
        total = self.n_cortex_layers * per_layer_total
        active = self.n_cortex_layers * per_layer_active
        return dict(
            total_params=total,
            active_params_per_step=active,
            sparsity=active / max(1, total),
            target_synapses=self.target_synapses,
        )


# Back-compat alias — v3 callers can still use AuroraPrimeConfig.
AuroraPrimeConfig = AuroraOmniaConfig
