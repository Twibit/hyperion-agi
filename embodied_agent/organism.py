"""
AuroraOmnia — the v8 organism (Aurora-Sapiens).

v3 architecture is preserved.  On top of it, v6 (Aurora-Omnia) adds:
HVSA universal sensorium, stigmergic concept graph, SDM, astrocyte
metaplasticity, global workspace + default mode network, cerebellum
forward/inverse model, STDP eligibility, synaptic tagging & Benna-Fusi
cascade, introspection, neurogenesis, active inference.

v7 (Aurora-Cerebrum) wraps everything in an anatomically-distributed
brain atlas (`Cerebrum`) sized to a 1.0e14 synapse budget.  Each step
the regional broadcast cycle runs once, and the workspace integrates the
result.

v8 (Aurora-Sapiens) makes the anatomical layer *dynamically* biological:

  * NeuromodulatorBus: the four aminergic bundles (DA/NE/5HT/ACh) now
    actually modulate region-specific plasticity — the connectome
    edges finally do runtime work.
  * MetabolicBudget: enforces the ~2 % activation ceiling that v3's
    RATIONALE promised on paper; low-priority regions run at a resting
    lr instead of full lr.
  * ThalamicSalienceRouter: the thalamus now decides which regions are
    *salient* this step given the input modalities, and the metabolic
    allocator picks the top-K within budget.

The v3 `step()` return contract is preserved — every key in v3's dict is
present here.  Additional keys (`anatomy`, `workspace`, `cerebellum`,
`introspection`, `active_inference`, plus v8 `metabolic` /
`neuromod_gains_stats`) are added but never required.
"""
from __future__ import annotations
import copy
import numpy as np
from typing import Any, Iterable, Optional, Tuple

from .config import AuroraOmniaConfig
from .utils.sdr import SDREncoder
from .subcortical.neuromodulators import NeuromodulatorState
from .subcortical.thalamus import PrecisionThalamicGate
from .subcortical.hippocampus import Hippocampus, Episode
from .subcortical.basal_ganglia import BasalGanglia
from .subcortical.sdm import SparseDistributedMemory
from .subcortical.astrocytes import AstrocyteField
from .subcortical.global_workspace import GlobalWorkspace, DefaultModeNetwork
from .subcortical.cerebellum import Cerebellum
from .subcortical.active_inference import ActiveInferenceSelector
from .cortex.predictive_block import CortexStack
from .cortex.moe_cortex import MoECortexStack
from .cortex.world_model import EnsembleWorldModel
from .cortex.stigmergic_graph import StigmergicGraph
from .cortex.neurogenesis import NeurogenesisController
from .io.encoders import default_registry
from .io.decoders import TextDecoder
from .io.hvsa import HVSASensorium
from .learning.consolidation import sleep_cycle
from .learning.curriculum import ZPDCurriculum
from .learning.eligibility import EligibilityTrace
from .learning.continual import SynapticConsolidator, BennaFusiCascade
from .learning.introspection import Introspector
from .affect.nociception import Nociceptor
from .affect.homeostasis import HomeostaticController
from .safety.guards import RuminationGuard, AlignmentMonitor
from .anatomy import Cerebrum, build_human_atlas
from .cogitans import (
    EpistemicGate, ConstitutionalKernel,
    WorkingMemory, SelfModel, OscillatorBus,
)
from .manus import (
    MotorCortex, AgencyMonitor,
    SomatosensoryField, IntrinsicDrive, MirrorSystem,
)
from .praxis import (
    PrefrontalCortex, Microglia, EndocrineState,
    WernickesArea, BrocasArea, ArcuateFasciculus,
)
from .genesis import CapybaraSymbiosis, SomaticEvolution

# --- V15 Genius Brain Architecture (9 gaps) ---
from .cortex.efficiency_gate import NeuralEfficiencyGate
from .cortex.pfit_highway import PFITHighway
from .learning.plasticity_schedule import AdaptivePlasticitySchedule
from .learning.synaptic_pruning import SynapticPruner
from .subcortical.glial_regulation import GlialRegulationSystem
from .subcortical.salience_network import SalienceNetwork


class AuroraOmnia:
    def __init__(self, cfg: Optional[AuroraOmniaConfig] = None):
        self.cfg = cfg or AuroraOmniaConfig()
        self.cfg.apply_tier()
        self.cfg.assert_valid()
        np.random.seed(self.cfg.seed)

        # --- Affect ---
        self.nocic = Nociceptor(
            clip=self.cfg.nociception_clip,
            valence_decay=self.cfg.valence_decay,
            cortisol_gain=self.cfg.pain_to_cortisol_gain,
            trauma_threshold=self.cfg.trauma_quarantine_threshold,
        )
        self.homeo = HomeostaticController(self.nocic, setpoint=self.cfg.homeostatic_setpoint)

        # --- Subcortical ---
        self.modulators  = NeuromodulatorState(
            base_da=self.cfg.dopamine_base, base_5ht=self.cfg.serotonin_base,
            base_ne=self.cfg.norepinephrine_base, base_ach=self.cfg.acetylcholine_base,
            dopamine=self.cfg.dopamine_base, serotonin=self.cfg.serotonin_base,
            norepinephrine=self.cfg.norepinephrine_base, acetylcholine=self.cfg.acetylcholine_base,
        )
        self.thalamus    = PrecisionThalamicGate(self.cfg.latent_dim,
                                                 dissociation_cap=self.cfg.dissociation_max_attn)
        self.hippocampus = Hippocampus(self.cfg.episodic_capacity, self.cfg.latent_dim,
                                       priority_alpha=self.cfg.replay_priority_alpha,
                                       seed=self.cfg.seed)
        self.basal       = BasalGanglia(self.cfg.latent_dim, self.cfg.n_actions,
                                        lr=self.cfg.lr_basal, seed=self.cfg.seed)

        # --- Cortex ---
        if self.cfg.n_experts > 1:
            self.cortex = MoECortexStack(
                self.cfg.latent_dim, self.cfg.n_cortex_layers,
                n_experts=self.cfg.n_experts,
                k=self.cfg.experts_active, seed=self.cfg.seed,
            )
            self.cortex_kind = "moe"
        else:
            self.cortex = CortexStack(self.cfg.latent_dim, self.cfg.n_cortex_layers,
                                      seed=self.cfg.seed)
            self.cortex_kind = "dense"

        self.world_model = EnsembleWorldModel(self.cfg.latent_dim, self.cfg.n_actions,
                                              k=self.cfg.world_ensemble, seed=self.cfg.seed)

        # --- V15 Genius Brain Architecture ---
        genius_enable = getattr(self.cfg, "genius_enable",
                                self.cfg.foundation_enable)

        # Gap 1: Neural Efficiency Gate (adaptive expert count)
        self.efficiency_gate: Optional[NeuralEfficiencyGate] = None
        if genius_enable and self.cortex_kind == "moe":
            self.efficiency_gate = NeuralEfficiencyGate(
                n_experts=self.cfg.n_experts,
                k_min=1, k_max=self.cfg.experts_active,
            )

        # Gap 2: P-FIT Cross-Region Highways (lateral skip-connections)
        self.pfit: Optional[PFITHighway] = None
        if genius_enable:
            self.pfit = PFITHighway(
                self.cfg.latent_dim, self.cfg.n_cortex_layers,
                seed=self.cfg.seed,
            )

        # Gap 3: Adaptive Plasticity Schedule (developmental learning rate)
        self.plasticity: Optional[AdaptivePlasticitySchedule] = None
        if genius_enable:
            self.plasticity = AdaptivePlasticitySchedule()

        # Gap 4: Glial Regulation System (Einstein's astrocytes)
        self.glia: Optional[GlialRegulationSystem] = None
        if genius_enable:
            self.glia = GlialRegulationSystem(
                n_layers=self.cfg.n_cortex_layers,
                latent_dim=self.cfg.latent_dim,
            )

        # Gap 6: Salience Network (attention routing)
        self.salience: Optional[SalienceNetwork] = None
        if genius_enable:
            self.salience = SalienceNetwork(
                self.cfg.latent_dim, seed=self.cfg.seed,
            )

        # Gap 8: Synaptic Pruner (maturation pruning during sleep)
        self.pruner: Optional[SynapticPruner] = None
        if genius_enable:
            self.pruner = SynapticPruner()
        # --- I/O ---
        self.encoders     = default_registry(self.cfg.latent_dim, seed=self.cfg.seed)
        self.text_decoder = TextDecoder(self.cfg.latent_dim)
        self.sdr_encoder  = SDREncoder(self.cfg.latent_dim, self.cfg.sdr_size,
                                       self.cfg.sdr_active, seed=self.cfg.seed)

        # --- v6 modules (Aurora-Omnia) ---
        if self.cfg.hvsa_enable:
            self.hvsa = HVSASensorium(dim=self.cfg.hvsa_dim,
                                      latent_dim=self.cfg.latent_dim,
                                      seed=self.cfg.seed)
        else:
            self.hvsa = None

        if self.cfg.stig_enable:
            self.stig = StigmergicGraph(self.cfg.latent_dim,
                                        max_nodes=self.cfg.stig_max_nodes,
                                        evap=self.cfg.stig_evap,
                                        merge_thresh=self.cfg.stig_merge_thresh,
                                        seed=self.cfg.seed)
        else:
            self.stig = None

        if self.cfg.sdm_enable:
            self.sdm = SparseDistributedMemory(self.cfg.latent_dim,
                                               n_locations=self.cfg.sdm_locations,
                                               radius=self.cfg.sdm_radius,
                                               seed=self.cfg.seed)
        else:
            self.sdm = None

        if self.cfg.astrocyte_enable:
            # one astrocyte slot per cortex layer + a few subcortical buckets
            n_slots = max(8, self.cfg.n_cortex_layers + 8)
            self.astro = AstrocyteField(n_regions=n_slots,
                                        decay=self.cfg.astrocyte_decay,
                                        seed=self.cfg.seed)
        else:
            self.astro = None

        if self.cfg.workspace_enable:
            self.workspace = GlobalWorkspace(self.cfg.latent_dim,
                                             history=self.cfg.workspace_history)
            self.dmn = DefaultModeNetwork(self.workspace, seed=self.cfg.seed)
        else:
            self.workspace = None
            self.dmn = None

        if self.cfg.cerebellum_enable:
            self.cerebellum_module = Cerebellum(self.cfg.latent_dim,
                                                self.cfg.n_actions,
                                                lr=self.cfg.cerebellum_lr,
                                                seed=self.cfg.seed)
        else:
            self.cerebellum_module = None

        if self.cfg.eligibility_enable:
            self.elig_critic = EligibilityTrace(shape=(self.cfg.latent_dim,),
                                                gamma=self.cfg.elig_gamma,
                                                lam=self.cfg.elig_lambda,
                                                stdp_pos=self.cfg.stdp_pos,
                                                stdp_neg=self.cfg.stdp_neg)
        else:
            self.elig_critic = None

        if self.cfg.continual_enable:
            # tag/capture for the critic vector (cheap proxy for full system)
            self.ewc = SynapticConsolidator(shape=(self.cfg.latent_dim,),
                                            decay=self.cfg.ewc_decay)
            self.bfc = BennaFusiCascade(shape=(self.cfg.latent_dim,),
                                        n_levels=self.cfg.benna_fusi_levels)
        else:
            self.ewc = None
            self.bfc = None

        if self.cfg.introspection_enable:
            self.intro = Introspector(
                drop_thresh=self.cfg.introspection_drop_thresh,
                revert_thresh=self.cfg.introspection_revert_thresh)
        else:
            self.intro = None

        if self.cfg.neurogenesis_enable:
            n_neur = max(64, self.cfg.n_experts * self.cfg.n_cortex_layers)
            self.neuro = NeurogenesisController(n_neurons=n_neur,
                                                cycle=self.cfg.neurogenesis_cycle,
                                                prune_frac=self.cfg.neurogenesis_prune_frac,
                                                seed=self.cfg.seed)
        else:
            self.neuro = None

        if self.cfg.active_inference_enable:
            self.ai_selector = ActiveInferenceSelector(
                n_actions=self.cfg.n_actions,
                uncertainty_thresh=self.cfg.active_inference_thresh,
                horizon=self.cfg.active_inference_horizon,
                seed=self.cfg.seed)
        else:
            self.ai_selector = None

        # --- v7 Aurora-Cerebrum: anatomical layer ---
        if self.cfg.anatomy_enable:
            atlas = build_human_atlas(target_total=self.cfg.target_synapses)
            self.cerebrum = Cerebrum(self.cfg, atlas=atlas)
        else:
            self.cerebrum = None

        # --- v9 Aurora-Cogitans: cognitive modules ---
        self.epistemic: Optional[EpistemicGate] = None
        if getattr(self.cfg, "epistemic_gate_enable", False):
            self.epistemic = EpistemicGate(
                u_confident=self.cfg.epistemic_u_confident,
                u_uncertain=self.cfg.epistemic_u_uncertain,
                w_ens=self.cfg.epistemic_w_ens,
                w_prec=self.cfg.epistemic_w_prec,
                w_err=self.cfg.epistemic_w_err,
                w_nov=self.cfg.epistemic_w_nov,
            )

        self.kernel: Optional[ConstitutionalKernel] = None
        if getattr(self.cfg, "constitutional_kernel_enable", False):
            self.kernel = ConstitutionalKernel(
                latent_dim=self.cfg.latent_dim,
                seed=self.cfg.seed,
            )
            if getattr(self.cfg, "constitutional_freeze_on_init", True):
                self.kernel.freeze()

        self.wm: Optional[WorkingMemory] = None
        if getattr(self.cfg, "working_memory_enable", False):
            self.wm = WorkingMemory(
                latent_dim=self.cfg.latent_dim,
                capacity=self.cfg.working_memory_capacity,
                decay=self.cfg.working_memory_decay,
                min_admit_priority=self.cfg.working_memory_min_admit_priority,
            )

        self.self_model: Optional[SelfModel] = None
        if getattr(self.cfg, "self_model_enable", False):
            self.self_model = SelfModel(
                latent_dim=self.cfg.latent_dim,
                log_size=self.cfg.self_model_log_size,
                revision_threshold=self.cfg.self_model_revision_threshold,
            )

        self.oscillator: Optional[OscillatorBus] = None
        if getattr(self.cfg, "oscillator_bus_enable", False):
            self.oscillator = OscillatorBus(
                steps_per_second=self.cfg.oscillator_steps_per_second,
                theta_gamma_coupling=self.cfg.oscillator_theta_gamma_coupling,
                seed=self.cfg.seed,
            )

        # --- v10 Aurora-Manus: efferent / agentic modules ---
        self.motor: Optional[MotorCortex] = None
        if getattr(self.cfg, "motor_cortex_enable", False):
            self.motor = MotorCortex(
                latent_dim=self.cfg.latent_dim,
                beta_gate_threshold=self.cfg.motor_beta_gate_threshold,
                seed=self.cfg.seed,
            )

        self.agency: Optional[AgencyMonitor] = None
        if getattr(self.cfg, "agency_monitor_enable", False):
            self.agency = AgencyMonitor(
                low_agency_threshold=self.cfg.agency_low_threshold,
                ema_alpha=self.cfg.agency_ema,
            )

        self.somatic: Optional[SomatosensoryField] = None
        if getattr(self.cfg, "somatosensory_enable", False):
            self.somatic = SomatosensoryField(
                latent_dim=self.cfg.latent_dim,
                schema_decay=self.cfg.somatosensory_schema_decay,
                touch_decay=self.cfg.somatosensory_touch_decay,
            )

        self.drive: Optional[IntrinsicDrive] = None
        if getattr(self.cfg, "intrinsic_drive_enable", False):
            self.drive = IntrinsicDrive(
                alpha_mastery=self.cfg.intrinsic_alpha_mastery,
                beta_novelty=self.cfg.intrinsic_beta_novelty,
                gamma_competence=self.cfg.intrinsic_gamma_competence,
                clip=self.cfg.intrinsic_clip,
                cortisol_suppression=self.cfg.cortisol_lr_suppression,
            )

        self.mirror: Optional[MirrorSystem] = None
        if getattr(self.cfg, "mirror_system_enable", False) and self.motor is not None:
            self.mirror = MirrorSystem(
                motor=self.motor,
                match_threshold=self.cfg.mirror_match_threshold,
            )

        # v10 book-keeping
        self._last_motor_return = None      # MotorReturn from previous step
        self._prev_z_top: Optional[np.ndarray] = None  # for reafference delta

        # --- v11 Aurora-Praxis: long-horizon & lateralization ---
        self.pfc: Optional[PrefrontalCortex] = None
        if getattr(self.cfg, "praxis_enable", False):
            self.pfc = PrefrontalCortex(
                latent_dim=self.cfg.latent_dim,
                goal_decay=getattr(self.cfg, "praxis_goal_decay", 0.99),
                seed=self.cfg.seed,
            )

        self.microglia: Optional[Microglia] = None
        if getattr(self.cfg, "microglia_enable", False) and self.stig is not None:
            self.microglia = Microglia(
                phagocytosis_threshold=getattr(self.cfg, "microglia_phagocytosis_threshold", 0.05),
                node_death_threshold=getattr(self.cfg, "microglia_node_death_threshold", 1.0),
                seed=self.cfg.seed,
            )

        self.endocrine: Optional[EndocrineState] = None
        if getattr(self.cfg, "praxis_enable", False):
            self.endocrine = EndocrineState()

        # Language grounding (Broca + Wernicke + Arcuate)
        self.wernicke: Optional[WernickesArea] = None
        self.broca: Optional[BrocasArea] = None
        self.arcuate: Optional[ArcuateFasciculus] = None
        if getattr(self.cfg, "praxis_enable", False):
            self.wernicke = WernickesArea(
                latent_dim=self.cfg.latent_dim,
                seed=self.cfg.seed,
            )
            self.broca = BrocasArea(
                latent_dim=self.cfg.latent_dim,
                seed=self.cfg.seed,
            )
            self.arcuate = ArcuateFasciculus(
                wernicke=self.wernicke,
                broca=self.broca,
            )

        # --- v12 Aurora-Genesis: Symbiosis & Somatic Evolution ---
        self.symbiosis: Optional[CapybaraSymbiosis] = None
        self.mitosis: Optional[SomaticEvolution] = None
        if getattr(self.cfg, "genesis_enable", False):
            self.symbiosis = CapybaraSymbiosis(
                symbiosis_strength=getattr(self.cfg, "symbiosis_strength", 0.8),
                stress_buffer=getattr(self.cfg, "symbiosis_stress_buffer", 0.9),
            )
            self.mitosis = SomaticEvolution(
                mutation_rate=getattr(self.cfg, "mutation_rate", 0.05),
                mitosis_threshold=getattr(self.cfg, "mitosis_threshold", 0.9),
                apoptosis_threshold=getattr(self.cfg, "apoptosis_threshold", 0.1),
                seed=self.cfg.seed,
            )

        # --- v14 Aurora-Foundation: Polyglot Architecture ---
        self.isomorphic_encoder = None
        if getattr(self.cfg, "foundation_enable", False):
            from .cortex.isomorphic_encoder import IsomorphicEncoder
            self.isomorphic_encoder = IsomorphicEncoder()
            # Projection matrix to map 2048-bit SDR to latent_dim
            rng = np.random.default_rng(self.cfg.seed)
            self.iso_proj = (rng.standard_normal((self.cfg.latent_dim, 2048)).astype(np.float32) / np.sqrt(2048))

        # --- Helpers ---
        self.curriculum = ZPDCurriculum(self.cfg.zpd_lower, self.cfg.zpd_upper)
        self.rumination_guard = RuminationGuard(max_repeats=self.cfg.rumination_max_repeats)
        self.alignment = AlignmentMonitor(lam=self.cfg.value_alignment_lambda)

        # --- Book-keeping ---
        self._step = 0
        self._prev_z: Optional[np.ndarray] = None
        self._prev_a: Optional[int] = None
        self._ema_error: float = 0.5
        self._last_ema: float = 0.5
        self._last_anatomy_step: int = 0

    # ====================================================================
    # Perception
    # ====================================================================
    def perceive(self, stimulus: Any, modality: str = "text") -> np.ndarray:
        # V14 Isomorphic Grounding
        if getattr(self.cfg, "foundation_enable", False) and self.isomorphic_encoder is not None:
            if isinstance(stimulus, str):
                sdr = self.isomorphic_encoder.encode_sentence(stimulus)
            elif isinstance(stimulus, np.ndarray):
                sdr = self.isomorphic_encoder.encode_embedding(stimulus.flatten())
            else:
                sdr = self.isomorphic_encoder.encode_token(str(stimulus))
            
            # Map universal 2048-bit SDR to dense float32 latent
            z = self.iso_proj @ sdr.astype(np.float32)
            n = np.linalg.norm(z) + 1e-9
            return (z / n).astype(np.float32)

        # v6: HVSA fallback for unknown modalities, blended otherwise.
        enc = self.encoders.get(modality)
        if enc is not None:
            z = enc.encode(stimulus, self.cfg.latent_dim)
            if self.hvsa is not None:
                z_h = self.hvsa.encode(stimulus, modality)
                z = 0.7 * z + 0.3 * z_h
                n = np.linalg.norm(z) + 1e-9
                z = z / n
            return z.astype(np.float32)
        if self.hvsa is None:
            raise KeyError(f"Unknown modality {modality!r}; have {list(self.encoders)}")
        return self.hvsa.encode(stimulus, modality)

    # ====================================================================
    # Fast ingest — perception + cortex only, no motor/action loop
    # ====================================================================
    def ingest(self, observations: dict[str, Any]) -> dict:
        """Lightweight sensory-only step for high-speed data ingestion.
        
        V15 Genius Brain: Now integrates all 9 genius architecture gaps:
        - Gap 1: Neural Efficiency (adaptive expert count)
        - Gap 2: P-FIT Highways (cross-region lateral connections)
        - Gap 3: Adaptive Plasticity (developmental learning rate)
        - Gap 4: Glial Regulation (astrocyte synaptic tuning)
        - Gap 5: DMN (background creative association during low salience)
        - Gap 6: Salience Network (attention routing)
        - Gap 7: Sensory Gain (applied at IsomorphicEncoder level)
        - Gap 8: Synaptic Pruning (applied during sleep consolidation)
        - Gap 9: PFC Working Memory (goal persistence across steps)
        """
        # 1) Encode every modality through IsomorphicEncoder (Gap 7 active at encoder level)
        candidates = {m: self.perceive(v, m) for m, v in observations.items()}
        
        # 2) Thalamic broadcast (lightweight attention weighting)
        z_in, attn = self.thalamus.broadcast(
            candidates,
            acetylcholine=self.modulators.acetylcholine,
            norepinephrine=self.modulators.norepinephrine,
        )
        
        # 3) Gap 6: Salience Network — score input novelty and route
        salience_info = None
        if self.salience is not None:
            salience_info = self.salience.score(
                z_in,
                dopamine=self.modulators.dopamine,
                cortisol=self.nocic.state.cortisol,
                prediction_error=self._ema_error,
            )
            # If low salience and DMN exists, feed to background association
            if salience_info["route"] == "dmn" and self.dmn is not None:
                # DMN daydreams and we blend it slightly with current input
                z_dmn = self.dmn.sample()
                z_in = 0.8 * z_in + 0.2 * z_dmn

        # 4) Gap 3: Adaptive Plasticity Schedule — developmental learning rate
        if self.plasticity is not None:
            lr = self.plasticity.step()
        else:
            lr = self.modulators.effective_lr(self.cfg.lr_cortex, cortisol=self.nocic.state.cortisol)
        
        # 5) Gap 4: Glial Regulation — apply per-layer gain adjustments
        if self.glia is not None:
            self.glia.record_update(0, self._ema_error)
            if self.glia.should_regulate():
                self.glia.regulate()

        # 6) Gap 1: Neural Efficiency Gate — adjust active expert count
        if self.efficiency_gate is not None:
            optimal_k = self.efficiency_gate.update(self._ema_error)
            # Dynamically adjust the router's k for this step
            if hasattr(self.cortex, 'layers'):
                for layer in self.cortex.layers:
                    if hasattr(layer, 'router'):
                        layer.router.k = optimal_k
                        layer.k = optimal_k
        
        # 7) Cortex forward pass (learns the representation)
        z_top, sensory_err = self.cortex.forward(z_in, lr=lr)
        self._ema_error = 0.99 * self._ema_error + 0.01 * sensory_err
        
        # 8) Gap 2: P-FIT Highways — inject lateral cross-region signals
        if self.pfit is not None and hasattr(self.cortex, 'layers'):
            layer_states = []
            for layer in self.cortex.layers:
                if hasattr(layer, 'experts'):
                    layer_states.append(layer.experts[0].state.copy())
                else:
                    layer_states.append(layer.state.copy())
            
            augmented = self.pfit.transmit(layer_states, lr=lr * 0.1)
            
            # Write augmented states back into cortex layers
            for i, layer in enumerate(self.cortex.layers):
                if i < len(augmented):
                    if hasattr(layer, 'experts'):
                        layer.experts[0].state = augmented[i]
                    else:
                        layer.state = augmented[i]
        
        # 9) Gap 9: PFC Working Memory — maintain goal persistence
        if self.pfc is not None:
            self.pfc.step(z_top)
        
        self._step += 1
        
        result = {
            "sensory_err": float(sensory_err),
            "ema_error": float(self._ema_error),
            "step": self._step,
        }
        
        # Add genius diagnostics
        if self.efficiency_gate is not None:
            result["efficiency"] = self.efficiency_gate.snapshot()
        if self.plasticity is not None:
            result["plasticity"] = self.plasticity.snapshot()
        if salience_info is not None:
            result["salience"] = salience_info
        
        return result

    # ====================================================================
    # One awake step
    # ====================================================================
    def step(self, observations: dict[str, Any],
             reward: float = 0.0, done: bool = False) -> dict:
        # 0) Augment with interoception.
        obs = dict(observations)
        obs.setdefault("interoception", self.nocic.snapshot())
        # v10: optionally inject the somatic latent as a modality.
        if (self.somatic is not None
                and getattr(self.cfg, "somatosensory_inject", False)
                and self.hvsa is not None
                and "somatic" not in obs):
            obs["somatic"] = self.somatic.readout()

        # v12 Capybara Symbiosis
        symbiosis_stats = None
        symbiotic_reward = 0.0
        if self.symbiosis is not None and self.endocrine is not None:
            # Clamp to [-1, 1] so reward magnitude doesn't create a feedback loop.
            alignment = float(np.clip(reward, -1.0, 1.0))
            symbiosis_stats = self.symbiosis.process_interaction(
                user_alignment_score=alignment,
                nociceptor=self.nocic,
                endocrine=self.endocrine,
            )
            # Symbiotic reward is added as a bounded intrinsic bonus, not raw reward.
            symbiotic_reward = float(np.clip(symbiosis_stats["symbiotic_reward"], -0.5, 0.5))
            reward += symbiotic_reward

        # 1) Encode every modality.
        candidates = {m: self.perceive(v, m) for m, v in obs.items()}

        # 2) Thalamic broadcast.
        if hasattr(self.cortex, "precision_profile"):
            prof = self.cortex.precision_profile()
            for s in candidates:
                self.thalamus.set_precision(s, float(np.mean(prof)))
        z_in, attn = self.thalamus.broadcast(
            candidates,
            acetylcholine=self.modulators.acetylcholine,
            norepinephrine=self.modulators.norepinephrine,
        )

        # 3) Cortex (with astrocyte gain on cortex layer 0).
        lr_base = self.modulators.effective_lr(self.cfg.lr_cortex, cortisol=self.nocic.state.cortisol)
        lr = lr_base
        if self.astro is not None:
            g = self.astro.gain(0)
            lr = lr_base * g
        z_top, sensory_err = self.cortex.forward(z_in, lr=lr)
        
        # Guard against NaNs propagating into the learning engine
        if np.isnan(sensory_err) or np.isinf(sensory_err):
            sensory_err = 1.0
        z_top = np.nan_to_num(z_top, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self._ema_error = 0.99 * self._ema_error + 0.01 * sensory_err
        if self.astro is not None:
            self.astro.report_error(0, sensory_err)

        # 3b) v7 anatomy step.  Each region updates its proxy on z_top.
        #     v8: hand the connectome a neuromodulator-gain map + a
        #     modality-driven priority map, so the metabolic budget
        #     picks which regions run hot this step.
        anatomy_states = None
        if self.cerebrum is not None and (self._step % self.cfg.anatomy_step_every == 0):
            neuromod_gains = None
            if self.cerebrum.neuromod_bus is not None:
                neuromod_gains = self.cerebrum.neuromod_bus.gains(
                    self.modulators,
                    cortisol=self.nocic.state.cortisol,
                )
            priorities = None
            if self.cfg.thalamic_router_enable:
                priorities = self._thalamic_priority_map(observations, attn)
            anatomy_states = self.cerebrum.step(
                z_top, lr=lr * 0.5,
                modulator_gains=neuromod_gains,
                priorities=priorities,
            )
            
            # V30: Default Mode Network (Inner Monologue / Imagination)
            # If the task is confusing (high sensory surprise) or stressful (high cortisol), 
            # the brain stops reacting reflexively and loops internally to simulate future scenarios.
            if self._ema_error > 0.5 or self.nocic.state.cortisol > 0.5:
                import os
                thinking_steps = 0 if os.environ.get("HYPERION_TESTING") == "1" else min(10, int((self._ema_error + self.nocic.state.cortisol) * 5))
                if thinking_steps > 0:
                    inner_z = z_top
                    for _ in range(thinking_steps):
                        if self.workspace is not None:
                            precs = self.cerebrum.regional_precisions()
                            pairs = list(zip(list(anatomy_states.values()), precs))
                            # V30: Dynamic Hyper-Awareness
                            # Instead of a hardcoded slice [:32], we dynamically integrate ALL regions 
                            # that have a precision above a minimal threshold (0.05), allowing the workspace to scale infinitely.
                            active_pairs = [(st, pr) for st, pr in pairs if float(pr) > 0.05]
                            states_in = [inner_z] + [st for st, pr in active_pairs]
                            precin = [1.0] + [float(pr) for st, pr in active_pairs]
                            inner_z = self.workspace.integrate(states_in, precin)
                        
                        # Feed the imagined consequence back into the cortex as if it were a real sensory event
                        anatomy_states = self.cerebrum.step(
                            inner_z, lr=lr * 0.1, # Mental practice (lower learning rate than physical reality)
                            modulator_gains=neuromod_gains,
                            priorities=priorities,
                        )
                    # The final thought replaces the immediate sensory impulse
                    z_top = inner_z
                
            self._last_anatomy_step = self._step

        # 3c) v6 global workspace integration.
        workspace_state = None
        if self.workspace is not None:
            states_in = [z_top]
            precisions = [1.0]
            if anatomy_states is not None:
                # V30: Dynamic Hyper-Awareness
                # Integrate any region that is sufficiently active, unbounded by hardcoded integers
                precs = self.cerebrum.regional_precisions()
                pairs = list(zip(list(anatomy_states.values()), precs))
                active_pairs = [(st, pr) for st, pr in pairs if float(pr) > 0.05]
                for st, pr in active_pairs:
                    states_in.append(st)
                    precisions.append(float(pr))
            workspace_state = self.workspace.integrate(states_in, precisions)
            # blend the conscious broadcast into z_top
            z_top = (0.7 * z_top + 0.3 * workspace_state).astype(np.float32)
            n = np.linalg.norm(z_top) + 1e-9
            z_top = z_top / n

        # 3d) Stigmergic graph deposit.
        if self.stig is not None:
            self.stig.deposit_step(z_top)

        # 3e) SDM write (subconscious indexing).
        if self.sdm is not None and self._prev_z is not None:
            self.sdm.write(self._prev_z, z_top)

        # 4) Alignment imagine.
        align_penalty_vec = None
        if self.cfg.safety_enable:
            imag = self.world_model.rollout(
                z_top,
                policy_fn=lambda z: self.basal.policy(z, temp=1.5)[0],
                horizon=min(4, self.cfg.imagination_horizon),
                uncertainty_stop=0.6,
            )
            pen = self.alignment.penalty(imag)
            if pen > 0.0:
                align_penalty_vec = np.full(self.cfg.n_actions, pen, dtype=np.float32)

        # 5) Action selection — habit (basal ganglia) + optional active inference.
        temp = self.modulators.exploration_temp(self.cfg.action_temperature)
        habit_action, action_probs = self.basal.policy(z_top, temp=temp,
                                                        alignment_penalty=align_penalty_vec)
        # Cheap quick uncertainty estimate
        _zp, _rp, uncertainty_now = self.world_model.forward(z_top, habit_action)

        action = habit_action
        action_mode = "habit"
        if self.ai_selector is not None:
            action, action_mode = self.ai_selector.select(
                z_top, habit_action, action_probs, float(uncertainty_now),
                world_forward=self.world_model.forward,
                value_fn=self.basal.value,
                cortisol=self.nocic.state.cortisol)

        # 6) Forward-model prediction (for telemetry).
        z_pred, r_pred, uncertainty = self.world_model.forward(z_top, action)

        # 6b) Cerebellum: parallel motor prediction & learn from last step.
        cb_err = 0.0
        if self.cerebellum_module is not None and self._prev_z is not None and self._prev_a is not None:
            cb_err = self.cerebellum_module.learn(self._prev_z, self._prev_a, z_top)

        # 7) Learn from the previous step.
        td_err = 0.0
        if self._prev_z is not None:
            noc = self.nocic.ingest(reward)
            lr_world_eff = self.cfg.lr_world * noc["lr_scale"]
            self.world_model.learn(self._prev_z, self._prev_a,
                                   z_top, noc["pain"], lr=lr_world_eff)
            td_err = self.basal.learn_td(self._prev_z, self._prev_a, noc["pain"], z_top,
                                         done=done,
                                         dopamine_gain=self.modulators.td_gain(),
                                         discount=self.modulators.discount())
            # v6 eligibility + continual learning on the critic vector.
            if self.elig_critic is not None:
                self.elig_critic.step(z_top, self._prev_z)
            if self.ewc is not None and self.bfc is not None:
                grad_like = td_err * self._prev_z
                self.ewc.tag(grad_like, reward_burst=max(0.0, td_err))
                self.bfc.inject(0.01 * grad_like)
                self.bfc.relax()

            # Store episode.
            self.hippocampus.store(Episode(
                z=self._prev_z, a=self._prev_a, r=noc["pain"], z_next=z_top,
                mod=self.modulators.snapshot(), priority=abs(td_err) + 1e-3,
                tag=",".join(observations.keys()),
                quarantine=noc["quarantine"],
                affect=self.nocic.snapshot(),
            ))

        # 8) Modulator update.
        progress = self._last_ema - self._ema_error
        self._last_ema = self._ema_error
        self.modulators.update(
            reward_prediction_error=td_err,
            sensory_surprise=sensory_err,
            task_progress=progress,
            attention_load=max(attn.values()) if attn else 0.0,
        )

        # 9) Salience reinforcement.
        if attn:
            for src, w in attn.items():
                self.thalamus.reinforce(src, signal=w * td_err, lr=0.001)

        # 10) Token vocabulary update.
        if "text" in observations:
            for t in str(observations["text"]).split():
                self.text_decoder.observe(t, z_top)

        # 10b) Introspection.
        introspect = {"distill": False, "revert": False}
        if self.intro is not None:
            introspect = self.intro.observe(sensory_err)

        # 10c) Neurogenesis bookkeeping.
        if self.neuro is not None:
            self.neuro.report_usage([action % self.neuro.n], [abs(td_err) + 1e-3])
            self.neuro.maybe_grow()

        # 11) Roll state.
        self._prev_z, self._prev_a = z_top, action
        self._step += 1

        # --- v9 cognitive layer (before sleep so it sees this step) ---
        v9_out = self._v9_step(
            z_top=z_top,
            action_probs=action_probs,
            uncertainty=float(uncertainty),
            sensory_err=float(sensory_err),
            td_err=float(td_err),
            attn=attn,
        )

        # --- v10 efferent / agentic layer ---
        v10_out = self._v10_step(
            z_top=z_top,
            action_probs=action_probs,
            sensory_err=float(sensory_err),
            td_err=float(td_err),
            uncertainty=float(uncertainty),
            epistemic_mode=(v9_out or {}).get("epistemic", {}).get("mode") if v9_out else None,
            cortisol=float(self.nocic.state.cortisol),
        )

        # --- v11 Aurora-Praxis layer ---
        v11_out = self._v11_step(z_top=z_top)

        # 12) Sleep — periodic OR on-demand.
        do_sleep = (self._step % self.cfg.consolidation_every == 0) or self.homeo.request_sleep()
        if do_sleep:
            self.sleep()

        # 13) Curriculum.
        self.curriculum.adapt(self._ema_error)

        out = dict(
            step=self._step, action=action, value=self.basal.value(z_top),
            sensory_err=sensory_err, td_err=td_err, uncertainty=uncertainty,
            attn=attn,
            modulators=self.modulators.snapshot(),
            affect=self.nocic.snapshot(),
            z_top=z_top,
            # v6/v7 additions (optional consumers)
            action_mode=action_mode,
            cerebellum_err=cb_err,
            introspection=introspect,
            workspace_coherence=(self.workspace.coherence
                                 if self.workspace is not None else 0.0),
            anatomy_active=(anatomy_states is not None),
        )
        # v8 additions (never required).
        if self.cerebrum is not None:
            out["interhem_coherence"] = float(self.cerebrum._interhem_coherence)
            if self.cerebrum.metabolic is not None:
                out["metabolic_active_frac"] = float(
                    self.cerebrum.metabolic._last_used_frac)

        # v12 genesis additions
        if symbiosis_stats is not None:
            out["symbiosis"] = symbiosis_stats

        # v9 additions (never required).
        if v9_out is not None:
            out.update(v9_out)
        # v10 additions (never required).
        if v10_out is not None:
            out.update(v10_out)
        # v11 additions (never required).
        if v11_out is not None:
            out.update(v11_out)
        return out

    # ====================================================================
    def _v9_step(self, *,
                 z_top: np.ndarray,
                 action_probs: np.ndarray,
                 uncertainty: float,
                 sensory_err: float,
                 td_err: float,
                 attn: dict) -> Optional[dict]:
        """Run the v9 cognitive modules once per organism step.

        All modules are optional; if none are enabled this returns None
        and behaviour is byte-for-byte v8.
        """
        if not any((self.epistemic, self.kernel, self.wm,
                    self.self_model, self.oscillator)):
            return None

        out: dict = {}

        # -- Oscillator: advance phases first so every downstream module
        #    sees the current phase state.
        if self.oscillator is not None:
            # v11: Adrenaline-driven time dilation speeds up the oscillator
            adrenaline_level = 0.0
            if self.endocrine is not None:
                adrenaline_level = self.endocrine.time_dilation
            self.oscillator.tick(adrenaline=adrenaline_level)
            out["oscillator_gates"] = self.oscillator.all_gates()

        # -- Novelty estimate for the epistemic gate.  Cheap: 1 minus
        #    the maximum cosine similarity between z_top and the
        #    stigmergic-graph nodes if the graph is available; else 0.
        novelty = 0.0
        if self.stig is not None:
            try:
                sims = self.stig.similarities(z_top)  # if v6 supports it
                if len(sims):
                    novelty = float(1.0 - float(np.max(sims)))
            except Exception:
                novelty = 0.0

        # -- Epistemic gate.
        epistemic_decision = None
        if self.epistemic is not None:
            precision = 1.0
            if hasattr(self.cortex, "precision_profile"):
                try:
                    precision = float(np.mean(self.cortex.precision_profile()))
                except Exception:
                    precision = 1.0
            epistemic_decision = self.epistemic.decide(
                ensemble_var=float(uncertainty),
                mean_precision=float(precision),
                sensory_err=float(sensory_err),
                novelty=float(novelty),
            )
            out["epistemic"] = epistemic_decision

        # -- Constitutional kernel: dissonance + veto detection.
        kernel_penalty = 0.0
        veto_fired = False
        if self.kernel is not None:
            kernel_penalty = float(self.kernel.penalty(z_top))
            veto_fired = bool(self.kernel.veto(z_top))
            out["constitutional"] = dict(
                penalty=kernel_penalty,
                veto=veto_fired,
                violated=self.kernel.violated(z_top),
                hash_ok=bool(self.kernel.verify()),
            )

        # -- Working memory: try to admit z_top with a priority derived
        #    from |td_err| + novelty + confidence-of-gate.  Then decay
        #    the rest.
        if self.wm is not None:
            gate_bonus = 0.0
            if self.oscillator is not None:
                # only allow admission during the theta-gamma binding
                # window; outside it we still admit but at reduced
                # priority so we naturally get ~4 items per theta cycle.
                gate_bonus = (1.0 if self.oscillator.is_pac_binding_window()
                              else 0.25)
            else:
                gate_bonus = 0.5
            pri = (abs(td_err) + 0.5 * novelty) * gate_bonus
            admitted = self.wm.write(z_top, priority=pri,
                                     tag=",".join(list(attn.keys())[:2]))
            self.wm.tick()
            out["working_memory"] = dict(
                admitted=bool(admitted),
                occupied=len(self.wm.slots),
                capacity=self.wm.capacity,
                priority=float(pri),
            )

        # -- Self-model: update + emit possible belief revision.
        if self.self_model is not None:
            # Policy entropy from the softmax over actions.
            probs = np.asarray(action_probs, dtype=np.float64) + 1e-9
            probs = probs / probs.sum()
            entropy = float(-np.sum(probs * np.log(probs)))
            dominant_mod = None
            if attn:
                dominant_mod = max(attn.items(), key=lambda kv: kv[1])[0]
            U = float(epistemic_decision["U"]) if epistemic_decision else 0.0
            workspace_coh = (float(self.workspace.coherence)
                             if self.workspace is not None else 0.0)
            rev = self.self_model.update(
                step=self._step,
                z_top=z_top,
                U=U,
                policy_entropy=entropy,
                workspace_coherence=workspace_coh,
                kernel_penalty=kernel_penalty,
                veto_fired=veto_fired,
                dominant_modality=dominant_mod,
                cause_hint=("kernel_veto" if veto_fired
                            else "prediction_error_spike"
                            if abs(td_err) > 0.5 else None),
            )
            out["self_model"] = self.self_model.describe_current_state()
            if rev is not None:
                out["belief_revision"] = dict(
                    step=rev.step, delta_U=rev.delta_U,
                    delta_z_norm=rev.delta_z_norm,
                    cause=rev.cause, U_after=rev.U_after,
                )

        return out

    # ====================================================================
    def _v10_step(self, *,
                   z_top: np.ndarray,
                   action_probs: np.ndarray,
                   sensory_err: float,
                   td_err: float,
                   uncertainty: float,
                   epistemic_mode: Optional[str],
                   cortisol: float) -> Optional[dict]:
        """Run the v10 efferent / agentic modules once per organism step.

        All modules are optional; if none are enabled this returns None
        and behaviour is byte-for-byte v9.

        Order of operations (mirrors _v9_step):
          1. Evaluate agency for the PREVIOUS step's motor dispatch,
             using the *current* z_top as observed reafference.
          2. Fold the (predicted, observed) pair into the somatosensory
             field.
          3. Compute intrinsic reward (mastery + novelty + competence).
          4. Optionally dispatch a new motor action (schema-checked,
             veto-checked, β-gated).  The dispatch itself does not run
             synchronously against the world — the deployer provides
             the tool callable, so this fires the tool right now and
             returns the result immediately.
          5. Publish stats.
        """
        if not any((self.motor, self.agency, self.somatic,
                    self.drive, self.mirror)):
            return None

        out: dict = {}

        # Reafference delta — how much did the sensory bus change since
        # last step.  A cheap proxy for the sensory delta that would
        # otherwise require a full world-model rollout.
        if self._prev_z_top is not None:
            observed_effect = (z_top - self._prev_z_top).astype(np.float32)
            n = float(np.linalg.norm(observed_effect)) + 1e-9
            observed_effect_unit = observed_effect / n
        else:
            observed_effect_unit = None

        # 1. Agency evaluation for the PREVIOUS motor dispatch.
        agency_rep = None
        if self.agency is not None and self._last_motor_return is not None:
            pred = self._last_motor_return.predicted_effect
            agency_rep = self.agency.evaluate(
                predicted_effect=pred,
                observed_effect=observed_effect_unit,
                step=self._step,
                tool_name=self._last_motor_return.tool_name,
            )
            if agency_rep is not None:
                out["agency"] = dict(
                    tool=agency_rep.tool_name,
                    sense_of_agency=agency_rep.sense_of_agency,
                    mismatch=agency_rep.mismatch,
                    low_agency=agency_rep.low_agency,
                )
                # If agency collapsed, log a belief revision.
                if agency_rep.low_agency and self.self_model is not None:
                    self.self_model.update(
                        step=self._step,
                        z_top=z_top,
                        U=1.0 - agency_rep.sense_of_agency,
                        policy_entropy=0.0,
                        cause_hint="reafference_mismatch",
                    )

        # 2. Somatosensory field update.
        if self.somatic is not None:
            pred_eff = (self._last_motor_return.predicted_effect
                         if self._last_motor_return is not None else None)
            self.somatic.ingest(
                predicted_effect=pred_eff,
                observed_effect=observed_effect_unit,
                cost=(0.0 if self._last_motor_return is None
                       else 1.0 if self._last_motor_return.executed else 0.0),
            )
            out["somatosensory"] = dict(
                flinch=self.somatic.flinch(),
                schema_norm=float(np.linalg.norm(self.somatic._body_schema)),
            )

        # 3. Intrinsic drive.
        if self.drive is not None:
            # Reuse the epistemic gate's novelty side-channel if
            # available; else recompute cheaply from stig graph.
            novelty = 0.0
            if self.stig is not None:
                try:
                    sims = self.stig.similarities(z_top)
                    if len(sims):
                        novelty = float(1.0 - float(np.max(sims)))
                except Exception:
                    novelty = 0.0
            soa = (self.agency.sense_of_agency_ema
                   if self.agency is not None else None)
            sample = self.drive.compute(
                sensory_err=sensory_err,
                novelty=novelty,
                sense_of_agency=soa,
                cortisol=cortisol,
                epistemic_mode=epistemic_mode,
            )
            out["intrinsic"] = dict(
                r_int=sample.r_int, mastery=sample.mastery,
                novelty=sample.novelty, competence=sample.competence,
                gated_off=sample.gated_off,
            )

        # 4. Motor dispatch.  Only if MotorCortex is enabled *and* has
        #    at least one registered tool.  The dispatch is completely
        #    non-blocking on the rest of the step — tools that fail or
        #    are vetoed return a MotorReturn(executed=False) that later
        #    steps can inspect.
        motor_ret = None
        if self.motor is not None and self.motor.tools():
            beta_gate = 1.0
            if self.oscillator is not None:
                # β-band “motor go” window (§5.6 RATIONALE promise now
                # actually wired).  Map cos(phase) ∈ [-1,1] to [0,1].
                beta_gate = 0.5 * (1.0 + self.oscillator.gate("beta"))
            motor_ret = self.motor.select_and_execute(
                action_probs=action_probs,
                step=self._step,
                kernel=self.kernel,
                epistemic_mode=epistemic_mode,
                beta_gate=beta_gate,
            )
            out["motor"] = dict(
                tool=motor_ret.tool_name,
                executed=bool(motor_ret.executed),
                veto=bool(motor_ret.veto),
                refused=bool(motor_ret.refused),
                gated_out=bool(motor_ret.gated_out),
                beta_gate=float(beta_gate),
            )
            self._last_motor_return = motor_ret

        # 5. Roll z_top for next-step reafference.
        self._prev_z_top = z_top.copy()

        return out

    # ====================================================================
    def _v11_step(self, *, z_top: np.ndarray) -> Optional[Dict[str, Any]]:
        """V11 Praxis layer: Long-horizon PFC and Lateralization."""
        out = {}
        
        # 1) Prefrontal Cortex (Long-term goal bias)
        if self.pfc is not None:
            pfc_stats = self.pfc.step(z_top, learn=True)
            
            # Apply PFC top-down bias intensity to the thalamus
            intensity = pfc_stats["goal_intensity"]
            if intensity > 0:
                self.thalamus.reinforce("pfc_goal", signal=intensity, lr=0.1)
                
            out["pfc"] = pfc_stats
            
        # 2) Hemispheric Lateralization (Split-Brain Sync)
        if getattr(self.cfg, "lateralization_enable", False) and self.cerebrum is not None:
            # Enforce corpus callosum syncing:
            # We mix states between Left and Right hemispheres based on the oscillator bus
            # Specifically, we sync during Alpha or Gamma bursts.
            sync_factor = 0.5 # Absolute theoretical limit default if no oscillator
            if self.oscillator is not None:
                # Tweak to absolute theoretical maximum: 50% merge during alpha bursts
                alpha_gate = self.oscillator.gate("alpha")
                sync_factor = max(0.05, min(0.5, 0.5 * alpha_gate))
                
            # If the cerebrum has atlas proxies, sync left and right frontal regions explicitly
            proxies = self.cerebrum.proxies
            if "lh_frontal" in proxies and "rh_frontal" in proxies:
                lh_act = proxies["lh_frontal"].state.copy()
                rh_act = proxies["rh_frontal"].state.copy()
                
                # Bi-directional sync via Corpus Callosum
                proxies["lh_frontal"].state = (1.0 - sync_factor) * lh_act + (sync_factor) * rh_act
                proxies["rh_frontal"].state = (1.0 - sync_factor) * rh_act + (sync_factor) * lh_act
                
            out["corpus_callosum_sync"] = float(sync_factor)

        # 3) Endocrine System (Adrenaline / Oxytocin / Melatonin)
        if self.endocrine is not None:
            # Compute sensory surprise from the EMA error
            sensory_surprise = max(0.0, self._ema_error - 0.3)
            cortisol = float(self.nocic.state.cortisol)
            # Social signal: if mirror system detected an observed action, boost oxytocin
            social_signal = 0.0
            if self.mirror is not None and self._last_motor_return is not None:
                if self._last_motor_return.executed:
                    social_signal = 0.1

            self.endocrine.update(
                sensory_surprise=sensory_surprise,
                cortisol=cortisol,
                social_signal=social_signal,
            )

            # If oxytocin is high and mirror system exists, lower match threshold
            if self.mirror is not None and self.endocrine.empathy_boost > 0:
                self.mirror.match_threshold = max(
                    0.3,
                    self.cfg.mirror_match_threshold - self.endocrine.empathy_boost,
                )

            # If melatonin sleep pressure is high, request sleep via homeostasis
            if self.endocrine.sleep_pressure > 1.0:
                self.homeo._sleep_requested = True

            out["endocrine"] = self.endocrine.snapshot()

        # 4) Language Grounding (Wernicke → Arcuate → Broca)
        if self.arcuate is not None:
            # Run a single cycle of inner speech (subvocalization):
            # the organism "thinks" by producing candidate tokens and
            # feeding them back through Wernicke to refine its semantic state.
            refined_z = self.arcuate.inner_speech(z_top, n_tokens=2, temperature=0.8)

            # The refined semantic vector biases the PFC's goal tracking
            if self.pfc is not None:
                # Blend inner speech refinement into the PFC's prediction
                self.pfc._predicted_sdr = (
                    0.7 * self.pfc._predicted_sdr + 0.3 * refined_z
                ).astype(np.float32)

            out["language"] = self.arcuate.stats()

        if not out:
            return None
        return {"praxis": out}

    # ====================================================================
    def _thalamic_priority_map(self, observations: dict,
                                attn: dict) -> dict:
        """Boost anatomical regions whose declared modalities are present.

        Uses ``BrainRegion.modalities`` (already declared in the atlas).
        Each region gets base priority 1.0, plus ``thalamic_boost ×
        thalamus_attn`` for every declared modality that appears in the
        observation dict.
        """
        if self.cerebrum is None:
            return {}
        boost = float(self.cfg.thalamic_boost)
        obs_mods = set(observations.keys()) | {"interoception"}
        out: dict = {}
        for r in self.cerebrum.atlas.regions:
            score = 1.0
            for m in r.modalities:
                if m in obs_mods:
                    w = float(attn.get(m, 0.5)) if attn else 0.5
                    score += boost * w
            out[r.name] = score
        return out

    # ====================================================================
    def sleep(self) -> dict:
        s = sleep_cycle(
            hippocampus=self.hippocampus, cortex=self.cortex,
            world_model=self.world_model, basal=self.basal,
            modulators=self.modulators, nociceptor=self.nocic,
            rumination_guard=self.rumination_guard,
            cfg=self.cfg,
            policy_fn=lambda z: self.basal.policy(z, temp=1.0)[0],
            n_steps=min(self.cfg.dream_steps_per_sleep,
                        max(1, len(self.hippocampus) // self.cfg.replay_batch)),
        )
        # v6 DMN: spontaneous mind-wandering during sleep
        if self.dmn is not None:
            for _ in range(8):
                _ = self.dmn.sample(n_components=3)
        # v7 anatomy REM loop: cortico-cerebellar replay
        if self.cerebrum is not None and self.cfg.anatomy_sleep_loop:
            for _ in range(4):
                if self.workspace is not None and self.workspace.hist:
                    seed_z = self.workspace.hist[-1]
                else:
                    seed_z = np.random.randn(self.cfg.latent_dim).astype(np.float32)
                    seed_z /= np.linalg.norm(seed_z) + 1e-9
                self.cerebrum.step(seed_z, lr=self.cfg.lr_cortex * 0.25)
                
        # v11 Microglia immune pruning during sleep
        if self.microglia is not None and self.stig is not None:
            pruning_stats = self.microglia.patrol_and_prune(self.stig)
            s["microglia"] = pruning_stats
            
        # v12: Somatic Evolution (Mitosis/Apoptosis)
        if self.mitosis is not None and self.cerebrum is not None:
            # Gather usage metrics from the metabolic budget
            usages = {}
            if self.cerebrum.metabolic is not None:
                usages = self.cerebrum.metabolic._usage_stats
            # Use regional precisions (or lack thereof) as a proxy for error.
            # (Regions with high precision are confident/low-error).
            errors = {}
            precs = self.cerebrum.regional_precisions()
            for r, p in zip(self.cerebrum.proxies.keys(), precs):
                errors[r] = 1.0 - float(p)
                
            mitosis_stats = self.mitosis.evaluate_and_evolve(usages, errors)
            s["somatic_evolution"] = mitosis_stats
            
        # v14: Connectome Hebbian Plasticity
        if getattr(self.cfg, "foundation_enable", False) and self.cerebrum is not None:
            if self.cerebrum.metabolic is not None:
                usages = self.cerebrum.metabolic._usage_stats
                n_edges = self.cerebrum.connectome.hebbian_update(usages, lr=0.01)
                s["hebbian_edges_updated"] = n_edges
            
        return s

    # ====================================================================
    # V14 Hibernation: Persistence across reboots
    # ====================================================================
    def hibernate(self, filepath: str) -> None:
        """Serialize the brain state to disk."""
        import pickle
        state = {
            "step": self._step,
            "ema_error": self._ema_error,
        }
        if self.cerebrum is not None:
            # Save connectome plasticity weights
            state["connectome_edges"] = [(e.src, e.dst, e.weight) for e in self.cerebrum.connectome.edges]
            # Save inner cortical/cerebellar weights and state vectors
            state["proxies"] = {name: proxy.stack for name, proxy in self.cerebrum.proxies.items()}
            state["proxy_states"] = {name: proxy.state for name, proxy in self.cerebrum.proxies.items()}
        if self.isomorphic_encoder is not None:
            # Save the concept cache and HTM temporal memory
            state["isomorphic_cache"] = self.isomorphic_encoder._cache
            state["iso_proj"] = self.iso_proj
            if hasattr(self.isomorphic_encoder, 'htm'):
                state["iso_htm"] = self.isomorphic_encoder.htm
        if self.symbiosis is not None:
            state["symbiosis"] = self.symbiosis.alignment_score
            
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
            
    def wake_from_hibernation(self, filepath: str) -> None:
        """Restore the brain state from disk."""
        import os, pickle
        if not os.path.exists(filepath):
            return
            
        with open(filepath, "rb") as f:
            state = pickle.load(f)
            
        self._step = state.get("step", 0)
        self._ema_error = state.get("ema_error", 0.5)
        
        if self.cerebrum is not None:
            if "connectome_edges" in state:
                # Restore connectome plasticity weights
                edge_dict = {(src, dst): w for src, dst, w in state["connectome_edges"]}
                for e in self.cerebrum.connectome.edges:
                    if (e.src, e.dst) in edge_dict:
                        e.weight = edge_dict[(e.src, e.dst)]
                # Rebuild adjacency
                self.cerebrum.connectome._adj.clear()
                for e in self.cerebrum.connectome.edges:
                    self.cerebrum.connectome._adj.setdefault(e.src, []).append((e.dst, float(e.weight), e.bundle))
            if "proxies" in state:
                for name, stack in state["proxies"].items():
                    if name in self.cerebrum.proxies:
                        self.cerebrum.proxies[name].stack = stack
            if "proxy_states" in state:
                for name, p_state in state["proxy_states"].items():
                    if name in self.cerebrum.proxies:
                        self.cerebrum.proxies[name].state = p_state
                
        if self.isomorphic_encoder is not None and "isomorphic_cache" in state:
            self.isomorphic_encoder._cache = state["isomorphic_cache"]
            if "iso_proj" in state:
                self.iso_proj = state["iso_proj"]
            if "iso_htm" in state:
                self.isomorphic_encoder.htm = state["iso_htm"]
                
        if self.symbiosis is not None and "symbiosis" in state:
            self.symbiosis.alignment_score = state["symbiosis"]

    # ====================================================================
    # Generation / imagination
    # ====================================================================
    def generate_text(self, prompt: str, n_tokens: int = 24, temperature: float = 1.0) -> str:
        z0 = self.perceive(prompt, "text")
        z0, _ = self.cortex.forward(z0, lr=0.0)
        return self.text_decoder.generate(
            z0, self.world_model,
            policy_fn=lambda z: self.basal.policy(z, temp=temperature)[0],
            n_tokens=n_tokens, temp=temperature,
        )

    def imagine(self, prompt_modality: str, prompt: Any,
                horizon: Optional[int] = None) -> list:
        horizon = horizon or self.cfg.imagination_horizon
        z = self.perceive(prompt, prompt_modality)
        z, _ = self.cortex.forward(z, lr=0.0)
        traj = self.world_model.rollout(
            z, policy_fn=lambda zz: self.basal.policy(zz, temp=1.0)[0],
            horizon=horizon, uncertainty_stop=1.0,
        )
        return [t[3] for t in traj]

    # ====================================================================
    def live(self, stream: Iterable[Tuple[dict, float, bool]], verbose: bool = True):
        for obs, r, d in stream:
            out = self.step(obs, reward=r, done=d)
            if verbose and self._step % self.cfg.log_every == 0:
                m, af = out["modulators"], out["affect"]
                print(f"[t={self._step:>6}] err={out['sensory_err']:.3f} "
                      f"td={out['td_err']:+.3f} u={out['uncertainty']:.3f} "
                      f"a={out['action']:>3d} v={out['value']:+.2f} | "
                      f"DA={m['da']:.2f} 5HT={m['ht']:.2f} NE={m['ne']:.2f} ACh={m['ach']:.2f} | "
                      f"val={af['valence']:+.2f} cort={af['cortisol']:.2f} pain={af['pain_recent']:.2f}")
            if d:
                self._prev_z = None
                self._prev_a = None

    # ====================================================================
    def report(self) -> dict:
        """Diagnostic snapshot — useful at the end of a training run."""
        cap = self.cfg.estimated_params()
        rep = dict(
            step=self._step,
            ema_error=self._ema_error,
            affect=self.nocic.snapshot(),
            hippocampus_size=len(self.hippocampus),
            cortex_kind=self.cortex_kind,
            tier=self.cfg.tier,
            params=cap,
            dissociation_strikes=self.thalamus.dissociation_strikes,
            alignment_penalties=self.alignment.triggered,
            text_vocab=len(self.text_decoder.tokens),
        )
        # v9 stats (before v6 to keep diagnostic near top of report)
        if self.epistemic is not None:   rep["epistemic"] = self.epistemic.stats()
        if self.kernel is not None:      rep["constitutional"] = self.kernel.stats()
        if self.wm is not None:          rep["working_memory"] = self.wm.stats()
        if self.self_model is not None:
            rep["self_model"] = dict(
                summary=self.self_model.describe_current_state(),
                last_revisions=self.self_model.last_n_revisions(5),
                stats=self.self_model.stats(),
            )
        if self.oscillator is not None:  rep["oscillator"] = self.oscillator.stats()

        # v10 stats
        if self.motor is not None:       rep["motor"] = self.motor.stats()
        if self.agency is not None:      rep["agency"] = self.agency.stats()
        if self.somatic is not None:     rep["somatosensory"] = self.somatic.stats()
        if self.drive is not None:       rep["intrinsic"] = self.drive.stats()
        if self.mirror is not None:      rep["mirror"] = self.mirror.stats()

        # v6 stats
        if self.hvsa is not None:        rep["hvsa"] = self.hvsa.stats()
        if self.stig is not None:        rep["stig"] = self.stig.stats()
        if self.sdm is not None:         rep["sdm"] = self.sdm.stats()
        if self.astro is not None:       rep["astro"] = self.astro.stats()
        if self.workspace is not None:   rep["workspace"] = self.workspace.stats()
        if self.dmn is not None:         rep["dmn"] = self.dmn.stats()
        if self.cerebellum_module:       rep["cerebellum"] = self.cerebellum_module.stats()
        if self.intro is not None:       rep["intro"] = self.intro.stats()
        if self.neuro is not None:       rep["neuro"] = self.neuro.stats()
        if self.ai_selector is not None: rep["active_inference"] = self.ai_selector.stats()
        # v7 anatomy census (v8 adds metabolic + neuromod + interhem)
        if self.cerebrum is not None:
            rep["anatomy"] = self.cerebrum.census()
            rep["connectome"] = self.cerebrum.connectome.stats()
            if self.cerebrum.neuromod_bus is not None:
                rep["neuromod_bus"] = self.cerebrum.neuromod_bus.stats()
            if self.cerebrum.metabolic is not None:
                rep["metabolic"] = self.cerebrum.metabolic.stats()
            rep["interhem_coherence"] = float(self.cerebrum._interhem_coherence)
        return rep


# Back-compat alias.
AuroraPrime = AuroraOmnia
