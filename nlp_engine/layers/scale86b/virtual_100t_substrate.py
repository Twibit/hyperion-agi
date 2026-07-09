from __future__ import annotations

from collections import Counter, OrderedDict
import re

import numpy as np

from nlp_engine.layers.complex_neuron import BiologicalNeuron, PyramidalNeuron, PurkinjeNeuron, VonEconomoNeuron
from nlp_engine.layers.scale86b.sdr import (
    SparseDistributedRepresentation,
    encode_tokens_to_sdr,
)
from nlp_engine.layers.scale86b.sparse_synapse_store import SparseSynapseStore
import rust_backend_shim


_TOKEN_RE = re.compile(r"[A-Za-z0-9_'\-]+")


class Virtual100TSubstrate:
    """Sparse virtual neuron substrate with bounded biological materialization.

    The important constraint is that scale is virtual. The substrate can
    expose an 86B, 100B, or 100T address space, but only the hot working set
    gets materialized as `BiologicalNeuron` objects. The SDR and synapse store
    carry the large-scale learning signal; the neuron objects add local
    biophysics for the active cascade.
    """

    def __init__(
        self,
        neurons_cerebellum: int = 80_000_000_000_000,
        neurons_cortex: int = 19_000_000_000_000,
        neurons_brainstem: int = 1_000_000_000_000,
        k_active: int = 4096,
        working_set: int = 64,
        ram_cap_synapses: int = 10_000_000,
        expected_total: int | None = 100_000_000_000_000,
        biology_sample_size: int = 512,
        rng_seed: int = 0xC0FFEE,
        atlas=None,
    ):
        self.neurons_cerebellum = int(neurons_cerebellum)
        self.neurons_cortex = int(neurons_cortex)
        self.neurons_brainstem = int(neurons_brainstem)
        self.total_neurons = (
            self.neurons_cerebellum
            + self.neurons_cortex
            + self.neurons_brainstem
        )

        if expected_total is not None and self.total_neurons != int(expected_total):
            raise ValueError(
                f"virtual neuron total must be {expected_total:,}; got {self.total_neurons:,}"
            )
        if self.total_neurons >= (1 << 63):
            raise ValueError("virtual neuron IDs must fit signed int64 for SparseSynapseStore")

        self.k_active = int(k_active)
        self.working_set_capacity = int(working_set)
        self.biology_sample_size = max(1, int(biology_sample_size))
        # V29: True Hardware Homeostasis (Adenosine Sleep Drive)
        # Instead of a hardcoded neuron cap, the brain secretes Adenosine based on actual RAM pressure.
        # This prevents toasting the phone while utilizing maximum available skull volume.
        self.adenosine = 0.0
        self.target_ram_mb = self._get_system_ram_mb() * 0.75  # Allow up to 75% of physical host RAM

        self.synapse_store = SparseSynapseStore(ram_cap_synapses=ram_cap_synapses)
        self.expected_reward = 0.0 # V22: Basal Ganglia Critic (RPE)
        # Compatibility with GlialNetwork, which expects `.synapses`.
        self.synapses = self.synapse_store

        # int neuron ID -> BiologicalNeuron. OrderedDict is an LRU cache.
        self.active_biology: OrderedDict[int, BiologicalNeuron] = OrderedDict()
        self._rng = np.random.default_rng(rng_seed)
        self._token_freq: Counter[str] = Counter()
        self._last_encode_stats: dict = {}
        self._last_recall_stats: dict = {}

        if atlas is not None:
            self._regions = {}
            current_offset = 0
            for r in atlas.regions:
                self._regions[r.name] = (current_offset, r.neurons)
                current_offset += r.neurons
            self.total_neurons = current_offset
        else:
            self._regions = {
                "cerebellum": (0, self.neurons_cerebellum),
                "cortex": (self.neurons_cerebellum, self.neurons_cortex),
                "brainstem": (
                    self.neurons_cerebellum + self.neurons_cortex,
                    self.neurons_brainstem,
                ),
            }

        scale_label = self._scale_label()
        print(f"[Virtual100TSubstrate] Initialized {scale_label} virtual neurons.")
        print(
            "[Virtual100TSubstrate] "
            f"k_active={self.k_active}, Adenosine RAM limit={self.target_ram_mb}MB."
        )
        
        # Instantiate NativeAgiKernel (CPU/GPU-agnostic compute layer)
        self.agi_kernel = rust_backend_shim.NativeAgiKernel(
            depth=3,
            n_per_layer=32,
            n_channels=8,
            top_k=4,
            dt_ms=25.0
        )
        self.dopamine = 0.2
        self.noradrenaline = 0.2
        self.serotonin = 0.2
        self.acetylcholine = 0.2
        self.arousal = 0.5
        self.throttle_cortex = False

    @classmethod
    def human_86b(cls, **kwargs) -> "Virtual100TSubstrate":
        """86B-neuron human-scale address space."""
        return cls(
            neurons_cerebellum=69_000_000_000,
            neurons_cortex=16_000_000_000,
            neurons_brainstem=1_000_000_000,
            expected_total=86_000_000_000,
            **kwargs,
        )

    @classmethod
    def century_100b(cls, **kwargs) -> "Virtual100TSubstrate":
        """100B-neuron address space."""
        return cls(
            neurons_cerebellum=80_000_000_000,
            neurons_cortex=19_000_000_000,
            neurons_brainstem=1_000_000_000,
            expected_total=100_000_000_000,
            **kwargs,
        )

    @classmethod
    def from_scale(cls, scale: str = "100t", **kwargs) -> "Virtual100TSubstrate":
        s = scale.lower().replace("_", "").replace("-", "")
        if s in {"86b", "human", "human86b"}:
            return cls.human_86b(**kwargs)
        if s in {"100b", "century", "century100b"}:
            return cls.century_100b(**kwargs)
        if s in {"100t", "trillion", "hundredtrillion"}:
            return cls(**kwargs)
        raise ValueError(f"unknown substrate scale: {scale!r}")

    def _scale_label(self) -> str:
        if self.total_neurons >= 1_000_000_000_000:
            return f"{self.total_neurons / 1_000_000_000_000:.0f}T"
        return f"{self.total_neurons / 1_000_000_000:.0f}B"

    def _tokenize(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(str(text).lower())
        return tokens or [str(text).lower().strip() or "<blank>"]

    def _region_for_domain(self, domain: str) -> str:
        if domain in self._regions:
            return domain
        domain = (domain or "prose").lower()
        if domain in {"motor", "habit", "skill", "procedural", "cerebellar"}:
            return "cerebellum_left" if "cerebellum_left" in self._regions else "cerebellum"
        if domain in {"instinct", "reflex", "autonomic", "brainstem", "vital"}:
            return "brainstem" if "brainstem" in self._regions else "brainstem"
        return "lh_frontal" if "lh_frontal" in self._regions else "cortex"

    def _biology_subset(self, sdr: SparseDistributedRepresentation) -> np.ndarray:
        if sdr.active_ids.size <= self.biology_sample_size:
            return sdr.active_ids.astype(np.int64, copy=False)
        idx = np.argpartition(-sdr.weights, self.biology_sample_size - 1)[: self.biology_sample_size]
        return sdr.active_ids[idx].astype(np.int64, copy=False)

    def _instantiate_biology(self, active_ids: np.ndarray) -> None:
        """Wake dormant integer IDs into hot `BiologicalNeuron` objects based on region."""
        for raw_nid in active_ids:
            nid = int(raw_nid)
            neuron = self.active_biology.get(nid)
            if neuron is None:
                # Region logic
                # V29: True Anatomical Cell Typing
                # Find which region this neuron ID belongs to
                region_name = "cortex"
                for r_name, (r_start, r_end) in self._regions.items():
                    if r_start <= nid < r_end:
                        region_name = r_name
                        break
                        
                hemisphere = "right" if "rh_" in region_name or "right" in region_name else "left"
                        
                if "cerebellum" in region_name:
                    neuron = PurkinjeNeuron(str(nid))
                elif "frontal" in region_name or "cingulate" in region_name or "insula" in region_name:
                    # Frontal and Executive areas are rich in Von Economo neurons (fast logic/intuition)
                    neuron = VonEconomoNeuron(str(nid), hemisphere=hemisphere)
                else:
                    # Standard cortex uses Pyramidal Neurons
                    neuron = PyramidalNeuron(str(nid), hemisphere=hemisphere)
                self.active_biology[nid] = neuron
            else:
                self.active_biology.move_to_end(nid)

    def _get_neuron_id(self, concept: str, domain: str, index: int) -> int:
        """Stable virtual coordinate retained for older call sites."""
        raw = rust_backend_shim.hash_neuron_id(
            concept.encode("utf-8"), domain.encode("utf-8"), index
        )
        return int(raw % self.total_neurons)

    def _stable_token_hash(self, text: str, domain: str) -> int:
        return self._get_neuron_id(text, domain, 0)


    def _get_system_ram_mb(self) -> float:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024.0
        except:
            pass
        return 1024.0  # Safe fallback for unknown systems

    def _get_ram_usage_mb(self) -> float:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except:
            return float(len(self.active_biology)) / 1000.0 # Fallback heuristic
        return 0.0

    def _propagate_calcium_waves(self):
        """V22: Astrocyte Syncytium - global slow calcium diffusion for credit assignment"""
        if not self.active_biology:
            return
            
        # Update Adenosine (Sleep Pressure) based on physical RAM
        current_ram = self._get_ram_usage_mb()
        ram_pressure = current_ram / self.target_ram_mb
        self.adenosine = 0.9 * self.adenosine + 0.1 * ram_pressure
        
        pool = sum(n.biochem.calcium for n in self.active_biology.values())
        avg_ca = pool / len(self.active_biology)
        
        to_evict = []
        for nid, n in self.active_biology.items():
            n.biochem.calcium = 0.9 * n.biochem.calcium + 0.1 * avg_ca
            
            # V29: Microglia Immune System (Synaptic Pruning)
            # If the cell is resting, microglia sweep through to delete dead synapses.
            if n.biochem.atp > 0.95 and hasattr(n, 'prune'):
                n.prune()
            
            # V29: Adenosine Sleep Eviction (Hardware Homeostasis)
            # If the brain is exhausted (RAM full), Adenosine forces cells to sleep.
            # Normal eviction: rests if ATP is full. Panic eviction: rests everything but active paths.
            sleep_threshold = 0.98 if self.adenosine < 0.8 else 0.5
            if n.biochem.calcium < (0.005 / max(0.1, self.adenosine)) and n.biochem.atp > sleep_threshold:
                to_evict.append(nid)
                
        for nid in to_evict:
            del self.active_biology[nid]
            
        # V29: Blackout
        if self.adenosine > 1.5:
            # Extreme RAM pressure. Brain faints to save the physical device.
            self.active_biology.clear()

    def encode(self, text: str, domain: str = "prose") -> SparseDistributedRepresentation:
        """Convert text into a sparse population code and fire hot neurons."""
        tokens = self._tokenize(text)
        region = self._region_for_domain(domain)
        base, region_size = self._regions[region]

        # V27: Metabolic Autoregulation (Dynamic Sparsity)
        if self.active_biology:
            global_atp = sum(n.biochem.atp for n in self.active_biology.values()) / len(self.active_biology)
        else:
            global_atp = 1.0
            
        dynamic_k = max(64, int(self.k_active * global_atp))

        # Correct argument order: region base, region size, then k_active.
        sdr = encode_tokens_to_sdr(
            tokens,
            region_base=base,
            region_size=region_size,
            k_active=dynamic_k,
            rng=self._rng,
        )
        sdr.region = region
        sdr.meta.update(
            {
                "text": text,
                "domain": domain,
                "tokens": tokens,
                "k_active": self.k_active,
                "scale": self._scale_label(),
            }
        )

        biology_ids = self._biology_subset(sdr)
        self._instantiate_biology(biology_ids)

        token_hash = self._stable_token_hash(" ".join(tokens), domain)
        fatigue_warnings = 0
        output_sum = 0.0
        atp_sum = 0.0
        calcium_sum = 0.0
        for raw_nid in biology_ids:
            nid = int(raw_nid)
            neuron = self.active_biology[nid]
            
            # V26: Continuous Probabilistic White Matter Bypass
            # Avoids hardcoded thresholds. The chance of saltatory conduction 
            # scales continuously with the physical myelin thickness (1.0 to 2.0).
            myelin = getattr(neuron, 'axon_myelin', 1.0)
            bypass_probability = max(0.0, myelin - 1.0)
            
            import numpy as np
            if np.random.rand() < bypass_probability:
                state = {"output": myelin, "spiked": True}
                # Habituated pathways skip slow dendritic math
            else:
                state = neuron.forward(token_hash ^ nid)
                neuron.biochem.calcium += 0.5 # V25: SDR-driven Ca2+ influx
            fatigue_warnings += int(state["fatigue"] > 0.8)
            output_sum += float(state["output"])
            atp_sum += float(state["atp"])
            calcium_sum += float(state.get("calcium", 0.0))
            self.active_biology.move_to_end(nid)

        n_bio = max(1, int(biology_ids.size))
        self._last_encode_stats = {
            "cascade_size": sdr.cascade_size,
            "biology_neurons_fired": int(biology_ids.size),
            "biology_fraction": round(float(biology_ids.size) / max(1, sdr.cascade_size), 6),
            "mean_output": output_sum / n_bio,
            "mean_atp": atp_sum / n_bio,
            "mean_calcium": calcium_sum / n_bio,
            "fatigue_warnings": fatigue_warnings,
        }
        sdr.meta["biology"] = self._last_encode_stats
        self._propagate_calcium_waves()

        # V30: Dynamic AgiKernel Gating
        # Feed the SDR representation into Modality 0 (Semantic) on the AgiKernel bus
        if sdr.weights.size > 0:
            norm_weights = sdr.weights / max(1e-5, float(sdr.weights.max()))
            self.agi_kernel.sense(0, norm_weights.tolist())
            
        # Tick the AGI kernel to update all subsystems
        mean_err, action_id = self.agi_kernel.tick()
        
        # Read updated neuromodulatory states back to the substrate
        self.dopamine = self.agi_kernel.dopamine
        self.noradrenaline = self.agi_kernel.noradrenaline
        self.serotonin = self.agi_kernel.serotonin
        self.acetylcholine = self.agi_kernel.acetylcholine
        self.arousal = self.agi_kernel.arousal
        self.throttle_cortex = self.agi_kernel.throttle_cortex

        for token in tokens:
            self._token_freq[token] += 1

        return sdr

    def hebbian_associate(
        self,
        sdr_a: SparseDistributedRepresentation,
        sdr_b: SparseDistributedRepresentation,
        dopamine: float = 1.0,
    ) -> int:
        """Reward-modulated Hebbian binding between two sparse populations."""
        # Feed reward to AGI kernel to update VTA & Hippocampal replay triggers
        self.agi_kernel.reward(dopamine)

        # V22: Dopaminergic Reward Prediction Error (RPE)
        rpe = dopamine - getattr(self, 'expected_reward', 0.0)
        self.expected_reward = 0.95 * getattr(self, 'expected_reward', 0.0) + 0.05 * dopamine
        dopamine = rpe # Override raw reward with prediction error

        if sdr_a.active_ids.size == 0 or sdr_b.active_ids.size == 0:
            return 0

        hot_pre = [int(nid) for nid in self._biology_subset(sdr_a) if int(nid) in self.active_biology]
        hot_post = [int(nid) for nid in self._biology_subset(sdr_b) if int(nid) in self.active_biology]
        hot = hot_pre + hot_post
        if hot:
            average_calcium = float(
                np.mean([self.active_biology[nid].biochem.calcium for nid in hot])
            )
            average_output = float(
                np.mean([
                    self.active_biology[nid].state_history[-1]
                    if self.active_biology[nid].state_history else 0.0
                    for nid in hot
                ])
            )
        else:
            average_calcium = 0.0
            average_output = 0.0

        try:
            effective_lr = rust_backend_shim.reward_modulated_stdp(
                max(0.05, average_output),
                1.0,
                dopamine,
                average_calcium,
            )
        except AttributeError:
            effective_lr = 0.05 * dopamine * (1.0 + average_calcium)
        effective_lr = float(np.clip(effective_lr, -0.5, 0.5))

        for nid in hot:
            self.active_biology[nid].modulate(reward=dopamine)
            self.active_biology.move_to_end(nid)

        n = min(sdr_a.active_ids.size, sdr_b.active_ids.size)
        if n == 0:
            return 0
        pre = sdr_a.active_ids[:n].astype(np.int64, copy=False)
        post = sdr_b.active_ids[:n].astype(np.int64, copy=False)
        if np.array_equal(pre, post) and n > 1:
            post = np.roll(post, -1)


        # V23: 6-Layer Canonical Microcircuit Structural Pruning
        structural_mask = np.ones(n, dtype=np.float32)
        from nlp_engine.layers.complex_neuron import PyramidalNeuron
        for i in range(n):
            raw_pre, raw_post = pre[i], post[i]
            pre_n = self.active_biology.get(int(raw_pre))
            post_n = self.active_biology.get(int(raw_post))
            if isinstance(pre_n, PyramidalNeuron) and isinstance(post_n, PyramidalNeuron):
                l_pre = pre_n.cortical_layer
                l_post = post_n.cortical_layer
                valid = False
                if l_pre == 4 and l_post in (2, 3): valid = True
                elif l_pre in (2, 3) and l_post == 5: valid = True
                elif l_pre == 5 and l_post == 6: valid = True
                elif l_pre == 6 and l_post == 4: valid = True
                elif l_pre == l_post: valid = True
                if not valid:
                    structural_mask[i] = 0.0

        deltas = (
            effective_lr
            * sdr_a.weights[:n].astype(np.float32, copy=False)
            * sdr_b.weights[:n].astype(np.float32, copy=False)
            * structural_mask
        )

        self.synapse_store.potentiate_batch(pre, post, deltas.astype(np.float32, copy=False))
        return int(n)

    def recall(
        self,
        sdr_cue: SparseDistributedRepresentation,
        k_top: int = 128,
    ) -> SparseDistributedRepresentation:
        """Fire a cue population and collect downstream recalled spikes."""
        post_ids, post_w = self.synapse_store.project(
            sdr_cue.active_ids.astype(np.int64, copy=False),
            sdr_cue.weights.astype(np.float32, copy=False),
            int(k_top),
        )
        if post_ids.size == 0:
            self._last_recall_stats = {"recalled": 0, "mean_atp": 0.0, "mean_output": 0.0}
            return SparseDistributedRepresentation(np.array([], dtype=np.int64), np.array([], dtype=np.float32))

        recalled = SparseDistributedRepresentation(post_ids, post_w, region=sdr_cue.region)
        biology_ids = self._biology_subset(recalled)
        self._instantiate_biology(biology_ids)

        output_sum = 0.0
        atp_sum = 0.0
        for raw_nid in biology_ids:
            nid = int(raw_nid)
            state = self.active_biology[nid].forward(nid)
            output_sum += float(state["output"])
            atp_sum += float(state["atp"])
            self.active_biology.move_to_end(nid)

        n_bio = max(1, int(biology_ids.size))
        self._last_recall_stats = {
            "recalled": int(post_ids.size),
            "biology_neurons_fired": int(biology_ids.size),
            "mean_atp": atp_sum / n_bio,
            "mean_output": output_sum / n_bio,
        }
        recalled.meta["biology"] = self._last_recall_stats
        self._propagate_calcium_waves()
        return recalled

    def sleep_consolidate(self, prune_fraction: float = 0.10) -> dict:
        """Sleep cycle: consolidate synapses and recharge hot biology."""
        prev = self.synapse_store.consolidation_fraction
        self.synapse_store.consolidation_fraction = float(prune_fraction)
        try:
            if self.synapse_store.disk_path:
                self.synapse_store._consolidate_to_disk()
            else:
                self.synapse_store._prune_coldest(fraction=prune_fraction)
        finally:
            self.synapse_store.consolidation_fraction = prev

        for neuron in self.active_biology.values():
            neuron.biochem.atp = min(1.0, neuron.biochem.atp + 0.25)
            neuron.biochem.calcium *= 0.6
            neuron.biochem.oxidative_stress *= 0.5

        while len(self.active_biology) > max(1, 100000) and self.adenosine > 1.0:
            self.active_biology.popitem(last=False)

        return self.stats()

    def stats(self) -> dict:
        if self.active_biology:
            atp = [n.biochem.atp for n in self.active_biology.values()]
            calcium = [n.biochem.calcium for n in self.active_biology.values()]
            mean_atp = float(np.mean(atp))
            mean_calcium = float(np.mean(calcium))
        else:
            mean_atp = 0.0
            mean_calcium = 0.0
        return {
            "total_neurons_virtual": self.total_neurons,
            "scale": self._scale_label(),
            "regions": {
                name: {"base": base, "size": size}
                for name, (base, size) in self._regions.items()
            },
            "k_active_per_concept": self.k_active,
            "biology_sample_size": self.biology_sample_size,
            "active_biology": len(self.active_biology),
            "adenosine": self.adenosine,
            "unique_tokens": len(self._token_freq),
            "mean_hot_atp": mean_atp,
            "mean_hot_calcium": mean_calcium,
            "last_encode": self._last_encode_stats,
            "last_recall": self._last_recall_stats,
            "synapse_store": self.synapse_store.stats(),
        }
