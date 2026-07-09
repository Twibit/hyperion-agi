from __future__ import annotations

import hashlib
import math


import rust_backend_shim

def _sigmoid(x: float) -> float:
    if x < -60.0:
        return 0.0
    if x > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _stable_seed(value: str) -> int:
    h = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


def _unit(seed: int, salt: int) -> float:
    x = (seed + 0x9E3779B97F4A7C15 * (salt + 1)) & ((1 << 64) - 1)
    x ^= (x >> 30)
    x = (x * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x ^= (x >> 27)
    x = (x * 0x94D049BB133111EB) & ((1 << 64) - 1)
    x ^= (x >> 31)
    return (x & ((1 << 53) - 1)) / float(1 << 53)


class AstrocyteSynapse:
    """Tripartite synapse with short-term plasticity and glial gating.

    This is still a compact toy model, but it captures the important
    biological levers that a learning substrate needs:
      - vesicle depletion/recovery
      - AMPA/NMDA-like fast/slow excitation
      - optional inhibitory sign
      - astrocyte calcium and gliotransmitter gain
      - reward-modulated eligibility traces
      - slow myelination for repeatedly useful paths
    """

    def __init__(
        self,
        base_weight: float = 1.0,
        receptor_type: str = "AMPA",
        release_probability: float = 0.35,
        nmda_ratio: float = 0.45,
    ):
        self.w_base = float(base_weight)
        self.receptor_type = receptor_type
        self.release_probability = float(release_probability)
        self.nmda_ratio = float(nmda_ratio)

        self.cleft_nt = 0.0
        self.astro_ca = 0.0
        self.glio_t = 0.0
        self.eligibility_trace = 0.0
        self.vesicle_pool = 1.0
        self.myelin = 1.0
        self._last_release = 0.0

    @property
    def is_dead(self) -> bool:
        """Microglia trigger: if base weight drops below 0.05, it's flagged for deletion."""
        return self.w_base < 0.05

    def step(self, spike: float, post_activity: float = 0.0) -> float:
        spike = max(0.0, min(1.5, float(spike)))
        post_activity = max(0.0, float(post_activity))

        release = spike * self.release_probability * self.vesicle_pool
        self.vesicle_pool = min(1.0, max(0.0, self.vesicle_pool + 0.035 - 0.12 * release))
        self._last_release = release

        if self.receptor_type == "GABA-A":
            self.cleft_nt = 0.4 * self.cleft_nt + 0.6 * release
            fast_current = self.cleft_nt
            slow_current = 0.0
            sign = -1.5
        elif self.receptor_type == "GABA-B":
            self.cleft_nt = 0.95 * self.cleft_nt + 0.05 * release
            fast_current = 0.0
            slow_current = self.cleft_nt
            sign = -0.8
        else:
            self.cleft_nt = 0.82 * self.cleft_nt + 0.18 * release
            nmda_gate = _sigmoid(5.0 * (post_activity - 0.25))
            fast_current = self.cleft_nt
            slow_current = self.nmda_ratio * self.cleft_nt * nmda_gate
            sign = 1.0

        self.astro_ca = 0.92 * self.astro_ca + 0.08 * self.cleft_nt
        if self.astro_ca > 0.35:
            self.glio_t = 0.86 * self.glio_t + 0.14 * (self.astro_ca - 0.35)
        else:
            self.glio_t *= 0.86

        # Eligibility is signed around a weak post-synaptic baseline, so
        # reward can potentiate useful coincidences and depress noise.
        self.eligibility_trace = (
            0.965 * self.eligibility_trace
            + release * (post_activity - 0.12)
        )

        glial_gain = 1.0 + 1.25 * self.glio_t
        return sign * self.w_base * self.myelin * glial_gain * (fast_current + slow_current)

    def post_update(self, post_activity: float) -> None:
        self.eligibility_trace = (
            0.97 * self.eligibility_trace
            + self._last_release * max(0.0, float(post_activity))
        )

    def modulate(self, reward: float, learning_rate: float = 0.01) -> None:
        reward = max(-3.0, min(3.0, float(reward)))
        calcium_gate = 0.25 + min(2.0, self.astro_ca)
        delta = learning_rate * reward * self.eligibility_trace * calcium_gate
        self.w_base = max(0.001, min(6.0, self.w_base + delta))

        # Useful, repeatedly co-active paths get slightly cheaper/faster.
        if reward > 0.0 and self.eligibility_trace > 0.0:
            self.myelin = min(2.5, self.myelin + learning_rate * 0.05 * reward)
        elif reward < 0.0:
            self.myelin = max(0.75, self.myelin + learning_rate * 0.02 * reward)


class DendriticCompartment:
    """Local dendritic branch with its own membrane and calcium state."""

    def __init__(
        self,
        n_synapses: int,
        seed: int,
        inhibitory: bool = False,
        base_weight: float = 0.8,
        threshold: float = 1.2,
        leak: float = 0.72,
    ):
        self.inhibitory = bool(inhibitory)
        self.threshold = float(threshold)
        self.leak = float(leak)
        self.potential = 0.0
        self.calcium = 0.0
        self.synapses: list[AstrocyteSynapse] = []
        for i in range(n_synapses):
            jitter = 0.75 + 0.5 * _unit(seed, i)
            rp = 0.22 + 0.35 * _unit(seed, 100 + i)
            nmda = 0.25 + 0.55 * _unit(seed, 200 + i)
            
            receptor = "AMPA"
            if inhibitory:
                receptor = "GABA-A" if i % 2 == 0 else "GABA-B"
                
            self.synapses.append(
                AstrocyteSynapse(
                    base_weight=base_weight * jitter,
                    receptor_type=receptor,
                    release_probability=rp,
                    nmda_ratio=nmda,
                )
            )

    @property
    def n_inputs(self) -> int:
        return len(self.synapses)

    def forward(self, inputs: list[float], post_activity: float = 0.0) -> float:
        current = 0.0
        for syn, inp in zip(self.synapses, inputs):
            current += syn.step(inp, post_activity)

        self.potential = self.leak * self.potential + current
        if self.inhibitory:
            local = -abs(self.potential)
            self.potential *= 0.55
        elif self.potential > self.threshold:
            local = self.potential - self.threshold
            self.potential *= 0.35
        else:
            local = 0.15 * self.potential

        self.calcium = 0.94 * self.calcium + 0.06 * abs(local)
        return local

    def post_update(self, post_activity: float) -> None:
        for syn in self.synapses:
            syn.post_update(post_activity)

    def modulate(self, reward: float, learning_rate: float = 0.01) -> None:
        for syn in self.synapses:
            syn.modulate(reward, learning_rate)

    def prune(self) -> int:
        """Microglia physically delete dead synapses."""
        original_count = len(self.synapses)
        self.synapses = [s for s in self.synapses if not s.is_dead]
        return original_count - len(self.synapses)


class DendriticTree:
    """Multi-compartment dendritic tree.

    The shape is intentionally compact enough for thousands of hot
    neurons, but it is no longer a single weighted sum: basal branches
    detect local evidence, apical branches provide context, inhibitory
    branches shunt runaway activity, and back-propagating action
    potentials write eligibility traces.
    """

    def __init__(self, seed: int):
        self.basal = [
            DendriticCompartment(6, seed + 101 * i, base_weight=0.75, threshold=1.15)
            for i in range(4)
        ]
        self.apical = [
            DendriticCompartment(5, seed + 701 * i, base_weight=0.45, threshold=0.85)
            for i in range(2)
        ]
        self.inhibitory = [
            DendriticCompartment(4, seed + 1301 * i, inhibitory=True, base_weight=0.55, threshold=0.6)
            for i in range(2)
        ]

    @property
    def n_basal_inputs(self) -> int:
        return sum(c.n_inputs for c in self.basal)

    @property
    def n_apical_inputs(self) -> int:
        return sum(c.n_inputs for c in self.apical)

    @property
    def n_inhibitory_inputs(self) -> int:
        return sum(c.n_inputs for c in self.inhibitory)

    def _split(self, values: list[float], compartments: list[DendriticCompartment]) -> list[list[float]]:
        out = []
        cursor = 0
        for comp in compartments:
            n = comp.n_inputs
            out.append(values[cursor:cursor + n])
            cursor += n
        return out

    def forward(
        self,
        basal_inputs: list[float],
        apical_inputs: list[float],
        inhibitory_inputs: list[float],
    ) -> float:
        basal_drive = sum(
            comp.forward(chunk)
            for comp, chunk in zip(self.basal, self._split(basal_inputs, self.basal))
        )
        apical_drive = sum(
            comp.forward(chunk, post_activity=max(0.0, basal_drive))
            for comp, chunk in zip(self.apical, self._split(apical_inputs, self.apical))
        )
        inhibitory_drive = sum(
            comp.forward(chunk, post_activity=max(0.0, basal_drive))
            for comp, chunk in zip(self.inhibitory, self._split(inhibitory_inputs, self.inhibitory))
        )

        apical_plateau = _sigmoid(apical_drive - 0.5)
        shunt = _sigmoid(abs(inhibitory_drive) - 0.35)
        somatic_voltage = max(0.0, basal_drive * (1.0 + 0.9 * apical_plateau) - 0.75 * shunt)

        for comp in self.basal + self.apical + self.inhibitory:
            comp.post_update(somatic_voltage)

        return somatic_voltage

    def modulate(self, reward: float, learning_rate: float = 0.01) -> None:
        for comp in self.basal + self.apical + self.inhibitory:
            comp.modulate(reward, learning_rate)

    def prune(self) -> int:
        pruned = 0
        for comp in self.basal + self.apical + self.inhibitory:
            pruned += comp.prune()
        return pruned


class NeuronBiochemistry:
    """Intracellular ATP, calcium buffering, stress and fatigue."""

    def __init__(self):
        self.atp = 1.0
        self.calcium = 0.0
        self.bdnf = 0.0  # Brain-Derived Neurotrophic Factor
        self.oxidative_stress = 0.0
        self.lactate = 0.0
        self.atp_cost_per_unit = 0.035
        self.atp_regen_rate = 0.014
        self.ca_influx_per_unit = 0.075
        self.ca_buffer_tau = 28.0

    def update(self, somatic_voltage: float, spiked: bool = False):
        v = max(0.0, float(somatic_voltage))
        if spiked or v > 1.0:
            self.atp = max(0.0, self.atp - self.atp_cost_per_unit * max(1.0, v))
            self.calcium += self.ca_influx_per_unit * v
            self.lactate = min(1.0, self.lactate + 0.03 * v)
        else:
            regen = self.atp_regen_rate * (1.0 + 0.5 * self.lactate)
            self.atp = min(1.0, self.atp + regen)
            self.lactate *= 0.96

        self.calcium *= math.exp(-1.0 / self.ca_buffer_tau)
        self.oxidative_stress = 0.985 * self.oxidative_stress + 0.015 * max(0.0, self.calcium - 0.75)
        self.bdnf = 0.99 * self.bdnf + 0.01 * max(0.0, v - 0.5)

    def get_fatigue_penalty(self) -> float:
        atp_penalty = 2.0 / (1.0 + math.exp((self.atp - 0.3) * 12.0))
        stress_penalty = min(1.0, self.oxidative_stress)
        return min(2.5, atp_penalty + stress_penalty)



class PurkinjeDendriticTree(DendriticTree):
    def __init__(self, seed: int):
        self.basal = [
            DendriticCompartment(10, seed + 101 * i, base_weight=0.75, threshold=1.15)
            for i in range(12)
        ]
        self.apical = []
        self.inhibitory = [
            DendriticCompartment(8, seed + 1301 * i, inhibitory=True, base_weight=0.6, threshold=0.5)
            for i in range(4)
        ]

class VonEconomoDendriticTree(DendriticTree):
    def __init__(self, seed: int):
        self.basal = [DendriticCompartment(2, seed + 101, base_weight=0.9, threshold=0.8)]
        self.apical = [DendriticCompartment(2, seed + 701, base_weight=0.9, threshold=0.8)]
        self.inhibitory = []

class BiologicalNeuron:
    """Sparse materialized neuron for the 86B/100B/100T virtual substrate."""

    def __init__(self, neuron_id: str):
        self.neuron_id = neuron_id
        self.seed = _stable_seed(neuron_id)
        self.dendrites = DendriticTree(self.seed)
        self.biochem = NeuronBiochemistry()

        self.membrane = -0.05 + 0.1 * _unit(self.seed, 9001)
        self.membrane_leak = 0.82 # Baseline
        self.threshold = 0.85 + 0.35 * _unit(self.seed, 9002)
        self.refractory_steps = 0
        self.firing_rate_ema = 0.0
        self.target_rate = 0.015 + 0.02 * _unit(self.seed, 9003)
        self.homeostatic_gain = 1.0
        self.axon_myelin = 1.0
        self.state_history: list[float] = []
        
        # neurorust physics engine integration
        self.izhikevich = rust_backend_shim.NativeIzhikevich("RegularSpiking", 1.0)

    def _drive(self, token_hash: int, n: int, salt: int) -> list[float]:
        x = (int(token_hash) ^ self.seed ^ (salt * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
        values = []
        for _ in range(n):
            x ^= (x << 13) & ((1 << 64) - 1)
            x ^= (x >> 7)
            x ^= (x << 17) & ((1 << 64) - 1)
            bit = x & 1
            amp = 0.35 + ((x >> 8) & 0xFF) / 255.0
            values.append(float(bit) * amp)
        return values

    def forward(self, token_hash: int) -> dict:
        if self.refractory_steps > 0:
            self.refractory_steps -= 1
            self.biochem.update(0.0, spiked=False)
            fatigue = self.biochem.get_fatigue_penalty()
            return {
                "voltage": 0.0,
                "atp": self.biochem.atp,
                "calcium": self.biochem.calcium,
                "fatigue": fatigue,
                "output": 0.0,
                "spiked": False,
                "threshold": self.threshold,
            }

        basal = self._drive(token_hash, self.dendrites.n_basal_inputs, 11)
        apical = self._drive(token_hash, self.dendrites.n_apical_inputs, 23)
        inhibitory = self._drive(token_hash, self.dendrites.n_inhibitory_inputs, 37)

        dendritic_voltage = self.dendrites.forward(basal, apical, inhibitory)
        fatigue = self.biochem.get_fatigue_penalty()
        metabolic_gain = max(0.05, self.biochem.atp) * max(0.0, 1.0 - 0.35 * fatigue)
        self.homeostatic_gain = max(0.25, min(2.5, self.homeostatic_gain))

        # Pass the total dendritic drive to the Native Izhikevich physics engine
        # Scale drive by 10x to match Izhikevich uA/cm2 scale
        drive_current = dendritic_voltage * metabolic_gain * self.homeostatic_gain * 10.0
        v_mem, spiked = self.izhikevich.step(1.0, drive_current)
        
        spike_strength = 1.0 if spiked else max(0.0, (v_mem + 65.0) / 100.0)

        self.biochem.update(dendritic_voltage, spiked=spiked)
        output = spike_strength * self.axon_myelin * max(0.0, 1.0 - 0.25 * fatigue)

        self.firing_rate_ema = 0.995 * self.firing_rate_ema + 0.005 * (1.0 if spiked else 0.0)
        rate_error = self.firing_rate_ema - self.target_rate
        self.threshold = max(0.45, min(2.5, self.threshold + 0.02 * rate_error))
        self.homeostatic_gain *= math.exp(-0.01 * rate_error)

        self.state_history.append(output)
        if len(self.state_history) > 32:
            del self.state_history[:-32]

        return {
            "voltage": dendritic_voltage,
            "atp": self.biochem.atp,
            "calcium": self.biochem.calcium,
            "fatigue": fatigue,
            "output": output,
            "spiked": spiked,
            "threshold": self.threshold,
        }

    def modulate(self, reward: float, learning_rate: float = 0.01) -> None:
        self.dendrites.modulate(reward, learning_rate)
        # V30: Dynamic Superhuman Hyper-Myelination
        # No hardcoded upper bounds! We use an asymptotic growth curve:
        # The thicker the myelin gets, the more reward is required to keep growing it.
        # This prevents math explosions while mathematically allowing infinite growth.
        if reward > 0.0 and self.firing_rate_ema > 0.0:
            growth_rate = 0.015 / max(1.0, self.axon_myelin)
            self.axon_myelin += growth_rate * reward
        elif reward < 0.0:
            self.axon_myelin = max(0.7, self.axon_myelin + 0.001 * reward)

    def prune(self) -> int:
        """Summons Microglia to prune dead synapses, and uses BDNF to sprout new ones."""
        pruned = self.dendrites.prune()
        
        # V29: BDNF Structural Plasticity (Dynamic Growth)
        # If the cell is highly stimulated, it physically grows a new basal dendritic branch to increase capacity.
        if self.biochem.bdnf > 0.8:
            new_seed = self.seed + len(self.dendrites.basal) * 999
            import random
            self.dendrites.basal.append(
                DendriticCompartment(n_synapses=random.randint(4, 8), seed=new_seed, base_weight=0.5, threshold=1.0)
            )
            self.biochem.bdnf *= 0.5 # Consume BDNF to grow the branch
            
        # V30: Dynamic Superhuman Eidetic Memory
        # Instead of a hardcoded threshold (2.5), structural pruning is resisted probabilistically
        # based on how much myelin the cell has built up. Infinite myelin = infinite structural immunity.
        import random
        survival_chance = 1.0 - (1.0 / max(1.0, self.axon_myelin))
        if random.random() > survival_chance:
            self.dendrites.basal = [b for b in self.dendrites.basal if len(b.synapses) > 0]
        
        return pruned

class PyramidalNeuron(BiologicalNeuron):
    def __init__(self, neuron_id: str, hemisphere: str = "left"):
        super().__init__(neuron_id)
        self.cortical_layer = (int(neuron_id) % 6) + 1
        # Left Brain: High leak (fast, sharp sequential integration -> Math/Language)
        # Right Brain: Low leak (slow, broad spatial integration -> Holistic/Creativity)
        self.membrane_leak = 0.55 if hemisphere == "left" else 0.92

class PurkinjeNeuron(BiologicalNeuron):
    def __init__(self, neuron_id: str):
        super().__init__(neuron_id)
        self.dendrites = PurkinjeDendriticTree(self.seed)

    def modulate(self, reward: float, learning_rate: float = 0.01) -> None:
        super().modulate(-reward, learning_rate * 1.5)

class VonEconomoNeuron(BiologicalNeuron):
    def __init__(self, neuron_id: str, hemisphere: str = "left"):
        super().__init__(neuron_id)
        self.dendrites = VonEconomoDendriticTree(self.seed)
        # Left Brain Logic vs Right Brain Intuition
        self.membrane_leak = 0.45 if hemisphere == "left" else 0.95

    def forward(self, token_hash: int) -> dict:
        self.biochem.atp = 1.0
        self.biochem.oxidative_stress = 0.0
        return super().forward(token_hash)

ComplexNeuron = BiologicalNeuron
