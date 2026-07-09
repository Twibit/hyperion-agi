import numpy as np
import math
from nlp_engine.layers.complex_neuron import ComplexNeuron

class CellularMatrix:
    """
    Simulates the true intracellular biology of neurons (Organelles).
    Instead of instantiating 86 billion classes (which would crash RAM),
    we use Sparse Intracellular Tracking using the ComplexNeuron object structure
    for neurons currently in the active cascade.
    """
    def __init__(self, golden_ratio=1.6180339887):
        self.PHI = golden_ratio
        self.PI = math.pi
        
        # State tracking for anchor cells: Neuron ID -> ComplexNeuron instance
        self.cells = {}
        self.global_clock = 0.0 # Time progression for oscillations
        
        # Continuous Mathematical Hidden State Vector
        self.hidden_dim = 256
        self.hidden_state = np.zeros(self.hidden_dim)
        
        # --- Thermodynamic System ---
        self.global_atp = 100.0 # 100% Full Energy
        self.current_wattage = 20.0 # Resting brain baseline
        self.adrenaline_level = 0.0 # Endocrine override tracker
        
    def metabolic_burn(self, dopamine_boost: float, token_count: int) -> float:
        """Burns global ATP based on cognitive load. Incorporates Flow State and Adrenaline."""
        self.current_wattage = 20.0 + (dopamine_boost * 15.0) + (token_count * 0.1)
        
        # 1. Flow State (Continuous Aerobic Recovery)
        # Mitochondria constantly regenerate ATP in the background
        aerobic_recovery_rate = 5.0
        
        # 2. Adrenaline (Anaerobic Limit Break)
        if dopamine_boost > 4.0:
            self.adrenaline_level = 1.0
            
        burn_rate = self.current_wattage * 0.05
        net_change = (aerobic_recovery_rate * 0.05) - burn_rate
        
        # Update ATP, allowing for debt if adrenaline is active
        if self.adrenaline_level > 0:
            self.global_atp = max(-50.0, self.global_atp + net_change) # Push into debt, cap at -50.0
            self.adrenaline_level -= 0.1 # Adrenaline wears off over time
            if self.global_atp <= -50.0:
                self.adrenaline_level = 0.0 # Safety Guard: Force adrenaline off to prevent death
        else:
            self.global_atp = max(0.0, min(100.0, self.global_atp + net_change))
            
        # 3. Fatigue Penalty (Brain Fog)
        if self.adrenaline_level <= 0 and self.global_atp < 20.0:
            return 0.6 # Mild plasticity reduction — never freezes learning
            
        return 1.0 # Normal plasticity
        
    def fast_self_heal(self):
        """Adaptive fast recovery. The more exhausted the brain, the harder mitochondria work to recover ATP."""
        if self.global_atp < 100.0:
            recovery_rate = max(0.5, (100.0 - self.global_atp) * 0.15) # Exponentially fast recovery when depleted
            self.global_atp = min(100.0, self.global_atp + recovery_rate)
            self.current_wattage = max(20.0, self.current_wattage - 5.0) # Gradually cool down down to 20W
        
    def _get_brainwaves(self) -> dict:
        t = self.global_clock
        return {
            'delta': math.sin(2 * self.PI * (t / 40.0)),
            'theta': math.sin(2 * self.PI * (t / 20.0)),
            'alpha': math.sin(2 * self.PI * (t / 10.0)),
            'gamma': math.sin(2 * self.PI * (t / 2.0))
        }

    def _genesis(self, nid: int):
        """Initializes the complete intracellular anatomy of a specific neuron using ComplexNeuron."""
        self.cells[nid] = ComplexNeuron(str(nid))
        # Override initial ATP to match old scale of 100 for compatibility
        self.cells[nid].biochem.atp = 1.0 # 1.0 is 100% in ComplexNeuron

    def process_action_potential(self, anchor_neurons: list) -> dict:
        """
        Flows electrical current into the anchor cells. 
        Simulates the entire biological cascade using Dendritic Trees and Tripartite Synapses.
        """
        self.global_clock += 1.0
        waves = self._get_brainwaves()
        gamma_shift = waves['gamma'] # Fast wave for intense focus
        
        successful_fires = []
        exhausted_failures = 0
        synaptic_fatigue = 0
        
        for nid in anchor_neurons:
            if nid not in self.cells:
                self._genesis(nid)
                
            cell = self.cells[nid]
            
            # Forward the complex logic tree
            # We use the neuron's ID mixed with the global clock as the pseudo-hash
            token_hash = abs(hash(str(nid) + str(self.global_clock)))
            
            state = cell.forward(token_hash)
            
            if state['fatigue'] >= 1.0:
                exhausted_failures += 1
            elif state['output'] > 0.5:
                successful_fires.append(nid)
            else:
                synaptic_fatigue += 1
                
        return {
            'fired': successful_fires,
            'exhausted': exhausted_failures,
            'fatigued': synaptic_fatigue
        }
        
    def cellular_respiration(self):
        """
        Called during sleep. 
        Mitochondria regenerate ATP.
        Driven by DELTA waves.
        """
        self.global_clock += 1.0
        waves = self._get_brainwaves()
        # Delta wave massively boosts physical healing and consolidation
        delta_boost = max(0.0, waves['delta']) 
        
        # Sleep forces an absolute return to Homeostasis
        self.global_atp = 100.0
        self.current_wattage = 5.0 # Deep sleep burns only 5 Watts
        
        for cell in self.cells.values():
            # In ComplexNeuron, max ATP is 1.0
            cell.biochem.atp = min(1.0, cell.biochem.atp + (0.15 * self.PHI * (1.0 + delta_boost)))

    def update_hidden_state(self, token_str: str) -> np.ndarray:
        """
        Updates the Hidden State Vector during continuous thought generation.
        Uses Golden Ratio (PHI) phase-shifting to ground the brain in 'time'.
        """
        self.global_clock += (self.PHI / self.PI)
        token_energy = (abs(hash(token_str)) % 1000) / 1000.0
        
        new_state = np.zeros(self.hidden_dim)
        for i in range(self.hidden_dim):
            phase = self.global_clock + (i * self.PHI)
            new_state[i] = math.tanh(0.85 * self.hidden_state[i] + token_energy * math.cos(phase))
            
        self.hidden_state = new_state
        return self.hidden_state

    def synaptic_pruning(self):
        """
        Prunes the CellularMatrix to prevent infinite memory growth.
        Caps the active cells dictionary at 50,000 using an energy-based logic,
        removing cells with the lowest ATP.
        """
        max_cells = 50000
        if len(self.cells) > max_cells:
            # Sort cells by ATP (descending) to keep the healthiest/most active.
            sorted_cells = sorted(
                self.cells.items(), 
                key=lambda item: item[1].biochem.atp, 
                reverse=True
            )
            self.cells = dict(sorted_cells[:max_cells])
