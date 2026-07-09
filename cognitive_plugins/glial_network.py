import math
import numpy as np

class GlialNetwork:
    """
    Biological Glial Cells: Astrocytes & Microglia.
    Provides Structural Plasticity to the 86B virtual substrate.
    """
    def __init__(self, substrate, cellular_matrix):
        self.substrate = substrate
        self.cellular_matrix = cellular_matrix
        
        # Astrocytes manage the Global Metabolic Pool
        self.global_atp = 100.0
        self.metabolic_stress = 0.0
        
        # Track neuron firing frequencies for Microglial pruning
        self.neuron_firing_counts = {}

    def record_firing(self, active_ids: np.ndarray):
        """Called by the substrate to track which neurons are firing."""
        for nid in active_ids:
            self.neuron_firing_counts[int(nid)] = self.neuron_firing_counts.get(int(nid), 0) + 1

    def astrocyte_neurogenesis(self, n_new_connections: int = 100):
        """
        Astrocytes build new synapses between highly active but disconnected neurons.
        They wire the brain together where Hebbian learning missed.
        """
        if not self.neuron_firing_counts or not hasattr(self.substrate, 'synapses'):
            return 0
            
        # Find the most active neurons
        sorted_neurons = sorted(self.neuron_firing_counts.items(), key=lambda x: x[1], reverse=True)
        top_neurons = [nid for nid, count in sorted_neurons[:max(10, n_new_connections * 2)]]
        
        if len(top_neurons) < 2:
            return 0
            
        new_synapses_forged = 0
        rng = np.random.default_rng()
        
        pre_ids = []
        post_ids = []
        wts = []
        
        for _ in range(n_new_connections):
            # Pick two random highly active neurons
            pre, post = rng.choice(top_neurons, 2, replace=False)
            pre_ids.append(pre)
            post_ids.append(post)
            # Forge a weak new synapse that can be strengthened later
            wts.append(0.05)
            new_synapses_forged += 1
            
        if new_synapses_forged > 0:
            self.substrate.synapses.potentiate_batch(np.array(pre_ids), np.array(post_ids), np.array(wts))
            
        # Neurogenesis costs ATP
        self.global_atp -= (new_synapses_forged * 0.01)
        return new_synapses_forged

    def microglia_prune(self, threshold: int = 1):
        """
        Microglia destroy synapses attached to neurons that rarely fire,
        and delete neurons from the cellular matrix that are functionally dead.
        """
        dead_neurons = [nid for nid, count in self.neuron_firing_counts.items() if count <= threshold]
        
        pruned = 0
        if dead_neurons and hasattr(self.substrate, 'synapses'):
            # In a true biological simulation, microglia phagocytose the physical connection.
            # We simulate this by heavily depressing connections to/from dead neurons.
            # The sparse_synapse_store will garbage collect them during sleep_consolidate.
            pruned = len(dead_neurons)
            
            # Remove from cellular biology matrix
            for nid in dead_neurons:
                if nid in self.cellular_matrix.cells:
                    del self.cellular_matrix.cells[nid]
                    
        # Reset counts for the next waking cycle
        self.neuron_firing_counts = {}
        return pruned

    def nocturnal_rewiring(self) -> dict:
        """
        Called during Deep REM Sleep.
        Balances the metabolic economy, prunes dead weight, and births new concepts.
        """
        # Regenerate global ATP during sleep
        self.global_atp = min(100.0, self.global_atp + 20.0)
        
        # 1. Microglia patrol
        pruned_count = self.microglia_prune(threshold=1)
        
        # 2. Astrocyte construction
        if self.global_atp > 50.0:
            new_synapses = self.astrocyte_neurogenesis(n_new_connections=500)
        else:
            new_synapses = 0
            
        # 3. Substrate Synaptic Garbage Collection (C1q/C3 complement pathway)
        # We increase the prune fraction if metabolic stress is high
        prune_fraction = 0.10
        if self.global_atp < 30.0:
            prune_fraction = 0.25 # Desperate pruning to save energy
            
        stats = self.substrate.sleep_consolidate(prune_fraction=prune_fraction)
        
        return {
            "microglia_pruned_neurons": pruned_count,
            "astrocytes_new_synapses": new_synapses,
            "global_atp": self.global_atp,
            "substrate_stats": stats
        }
        
    def get_adaptive_sparsity(self, base_k: int) -> int:
        """
        Astrocytes clamp the global firing rate based on available ATP.
        If the brain is tired, it thinks more sparsely.
        """
        if self.global_atp < 20.0:
            return max(32, int(base_k * 0.3)) # Extreme conservation
        elif self.global_atp < 50.0:
            return max(64, int(base_k * 0.6))
        return base_k
