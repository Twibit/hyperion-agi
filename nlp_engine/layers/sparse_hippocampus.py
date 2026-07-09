import numpy as np
from collections import defaultdict

class SparseHippocampus:
    """
    Biological Trace STDP Network (Linear RNN / Continuous Attention).
    Replaces discrete N-Grams with a decaying cognitive trace.
    """
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        # Synaptic weights: pre_neuron -> {post_neuron: weight}
        self.synapses = defaultdict(lambda: defaultdict(float))
        
    def stdp_sequence_memorize(self, sequence_of_spikes):
        """Wires synapses based on Biological Trace Hebbian Learning."""
        trace = np.zeros(self.num_neurons, dtype=np.float32)
        
        for spike in sequence_of_spikes:
            post_active = np.nonzero(spike)[0]
            if len(post_active) == 0:
                continue
                
            pre_active = np.nonzero(trace)[0]
            
            # STDP: Strengthen connections from active trace to new spike
            for pre in pre_active:
                t_val = trace[pre]
                if t_val > 0.05: # Optimization: ignore negligible trace
                    for post in post_active:
                        self.synapses[pre][post] += float(t_val)
                        
            # Update biological trace
            trace *= 0.6  # Decay factor
            trace[post_active] += 1.0 # Add current spike

    def sequence_generate(self, initial_spike, sequence_length=150, threshold=0.0, max_active=3, temperature=0.5):
        """Generates stream of consciousness using Continuous Trace Prediction."""
        sequence = [initial_spike]
        dendritic_pathway = [] # For RLHF Neuromodulation
        
        trace = np.zeros(self.num_neurons, dtype=np.float32)
        current_spike = initial_spike
        
        for _ in range(sequence_length - 1):
            # Update trace with current thought
            trace *= 0.6
            active_current = np.nonzero(current_spike)[0]
            if len(active_current) > 0:
                trace[active_current] += 1.0
                
            membrane = np.zeros(self.num_neurons, dtype=np.float32)
            pre_active = np.nonzero(trace)[0]
            
            # Forward pass: Project trace through synapses
            for pre in pre_active:
                t_val = trace[pre]
                if t_val > 0.05:
                    for post, weight in self.synapses[pre].items():
                        membrane[post] += t_val * weight
            
            # Add stochastic temperature (Simulated Annealing)
            if temperature > 0:
                active_mask = membrane > 0
                if np.any(active_mask):
                    noise = np.random.normal(0, temperature, size=np.count_nonzero(active_mask))
                    membrane[active_mask] += noise.astype(np.float32)
                    
            # K-Winner-Take-All
            if np.count_nonzero(membrane > 0) > max_active:
                sorted_indices = np.argsort(membrane)
                weak = sorted_indices[:-max_active]
                membrane[weak] = 0
                
            next_spikes = membrane > threshold
            if not np.any(next_spikes):
                # Spontaneous noise to prevent complete death
                if len(self.synapses) > 0:
                    random_pre = list(self.synapses.keys())[np.random.randint(0, len(self.synapses))]
                    random_post = list(self.synapses[random_pre].keys())[0]
                    next_spikes[random_post] = True
                else:
                    break
                
            sequence.append(next_spikes)
            
            # Record pathway for Dopamine/Cortisol targeting
            next_active = np.nonzero(next_spikes)[0]
            for pre in pre_active:
                if trace[pre] > 0.5: # Primary drivers
                    for target in next_active:
                        dendritic_pathway.append((pre, target))
                        
            current_spike = next_spikes
            
        return sequence, dendritic_pathway

    def apply_dopamine(self, reward_signal, dendritic_pathway):
        """Strengthens all synapses that led to the good output."""
        for pre, post in dendritic_pathway:
            self.synapses[pre][post] += reward_signal

    def apply_cortisol(self, trauma_signal, dendritic_pathway):
        """Synaptic depotentiation: prunes synapses that caused terrible output."""
        for pre, post in dendritic_pathway:
            self.synapses[pre][post] -= trauma_signal
            if self.synapses[pre][post] <= 0:
                del self.synapses[pre][post]

    def sharp_wave_ripple(self, universal_cortex):
        """
        Hippocampal Replay: 
        Replays strong synaptic pathways at high speed to train the Neocortex (UniversalCortex) offline.
        """
        print("[Hippocampus] Initiating Sharp-Wave Ripple. Replaying memories for neocortical consolidation...")
        import time
        if len(self.synapses) == 0:
            return
            
        # Find the strongest connections
        strong_pairs = []
        for pre, posts in self.synapses.items():
            for post, weight in posts.items():
                if weight > 1.0: # Only replay significant memories
                    strong_pairs.append((pre, post, weight))
                    
        # Sort by weight (strongest first)
        strong_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Take top 10 for replay
        for pre, post, weight in strong_pairs[:10]:
            print(f"  -> [Sharp-Wave] Replaying trace ({pre} -> {post}) at 10x speed...")
            # We would ideally reconstruct the words from the hash, but for now we simulate the reinforcement
            universal_cortex.dopamine_boost = 2.0
            # Simulating the offline reinforcement of these pathways
            time.sleep(0.01) # 10x biological speed (fast replay)

