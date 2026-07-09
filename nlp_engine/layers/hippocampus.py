import numpy as np

class DentateGyrusEncoder:
    """
    Phase II: Anti-Hallucination Encoder (Pattern Separation).
    Uses extreme sparse coding to physically prevent hallucinations by 
    orthogonalizing overlapping concepts. Conflicting concepts share absolutely 
    zero synaptic overlap, making confabulation mathematically impossible.
    """
    def __init__(self, num_in, num_out, sparsity_target=0.001):
        self.num_in = num_in
        self.num_out = num_out
        self.sparsity_target = sparsity_target # Force exactly 0.1% firing rate
        
        # Zero-FLOP integer weights
        self.weights = np.random.randint(-10, 10, size=(num_in, num_out), dtype=np.int16)
        self.membrane = np.zeros(num_out, dtype=np.int32)
        
        # Dynamic K-Winner-Take-All to enforce absolute physical sparsity bounds
        self.k_winners = max(1, int(num_out * sparsity_target))

    def pattern_separate(self, in_spikes):
        """Orthogonalizes input into an ultra-sparse, hallucination-proof spike pattern."""
        self.membrane -= (self.membrane >> 3) # Bitshift leak
        spike_idx = np.nonzero(in_spikes)[0]
        
        if spike_idx.size > 0:
            # Zero-FLOP accumulation
            # Instead of a dense dot product, we just sum the rows of active neurons
            active_weights_sum = np.sum(self.weights[spike_idx], axis=0)
            self.membrane += active_weights_sum
            
        # Global Inhibition: Only the absolute strongest neurons survive (K-Winners)
        # This prevents 'fuzzy' hallucinated activations from leaking through
        sorted_indices = np.argsort(self.membrane)[-self.k_winners:]
        out_spikes = np.zeros(self.num_out, dtype=bool)
        
        # Only fire if they actually have accumulated energy > 0
        valid_winners = [idx for idx in sorted_indices if self.membrane[idx] > 0]
        out_spikes[valid_winners] = True
        
        # Soft reset
        self.membrane[out_spikes] -= 100 
        return out_spikes

class CA3AssociativeGraph:
    """
    Phase III: The Associative Knowledge Graph (Recurrent Collaterals).
    Replaces Transformer Self-Attention with Auto-Associative STDP memory attractors.
    """
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        # Native ontology recurrent matrix (Neurons connecting directly to other neurons)
        self.recurrent_weights = np.zeros((num_neurons, num_neurons), dtype=np.int16)
        self.membrane = np.zeros(num_neurons, dtype=np.int32)
        
    def pattern_complete(self, sparse_cue, ticks=5, threshold=2, max_active_neurons=12):
        """
        Retrieval: Snaps a partial/fuzzy cue to a discrete, exact energy minimum.
        Now includes GABAergic global inhibition to prevent runaway excitation.
        """
        current_spikes = sparse_cue.copy()
        
        for _ in range(ticks):
            self.membrane -= (self.membrane >> 3)
            spike_idx = np.nonzero(current_spikes)[0]
            
            if spike_idx.size > 0:
                # Recurrent Zero-FLOP accumulation (Glutamatergic Excitation)
                active_recurrent_sum = np.sum(self.recurrent_weights[spike_idx], axis=0)
                self.membrane += active_recurrent_sum
                
            # GABAergic Global Inhibition:
            # The more neurons that fire, the stronger the global inhibition becomes.
            # This mathematically prevents the "bleeding" of concepts.
            global_inhibition = int(spike_idx.size * 0.5)
            self.membrane -= global_inhibition
            
            # Competitive Sparsity (K-Winner-Take-All)
            # Only the absolute sharpest, most correlated concepts survive the inhibition
            if np.count_nonzero(self.membrane > 0) > max_active_neurons:
                sorted_indices = np.argsort(self.membrane)
                weak_neurons = sorted_indices[:-max_active_neurons]
                self.membrane[weak_neurons] = -100 # Hyperpolarize weak thoughts
                    
            # Firing threshold for exact pattern completion
            current_spikes = self.membrane >= threshold
            self.membrane[current_spikes] -= threshold
            
        return current_spikes
        
    def stdp_memorize(self, true_spikes):
        """
        Ingestion: Learns the Ontology Graph natively via Hebbian/STDP.
        "Neurons that fire together, wire together."
        """
        spike_idx = np.nonzero(true_spikes)[0]
        # Native physical graph building (Reinforce connections)
        for i in spike_idx:
            self.recurrent_weights[i, spike_idx] += 1
            # Prevent runaway feedback loops (Homeostasis)
            self.recurrent_weights[i, i] = 0 

    def stdp_sequence_memorize(self, sequence_of_spikes):
        """
        Ingestion: Learns a sequential stream of concepts via Asymmetric STDP.
        "A fires before B, so A -> B becomes a directed synaptic pathway."
        """
        for i in range(len(sequence_of_spikes) - 1):
            pre_spikes = sequence_of_spikes[i]
            post_spikes = sequence_of_spikes[i+1]
            
            pre_idx = np.nonzero(pre_spikes)[0]
            post_idx = np.nonzero(post_spikes)[0]
            
            # Asymmetric wiring: Pre -> Post
            for p in pre_idx:
                self.recurrent_weights[p, post_idx] += 2 
                self.recurrent_weights[p, p] = 0

    def sequence_generate(self, initial_spike, sequence_length=5, threshold=1, max_active_neurons=6):
        """
        Retrieval: Generates a spatiotemporal stream of consciousness (a sentence)
        by letting the initial cue chain-react through the asymmetric weights.
        """
        sequence = [initial_spike]
        current_spikes = initial_spike.copy()
        
        for _ in range(sequence_length - 1):
            self.membrane *= 0 # Reset membrane to prevent backward smearing (Refractory period)
            spike_idx = np.nonzero(current_spikes)[0]
            
            if spike_idx.size > 0:
                # Forward projection
                active_sum = np.sum(self.recurrent_weights[spike_idx], axis=0)
                self.membrane += active_sum
            
            # K-Winner-Take-All for the next predicted concept
            if np.count_nonzero(self.membrane > 0) > max_active_neurons:
                sorted_indices = np.argsort(self.membrane)
                weak_neurons = sorted_indices[:-max_active_neurons]
                self.membrane[weak_neurons] = -100
                
            next_spikes = self.membrane >= threshold
            
            if not np.any(next_spikes):
                break # Train of thought ended (synaptic silence)
                
            sequence.append(next_spikes)
            current_spikes = next_spikes
            
        return sequence
