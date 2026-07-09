import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from nlp_engine.layers.hippocampus import CA3AssociativeGraph

class NativeLanguageEncoder:
    """
    Translates words into discrete Sparse Spikes, completely bypassing Transformers.
    """
    def __init__(self, spike_dim):
        self.spike_dim = spike_dim
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_spikes = {}
        self.vocab_counter = 0
        np.random.seed(42) # For reproducible neural spikes

    def get_spike(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.vocab_counter
            self.idx_to_word[self.vocab_counter] = word
            
            # Create an ultra-sparse physical spike pattern for the concept
            spike = np.zeros(self.spike_dim, dtype=bool)
            
            # Randomly activate exactly 3 distinct neurons out of 1000 to represent this concept
            active_neurons = np.random.choice(self.spike_dim, 3, replace=False)
            spike[active_neurons] = True
            
            self.word_spikes[word] = spike
            self.vocab_counter += 1
            
        return self.word_spikes[word]

def run_language_experiment():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Native Language Acquisition (Hippocampus)    ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    spike_dim = 1000 # The Hippocampal Matrix has 1,000 neurons
    
    # 1. Initialize the Organs
    encoder = NativeLanguageEncoder(spike_dim=spike_dim)
    hippocampus = CA3AssociativeGraph(num_neurons=spike_dim)
    
    # 2. The Raw Input Environment (Teaching basic concepts)
    sentences = [
        "the cat runs fast",
        "the dog barks loud",
        "the cat sleeps deep",
        "the bird flies high",
        "the dog runs fast",
        "the cheetah runs fast"
    ]
    
    print("[+] Ingesting Vocabulary and Syntactic Concepts...")
    for sentence in sentences:
        words = sentence.split()
        print(f"  -> Experiencing concept: '{sentence}'")
        
        # We simulate the organism's Chemotaxis Saccades moving through the sentence
        for i in range(len(words) - 1):
            concept_A = words[i]
            concept_B = words[i+1]
            
            spike_A = encoder.get_spike(concept_A)
            spike_B = encoder.get_spike(concept_B)
            
            # Combine them into the active working memory (co-firing in the brain)
            co_firing_spikes = spike_A | spike_B
            
            # 3. Hebbian Learning: The Hippocampus physically wires them together
            # "Neurons that fire together, wire together."
            hippocampus.stdp_memorize(co_firing_spikes)
            
    print("\n[+] Hebbian Synapses fused. The biological ontology graph is built.")
    print(f"[+] Total concepts learned natively: {encoder.vocab_counter}")
    
    # 4. Pattern Completion (Testing the Intelligence)
    print("\n[!] Testing Native Pattern Completion...")
    test_cues = ["cat", "dog", "bird", "runs"]
    
    for cue in test_cues:
        print(f"\n  [Sensory Input]: '{cue}'")
        cue_spike = encoder.get_spike(cue)
        
        # The Hippocampus recurrently loops to pull up the associated concepts.
        # It snaps to the discrete energy minimum (the learned habit/concept).
        completed_pattern = hippocampus.pattern_complete(cue_spike, ticks=5)
        
        # Let's decode what other neurons lit up in the matrix!
        active_words = []
        for word, w_spike in encoder.word_spikes.items():
            if word != cue:
                # If at least 2 of the 3 concept neurons fired, the concept was successfully recalled
                if np.sum(completed_pattern & w_spike) >= 2: 
                    active_words.append(word)
                
        print(f"  [Hippocampal Matrix Retrieval]: {active_words}")

if __name__ == "__main__":
    run_language_experiment()
