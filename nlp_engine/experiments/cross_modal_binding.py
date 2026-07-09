import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from nlp_engine.layers.hippocampus import CA3AssociativeGraph
from teach_language import NativeLanguageEncoder

class RetinaSensor:
    """
    Biological Retina. Converts a 2D image array into a 1D sparse spike train 
    using simulated ganglion cells (edge detection/thresholding).
    """
    def __init__(self, spike_dim):
        self.spike_dim = spike_dim
        # A fixed random projection to simulate the optic nerve mapping pixels to the visual cortex
        np.random.seed(101)
        self.optic_nerve_mapping = np.random.randint(0, 10000, size=self.spike_dim)
        
    def image_to_spikes(self, mock_image_pixels):
        """
        Takes raw pixels and fires specific neurons based on pixel intensity.
        """
        flat_pixels = mock_image_pixels.flatten()
        visual_spikes = np.zeros(self.spike_dim, dtype=bool)
        
        # If the pixel value matches the optic nerve threshold, fire the neuron
        for i in range(self.spike_dim):
            mapped_pixel_idx = self.optic_nerve_mapping[i] % len(flat_pixels)
            if flat_pixels[mapped_pixel_idx] > 200: # High intensity (bright pixel / visual edge)
                visual_spikes[i] = True
                
        # Enforce extreme sparsity (only the top 5 visual features fire)
        active_indices = np.where(visual_spikes)[0]
        if len(active_indices) > 5:
            np.random.shuffle(active_indices)
            visual_spikes[active_indices[5:]] = False
            
        return visual_spikes

def run_multimodal_experiment():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Cross-Modal Binding (Vision + Language)      ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    # The brain has 2000 neurons (1000 for language, 1000 for vision)
    brain_dim = 2000 
    
    # Initialize the Organs
    language_encoder = NativeLanguageEncoder(spike_dim=1000)
    retina = RetinaSensor(spike_dim=1000)
    
    # The Hippocampus binds both modalities together in one giant matrix
    hippocampus = CA3AssociativeGraph(num_neurons=brain_dim)
    
    print("[+] Simulating Sensory Input from the Internet...")
    
    # 1. The Organism "Hears" the word "cat"
    word_spikes = language_encoder.get_spike("cat")
    
    # 2. The Organism "Sees" an image of a cat
    # Mocking a 100x100 image array representing the cat's visual features
    np.random.seed(99)
    image_of_cat = np.random.randint(0, 255, size=(100, 100))
    visual_spikes = retina.image_to_spikes(image_of_cat)
    
    # 3. Multimodal Binding
    # Concatenate the language spikes (Neurons 0-999) and visual spikes (Neurons 1000-1999)
    combined_brain_state = np.concatenate([word_spikes, visual_spikes])
    
    print("  -> The word 'cat' and the visual image of a cat are co-firing in the brain.")
    
    # Hebbian Learning: The Hippocampus physically wires the visual spikes to the audio spikes
    hippocampus.stdp_memorize(combined_brain_state)
    
    print("\n[+] Cross-Modal Synapses fused. The concept of 'cat' is now grounded in reality.")
    
    # ---------------------------------------------------------
    # 4. Testing Intelligence: Show it an image without the word
    # ---------------------------------------------------------
    print("\n[!] Testing Visual Recognition (Show Image, Mute Audio)...")
    
    # We show it the same image pixels, but ZERO language spikes (silence)
    test_brain_state = np.concatenate([np.zeros(1000, dtype=bool), visual_spikes])
    
    # The Hippocampus pattern-completes the missing information
    # The visual spikes will trigger the recurrent loops, which will pull up the audio spikes
    completed_pattern = hippocampus.pattern_complete(test_brain_state, ticks=5, threshold=1, max_active_neurons=10)
    
    # Split the completed pattern back out to see if it hallucinated the word "cat"!
    recalled_language_spikes = completed_pattern[:1000]
    
    recalled_word = "Unknown (Silent)"
    for word, w_spike in language_encoder.word_spikes.items():
        if np.sum(recalled_language_spikes & w_spike) >= 2:
            recalled_word = word
            
    print(f"  [Sensory Input]: Raw Pixels (Image of a Cat)")
    print(f"  [Hippocampal Retrieval (Internal Voice)]: '{recalled_word}'")

if __name__ == "__main__":
    run_multimodal_experiment()
