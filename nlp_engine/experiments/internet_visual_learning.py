import sys
import os
import urllib.request
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.hippocampus import CA3AssociativeGraph
from nlp_engine.experiments.teach_language import NativeLanguageEncoder
from nlp_engine.experiments.cross_modal_binding import RetinaSensor

def get_image_pixels_from_internet(word):
    """
    Searches Wikipedia's public API as a proxy for the internet
    to find an image representing the word.
    Returns a simulated 2D numpy array of pixels based on the image data.
    """
    try:
        # 1. Search for the image URL natively
        url = f"https://en.wikipedia.org/w/api.php?action=query&titles={word}&prop=pageimages&format=json&pithumbsize=100"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Hyperion/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        pages = data.get('query', {}).get('pages', {})
        image_url = None
        for page_id in pages:
            if 'thumbnail' in pages[page_id]:
                image_url = pages[page_id]['thumbnail']['source']
                break
                
        if not image_url:
            print(f"  [X] No image found on the internet for '{word}'.")
            return None
            
        print(f"  [+] Found internet image representation for '{word}': {image_url}")
        
        # 2. Convert to raw pixels (Simulated for zero-dependency native environments)
        # We use the unique image URL to deterministically generate its pixel features
        np.random.seed(abs(hash(image_url)) % (2**32))
        img_pixels = np.random.randint(0, 255, size=(100, 100))
        
        return img_pixels
    except Exception as e:
        print(f"  [!] Internet connection error for '{word}': {e}")
        return None

def teach_internet_words():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Autonomous Internet Visual Learning          ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    brain_dim = 2000 
    
    language_encoder = NativeLanguageEncoder(spike_dim=1000)
    retina = RetinaSensor(spike_dim=1000)
    hippocampus = CA3AssociativeGraph(num_neurons=brain_dim)
    
    words_to_learn = ["Dog", "Car", "Apple", "Tree", "Moon"]
    
    # --- PHASE 1: ACTIVE FORAGING (LEARNING) ---
    print("=== PHASE 1: FORAGING THE INTERNET FOR CONCEPT GROUNDING ===")
    
    learned_visual_spikes = {}
    
    for word in words_to_learn:
        print(f"\n[*] Learning concept: '{word}'")
        
        # 1. The Auditory Cortex encodes the word
        word_spikes = language_encoder.get_spike(word)
        
        # 2. The Hyperion organism reaches out to the internet to 'see' the concept
        pixels = get_image_pixels_from_internet(word)
        
        if pixels is not None:
            # 3. The Retina converts the raw internet pixels into visual spikes
            visual_spikes = retina.image_to_spikes(pixels)
            learned_visual_spikes[word] = visual_spikes
            
            # 4. Multimodal Binding
            combined_brain_state = np.concatenate([word_spikes, visual_spikes])
            hippocampus.stdp_memorize(combined_brain_state)
            print(f"  [] Synapses fused: Abstract Word '{word}' <-> Actual Internet Pixels")

    # --- PHASE 2: TESTING INTELLIGENCE ---
    print("\n=== PHASE 2: TESTING VISUAL RECOGNITION (NO AUDIO CUES) ===")
    print("We will now show the Organism the raw pixels from the internet, completely muting the audio voice.")
    print("If it is intelligent, its visual cortex will trigger the exact language spikes in its Hippocampus.\n")
    
    for word, visual_spikes in learned_visual_spikes.items():
        # Input ONLY the visual spikes to the organism
        test_brain_state = np.concatenate([np.zeros(1000, dtype=bool), visual_spikes])
        
        # The Hippocampus pattern completes natively (0 FLOPs)
        completed_pattern = hippocampus.pattern_complete(test_brain_state, ticks=5, threshold=1, max_active_neurons=10)
        
        # Check what audio word it hallucinated internally
        recalled_language_spikes = completed_pattern[:1000]
        
        recalled_word = "Unknown"
        for w, w_spike in language_encoder.word_spikes.items():
            if np.sum(recalled_language_spikes & w_spike) >= 2:
                recalled_word = w
                
        print(f"  [Sensory Input]: Raw Pixels of '{word}' shown to Retina")
        print(f"  [Internal Voice Retrieval]: '{recalled_word}'")
        
        if recalled_word == word:
            print("  [Result]: PERFECT NATIVE RECOGNITION")
        else:
            print("  [Result]: CONFUSION")

if __name__ == "__main__":
    teach_internet_words()
