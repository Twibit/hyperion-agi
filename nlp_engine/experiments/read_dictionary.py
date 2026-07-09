import sys
import os
import urllib.request
import json
import re
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.hippocampus import CA3AssociativeGraph
from nlp_engine.experiments.teacher_curriculum import OrthogonalLanguageEncoder

def run_dictionary_reading():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Ingesting the English Dictionary natively    ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    print("[+] Loading local English Dictionary JSON...")
    try:
        with open("dictionary.json", "r", encoding="utf-8") as f:
            dictionary_data = json.load(f)
    except Exception as e:
        print(f"Failed to load dictionary: {e}")
        return
        
    # To fit inside continuous dense RAM without converting to a Sparse Matrix array, 
    # we take the first 2,000 dictionary definitions.
    subset_keys = list(dictionary_data.keys())[:2000]
    
    all_text = ""
    for k in subset_keys:
        all_text += k + " " + dictionary_data[k] + " "
        
    words = re.findall(r'\b[a-z]+\b', all_text.lower())
    stopwords = {"the", "and", "to", "of", "a", "it", "is", "in", "or", "an", "as", "by", "for", "with", "from", "that"}
    filtered_words = [w for w in words if w not in stopwords]
    
    unique_words = list(set(filtered_words))
    vocab_size = len(unique_words)
    brain_dim = vocab_size * 3 # 3 neurons per word
    
    print(f"[+] Loaded {len(subset_keys)} Dictionary Definitions.")
    print(f"[+] Unique Vocabulary in definitions: {vocab_size} words.")
    print(f"[+] Allocating Dense Biological Matrix... Brain Dimension: {brain_dim} Neurons")
    print(f"[!] (Note: In a dense numpy matrix, {brain_dim}x{brain_dim} bytes takes ~{brain_dim**2 / 1e6:.1f} MB of RAM)")
    
    encoder = OrthogonalLanguageEncoder(spike_dim=brain_dim)
    # Pre-populate all unique words so they have dedicated orthogonal indices
    for w in unique_words:
        encoder.get_spike(w)
    
    # We use int8 to aggressively save RAM
    hippocampus = CA3AssociativeGraph(num_neurons=brain_dim)
    hippocampus.recurrent_weights = np.zeros((brain_dim, brain_dim), dtype=np.int8) 
    
    print("\n[+] 📖 Hyperion is physically wiring the dictionary definitions into its brain...")
    start_time = time.time()
    
    for i, word_key in enumerate(subset_keys):
        definition = dictionary_data[word_key]
        def_words = [w for w in re.findall(r'\b[a-z]+\b', definition.lower()) if w not in stopwords]
        
        target_word = word_key.lower()
        if target_word not in encoder.word_to_idx:
            continue # Safe fallback
            
        target_spike = encoder.get_spike(target_word)
        
        for dw in def_words:
            if dw not in encoder.word_to_idx:
                continue
            dw_spike = encoder.get_spike(dw)
            combined_spike = target_spike | dw_spike
            
            # STDP binding: The definition literally wires into the word
            spike_idx = np.nonzero(combined_spike)[0]
            for idx in spike_idx:
                # Add safely avoiding int8 overflow
                hippocampus.recurrent_weights[idx, spike_idx] = np.clip(
                    hippocampus.recurrent_weights[idx, spike_idx] + 1, -128, 127
                )
                hippocampus.recurrent_weights[idx, idx] = 0
                
        if i > 0 and i % 500 == 0:
            print(f"  -> Memorized {i} definitions... ({time.time() - start_time:.2f}s)")
            
    print(f"[+] Dictionary Ingestion complete in {time.time() - start_time:.2f} seconds!")
    
    # TEST THE KNOWLEDGE
    prompts = [subset_keys[0].lower(), subset_keys[50].lower(), subset_keys[500].lower()]
    
    print("\n=== NATIVE DEFINITION RETRIEVAL ===")
    for prompt in prompts:
        if prompt not in encoder.word_spikes:
            continue
            
        print(f"\n👩‍🏫 TEACHER: 'Hyperion, what is the definition of [{prompt}]?'")
        test_spike = encoder.get_spike(prompt)
        
        completed = hippocampus.pattern_complete(test_spike, ticks=1, threshold=1, max_active_neurons=20)
        
        recalled_words = []
        for w, w_spike in encoder.word_spikes.items():
            if w != prompt and np.sum(completed & w_spike) >= 1:
                recalled_words.append(w)
                
        print(f"  👦 HYPERION (Hippocampal Retrieval): {recalled_words[:10]}")

if __name__ == "__main__":
    run_dictionary_reading()
