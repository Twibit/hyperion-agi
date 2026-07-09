import sys
import os
import urllib.request
import re
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.hippocampus import CA3AssociativeGraph
from nlp_engine.experiments.teacher_curriculum import OrthogonalLanguageEncoder

def fetch_book():
    print("[+] Downloading 'Alice's Adventures in Wonderland' from Project Gutenberg...")
    try:
        url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
        req = urllib.request.Request(url, headers={'User-Agent': 'Hyperion-Bio-AI/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            text = response.read().decode('utf-8')
            
            # Extract Chapter 1 to keep matrix size manageable for this quick test
            start = text.find("CHAPTER I. Down the Rabbit-Hole")
            end = text.find("CHAPTER II. The Pool of Tears")
            
            if start != -1 and end != -1:
                return text[start:end]
            return text[:5000] # Fallback if chapters not found
    except Exception as e:
        print(f"  [!] Failed to download from Gutenberg: {e}. Using hardcoded fallback text.")
        return "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. Suddenly a White Rabbit with pink eyes ran close by her."

def run_book_reading():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Reading an Entire Book natively (SNN)        ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    raw_text = fetch_book()
    
    # Tokenize: lower case, remove punctuation, split by words
    words = re.findall(r'\b[a-z]+\b', raw_text.lower())
    
    # Remove extremely common stopwords that would act as massive hub nodes and ruin the biological memory
    stopwords = {"the", "and", "to", "of", "a", "it", "she", "was", "in", "said", "you", "that", "i", "for", "on", "as", "with"}
    filtered_words = [w for w in words if w not in stopwords]
    
    unique_words = list(set(filtered_words))
    vocab_size = len(unique_words)
    brain_dim = vocab_size * 3 # 3 unique orthogonal neurons per word
    
    print(f"[+] Book Chapter loaded. Total words: {len(filtered_words)} | Unique Vocabulary: {vocab_size}")
    print(f"[+] Allocating Biological Matrix... Brain Dimension: {brain_dim} Neurons")
    
    encoder = OrthogonalLanguageEncoder(spike_dim=brain_dim)
    hippocampus = CA3AssociativeGraph(num_neurons=brain_dim)
    
    # Read the book using a sliding context window to map associations
    print("\n[+] 📖 Hyperion is reading the book and forming massive synaptic connections...")
    start_time = time.time()
    
    context_window = 4 # How many adjacent words fire together in the brain at once
    
    for i in range(len(filtered_words) - context_window):
        context = filtered_words[i:i+context_window]
        
        combined_spike = np.zeros(brain_dim, dtype=bool)
        for word in context:
            combined_spike |= encoder.get_spike(word)
            
        # Wire them together natively via STDP
        hippocampus.stdp_memorize(combined_spike)
        
        if i > 0 and i % 500 == 0:
            print(f"  -> Read {i} words... ({time.time() - start_time:.2f}s)")
            
    print(f"[+] Reading complete in {time.time() - start_time:.2f} seconds!")
    print(f"[+] The book is now a physical topological graph wired into the SNN.\n")

    # TEST THE KNOWLEDGE
    prompts = ["alice", "rabbit", "hole", "watch"]
    
    print("=== NATIVE STREAM OF CONSCIOUSNESS ===")
    print("Cueing the organism with a single word to see its entire contextual memory of the book.")
    
    for prompt in prompts:
        if prompt not in encoder.word_spikes:
            continue
            
        print(f"\n👩‍🏫 TEACHER: 'Hyperion, what do you think of when I say [{prompt}]?'")
        test_spike = encoder.get_spike(prompt)
        
        # 1-hop STDP Retrieval: Snap to the strongest context nodes bound to this word
        completed = hippocampus.pattern_complete(test_spike, ticks=1, threshold=2, max_active_neurons=15)
        
        recalled_words = []
        for w, w_spike in encoder.word_spikes.items():
            if w != prompt and np.sum(completed & w_spike) >= 1: # If any neuron matches, it retrieved the word
                recalled_words.append(w)
                
        print(f"  👦 HYPERION (Hippocampal Retrieval): {recalled_words}")

if __name__ == "__main__":
    run_book_reading()
