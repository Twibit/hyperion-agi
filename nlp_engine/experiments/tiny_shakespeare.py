import sys
import os
import urllib.request
import re
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.sparse_hippocampus import SparseHippocampus
from nlp_engine.experiments.teacher_curriculum import OrthogonalLanguageEncoder

def run_tiny_shakespeare():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: The Tiny Shakespeare SNN Benchmark           ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    print("[+] Downloading Tiny Shakespeare dataset (1MB)...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Hyperion/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            raw_text = response.read().decode('utf-8')
    except Exception as e:
        print(f"Failed: {e}")
        return
        
    print(f"[+] Download complete. Dataset size: {len(raw_text)} characters.")
    
    # Advanced Tokenizer
    print("[+] Tokenizing...")
    tokens = re.findall(r'[A-Za-z]+|[.,!?;:]', raw_text.lower())
    
    unique_tokens = list(set(tokens))
    vocab_size = len(unique_tokens)
    brain_dim = vocab_size * 3 
    
    print(f"[+] Total Words/Tokens: {len(tokens)}")
    print(f"[+] Unique Vocabulary: {vocab_size}")
    print(f"[+] Allocating Sparse Biological Matrix: {brain_dim} Neurons")
    
    encoder = OrthogonalLanguageEncoder(spike_dim=brain_dim)
    for w in unique_tokens:
        encoder.get_spike(w)
        
    sparse_brain = SparseHippocampus(num_neurons=brain_dim)
    
    print("\n[+] 📖 Hyperion is physically wiring Tiny Shakespeare via STDP...")
    start_time = time.time()
    
    for i in range(len(tokens) - 1):
        pre = tokens[i]
        post = tokens[i+1]
        sparse_brain.stdp_sequence_memorize([encoder.get_spike(pre), encoder.get_spike(post)])
        
        if i % 50000 == 0 and i > 0:
            print(f"  -> Ingested {i} tokens... ({time.time() - start_time:.2f}s)")
            
    print(f"[+] Ingestion complete in {time.time() - start_time:.2f} seconds!")
    
    # --- NATIVE GENERATION ---
    prompts = ["romeo", "juliet", "thou", "to", "o"]
    
    print("\n=== GENERATING SHAKESPEAREAN TEXT NATIVELY ===")
    
    for prompt in prompts:
        if prompt not in encoder.word_to_idx:
            continue
            
        print(f"\n🎭 PROMPT: '{prompt}'")
        
        test_spike = encoder.get_spike(prompt)
        
        # Max active set to 1 to just pick the strongest pathway without superposition noise
        generated_spikes = sparse_brain.sequence_generate(
            test_spike, sequence_length=20, threshold=1, max_active=1
        )
        
        spoken_sentence = []
        for step_spike in generated_spikes:
            active_neurons = np.nonzero(step_spike)[0]
            recalled_words = set()
            for n in active_neurons:
                word_idx = n // 3
                if word_idx in encoder.idx_to_word:
                    recalled_words.add(encoder.idx_to_word[word_idx])
            
            if recalled_words:
                spoken_sentence.append("/".join(recalled_words))
            else:
                break
                
        print(f"  👦 HYPERION: {' '.join(spoken_sentence)}")

if __name__ == "__main__":
    run_tiny_shakespeare()
