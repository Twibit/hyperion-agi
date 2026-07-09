import sys
import os
import time
import re
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from nlp_engine.layers.sparse_hippocampus import SparseHippocampus
from nlp_engine.experiments.teacher_curriculum import OrthogonalLanguageEncoder
from nlp_engine.experiments.capitalization_loop import fetch_random_wikipedia

def generate_essay(sparse_brain, encoder, prompt_word="The", length=150):
    print(f"\n[Essay Generation] Prompt: '{prompt_word}'")
    
    if prompt_word not in encoder.word_to_idx:
        encoder.get_spike(prompt_word)
        
    test_spike = encoder.get_spike(prompt_word)
    # Using threshold=0 guarantees spontaneous firing if the brain loses train of thought
    generated_spikes = sparse_brain.sequence_generate(test_spike, sequence_length=length, threshold=0, max_active=3, temperature=0.5)
    
    spoken_sentence = []
    
    for step_spike in generated_spikes:
        active_neurons = np.nonzero(step_spike)[0]
        recalled_words = set()
        for n in active_neurons:
            word_idx = int(n) // 3
            if word_idx in encoder.idx_to_word:
                recalled_words.add(encoder.idx_to_word[word_idx])
        
        if recalled_words:
            # Try to pick a word that is not an immediate stutter
            word = list(recalled_words)[0]
            if len(spoken_sentence) > 0 and word == spoken_sentence[-1]:
                alternative = [w for w in recalled_words if w != spoken_sentence[-1]]
                if alternative:
                    word = alternative[0]
            spoken_sentence.append(word)
        else:
            break
            
    # Format the paragraph by removing spaces before punctuation
    text_output = ' '.join(spoken_sentence)
    text_output = re.sub(r' ([.,!?;:\'"])', r'\1', text_output)
    
    # Capitalize the first letter
    if len(text_output) > 0:
        text_output = text_output[0].upper() + text_output[1:]
        
    print(f"\n{text_output}\n")
    print(f"--- Word Count: {len(spoken_sentence)} ---\n")

def run_llm_loop():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Continuous LLM Generation Training Loop      ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    brain_dim = 1500000 
    encoder = OrthogonalLanguageEncoder(spike_dim=brain_dim)
    sparse_brain = SparseHippocampus(num_neurons=brain_dim)
    
    total_words = 0
    
    # PRE-TRAINING: Give the organism a structural foundation instantly
    print("[+] Pre-Training Foundation: Injecting 100,000 words of Shakespearean grammar...")
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Hyperion-Teacher/4.0'})
        with urllib.request.urlopen(req) as response:
            text = response.read().decode('utf-8')
            tokens = re.findall(r'[A-Za-z]+|[.,!?;:\'"]|\n', text)
            limit = min(100000, len(tokens) - 1)
            unique_tokens = set(tokens[:limit])
            for w in unique_tokens:
                encoder.get_spike(w)
            # Batched sequence memorize for Dendritic Trees
            for i in range(limit):
                pre_word = tokens[i]
                post_word = tokens[i+1]
                # Because the new logic takes a continuous sequence, we should pass chunks!
                # Wait, stdp_sequence_memorize expects a list of spikes
                pass
                
            # Actually, let's just pass the whole thing in chunks of 100
            chunk_size = 100
            for i in range(0, limit, chunk_size):
                chunk_tokens = tokens[i:min(i+chunk_size, limit)]
                spikes = [encoder.get_spike(w) for w in chunk_tokens]
                sparse_brain.stdp_sequence_memorize(spikes)
                
            total_words += limit
            print("[+] Foundation Injected! Organism has acquired structural English syntax.")
    except Exception as e:
        print(f"Failed pre-training: {e}")
    
    last_eval_time = time.time()
    EVAL_INTERVAL = 10  # evaluate every 10 seconds to show the user fast output!
    
    print("\nStarted Continuous Internet Training. Press Ctrl+C to stop.\n")
    
    while True:
        current_time = time.time()
        if current_time - last_eval_time >= EVAL_INTERVAL:
            print("\n=== ⏱️ CHECKPOINT: ESSAY GENERATION ===")
            generate_essay(sparse_brain, encoder, prompt_word="The", length=150)
            last_eval_time = current_time

        title, text = fetch_random_wikipedia()
        if not text:
            time.sleep(1)
            continue
            
        # CASE-SENSITIVE TOKENIZER (Includes Punctuation)
        tokens = re.findall(r'[A-Za-z0-9]+|[.,!?;:\'"]', text)
        if len(tokens) < 10:
            continue
            
        total_words += len(tokens)
        
        unique_tokens = set(tokens)
        for w in unique_tokens:
            encoder.get_spike(w)
            
        spikes = [encoder.get_spike(w) for w in tokens]
        sparse_brain.stdp_sequence_memorize(spikes)
            
        print(f"  [+] Ingested: '{title}' ({len(tokens)} tokens) | Total Words Ingested: {total_words}")

if __name__ == "__main__":
    run_llm_loop()
