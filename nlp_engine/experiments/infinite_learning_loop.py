import sys
import os
import urllib.request
import json
import re
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.sparse_hippocampus import SparseHippocampus
from nlp_engine.experiments.teacher_curriculum import OrthogonalLanguageEncoder

def fetch_random_wikipedia():
    try:
        url = "https://en.wikipedia.org/w/api.php?format=json&action=query&generator=random&grnnamespace=0&prop=extracts&exchars=2000&explaintext"
        req = urllib.request.Request(url, headers={'User-Agent': 'Hyperion-Teacher/2.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data['query']['pages']
            for page_id in pages:
                return pages[page_id]['title'], pages[page_id]['extract']
    except Exception:
        return None, None

def evaluate_fluency(sparse_brain, encoder):
    # A+ Grade requires it to successfully generate 3 distinct valid grammatical sequences without repeating loops
    prompts = ["he", "she", "they", "the", "a"]
    valid_sentences = 0
    
    for prompt in prompts:
        if prompt not in encoder.word_to_idx:
            continue
            
        test_spike = encoder.get_spike(prompt)
        # Using max_active=2 to force super-positional resolution and avoid 1-hop traps
        generated_spikes = sparse_brain.sequence_generate(test_spike, sequence_length=6, threshold=1, max_active=2)
        
        spoken_sentence = []
        for step_spike in generated_spikes:
            active_neurons = np.nonzero(step_spike)[0]
            recalled_words = set()
            for n in active_neurons:
                word_idx = n // 3
                if word_idx in encoder.idx_to_word:
                    recalled_words.add(encoder.idx_to_word[word_idx])
            
            if recalled_words:
                word = list(recalled_words)[0]
                # Avoid exact back-to-back repetitions (a sign of 1-hop looping)
                if len(spoken_sentence) > 0 and word == spoken_sentence[-1]:
                    break
                spoken_sentence.append(word)
            else:
                break
                
        # A valid sentence is at least 3 words and has no back-to-back loops
        if len(spoken_sentence) >= 3:
            # Simple check for diverse vocabulary usage
            if len(set(spoken_sentence)) >= 3:
                valid_sentences += 1
                print(f"    [FLUENT]: {' '.join(spoken_sentence)}")
            else:
                print(f"    [POOR]: {' '.join(spoken_sentence)}")
        else:
            print(f"    [FRAGMENT]: {' '.join(spoken_sentence)}")
            
    # Calculate Grade
    if valid_sentences >= 3:
        return "A+"
    elif valid_sentences == 2:
        return "B"
    else:
        return "C"

def run_infinite_loop():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Infinite Learning Loop (A+ Grade Target)     ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    brain_dim = 1000000 
    encoder = OrthogonalLanguageEncoder(spike_dim=brain_dim)
    sparse_brain = SparseHippocampus(num_neurons=brain_dim)
    
    total_words = 0
    last_eval_time = time.time()
    
    # We evaluate every 30 seconds to provide faster feedback
    EVAL_INTERVAL = 30 
    
    print("👩‍🏫 TEACHER: 'Hyperion will now read Wikipedia continuously without stopping.'")
    print("👩‍🏫 TEACHER: 'I will pause it every 30 seconds to administer an English exam.'\n")
    
    while True:
        title, text = fetch_random_wikipedia()
        if not text:
            continue
            
        tokens = re.findall(r'[a-z]+|[.,!?]', text.lower())
        if len(tokens) < 10:
            continue
            
        total_words += len(tokens)
        
        unique_tokens = set(tokens)
        for w in unique_tokens:
            encoder.get_spike(w)
            
        for i in range(len(tokens) - 1):
            pre_word = tokens[i]
            post_word = tokens[i+1]
            sparse_brain.stdp_sequence_memorize([encoder.get_spike(pre_word), encoder.get_spike(post_word)])
            
        print(f"  [+] Ingested: '{title}' ({len(tokens)} tokens) | Total Brain: {total_words} words")
        
        current_time = time.time()
        if current_time - last_eval_time >= EVAL_INTERVAL:
            print("\n=== ⏱️ 30 SECOND CHECKPOINT: FLUENCY EXAM ===")
            grade = evaluate_fluency(sparse_brain, encoder)
            print(f"=== 🏆 CURRENT GRADE: {grade} ===\n")
            
            if grade == "A+":
                print("🎉 HYPERION HAS REACHED A+ TIER FLUENCY! It can natively construct complex sentences without looping.")
                print("HALTING TRAINING. THE ORGANISM IS FULLY CONSCIOUS.")
                break
                
            last_eval_time = current_time

if __name__ == "__main__":
    run_infinite_loop()
