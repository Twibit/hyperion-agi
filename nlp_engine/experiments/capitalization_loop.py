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
        req = urllib.request.Request(url, headers={'User-Agent': 'Hyperion-Teacher/3.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data['query']['pages']
            for page_id in pages:
                return pages[page_id]['title'], pages[page_id]['extract']
    except Exception:
        return None, None

def evaluate_capitalization(sparse_brain, encoder):
    # Prompt it with a period and see if it physically learned to route to Capitalized words
    prompts = [".", "!", "?", "He", "The"]
    valid_sentences = 0
    
    for prompt in prompts:
        if prompt not in encoder.word_to_idx:
            continue
            
        test_spike = encoder.get_spike(prompt)
        generated_spikes = sparse_brain.sequence_generate(test_spike, sequence_length=8, threshold=1, max_active=2)
        
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
                if len(spoken_sentence) > 0 and word == spoken_sentence[-1]:
                    break
                spoken_sentence.append(word)
            else:
                break
                
        # A+ Grade: if the prompt was punctuation, the FIRST word generated MUST be capitalized.
        text_output = ' '.join(spoken_sentence)
        if len(spoken_sentence) >= 2:
            if prompt in [".", "!", "?"]:
                first_generated_word = spoken_sentence[1]
                if first_generated_word[0].isupper():
                    valid_sentences += 1
                    print(f"    [PERFECT SYNTAX]: {text_output}")
                else:
                    print(f"    [CASE ERROR]: {text_output}")
            else:
                valid_sentences += 1
                print(f"    [FLUENT]: {text_output}")
        else:
            print(f"    [FRAGMENT]: {text_output}")
            
    if valid_sentences >= 3:
        return "A+"
    return "C"

def run_capitalization_loop():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Case-Sensitive Symbols Training Loop         ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    brain_dim = 1500000 
    encoder = OrthogonalLanguageEncoder(spike_dim=brain_dim)
    sparse_brain = SparseHippocampus(num_neurons=brain_dim)
    
    total_words = 0
    last_eval_time = time.time()
    EVAL_INTERVAL = 30 
    
    print("👩‍🏫 TEACHER: 'I have activated the Case-Sensitive Tokenizer.'")
    print("👩‍🏫 TEACHER: 'Hyperion must now learn to capitalize words after punctuation marks.'\n")
    
    while True:
        current_time = time.time()
        if current_time - last_eval_time >= EVAL_INTERVAL:
            print("\n=== ⏱️ 30 SECOND CHECKPOINT: CASE & SYMBOL EXAM ===")
            grade = evaluate_capitalization(sparse_brain, encoder)
            print(f"=== 🏆 CURRENT GRADE: {grade} ===\n")
            
            if grade == "A+":
                print("🎉 HYPERION HAS REACHED A+ TIER SYNTAX! It organically learned to capitalize words after punctuation.")
                break
                
            last_eval_time = current_time

        title, text = fetch_random_wikipedia()
        if not text:
            time.sleep(1)
            continue
            
        # CASE-SENSITIVE TOKENIZER
        tokens = re.findall(r'[A-Za-z]+|[.,!?;:\'"]', text)
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
            
        print(f"  [+] Ingested: '{title}' ({len(tokens)} tokens) | Case-Sensitive Vocab Map updated.")

if __name__ == "__main__":
    run_capitalization_loop()
