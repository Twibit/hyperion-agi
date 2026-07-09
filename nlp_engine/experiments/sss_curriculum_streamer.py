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
        url = "https://en.wikipedia.org/w/api.php?format=json&action=query&generator=random&grnnamespace=0&prop=extracts&exchars=1500&explaintext"
        req = urllib.request.Request(url, headers={'User-Agent': 'Hyperion-Teacher/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data['query']['pages']
            for page_id in pages:
                return pages[page_id]['title'], pages[page_id]['extract']
    except Exception:
        return None, None

def run_sss_tier_training():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: SSS+ Tier Teacher/Proctor Training Pipeline  ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    print("👩‍🏫 TEACHER: 'I am the Proctor. Activating the Sparse Tensor memory matrix.'")
    print("👩‍🏫 TEACHER: 'We will now bypass all Dense RAM limits and stream raw internet data continuously.'")
    
    # 3 Million Neurons (Supports ~1,000,000 words orthogonally)
    brain_dim = 3000000 
    encoder = OrthogonalLanguageEncoder(spike_dim=brain_dim)
    sparse_brain = SparseHippocampus(num_neurons=brain_dim)
    
    print(f"\n[+] Brain instantiated at {brain_dim} Neurons using O(K) sparse memory allocation.")
    print("👩‍🏫 TEACHER: 'Let the SSS+ Tier Training begin. Scanning Wikipedia...'\n")
    
    target_articles = 5
    articles_read = 0
    total_words = 0
    prompt_word = "the"
    
    while articles_read < target_articles:
        title, text = fetch_random_wikipedia()
        if not text:
            continue
            
        # Advanced Tokenizer (preserves punctuation structure)
        tokens = re.findall(r'[a-z]+|[.,!?]', text.lower())
        if len(tokens) < 10:
            continue
            
        total_words += len(tokens)
        
        # Pre-allocate orthogonal neurons safely
        unique_tokens = set(tokens)
        for w in unique_tokens:
            encoder.get_spike(w)
            
        # Ingest the sequential stream into the Sparse Hippocampus
        for i in range(len(tokens) - 1):
            pre_word = tokens[i]
            post_word = tokens[i+1]
            sparse_brain.stdp_sequence_memorize([encoder.get_spike(pre_word), encoder.get_spike(post_word)])
            
        articles_read += 1
        print(f"  [+] Ingested Wikipedia Article {articles_read}/{target_articles}: '{title}' ({len(tokens)} tokens)")
        prompt_word = tokens[min(5, len(tokens)-1)] # Grab a word from the article to prompt it with later
        
    print(f"\n[+] Training Session Complete. {total_words} tokens physically wired into the sparse matrix.")
    print("👩‍🏫 TEACHER: 'Time for the final exam. SSS+ Tier Fluency Test on the knowledge just ingested.'")
    
    print(f"\n👩‍🏫 PROCTOR: 'Construct a sentence starting with [{prompt_word}]'")
    
    test_spike = encoder.get_spike(prompt_word)
    generated_spikes = sparse_brain.sequence_generate(test_spike, sequence_length=15, threshold=1, max_active=10)
    
    spoken_sentence = []
    for step_spike in generated_spikes:
        recalled_words = []
        for w, w_spike in encoder.word_spikes.items():
            if np.sum(step_spike & w_spike) >= 1:
                recalled_words.append(w)
        
        if recalled_words:
            # We join multiple predictions with a slash
            spoken_sentence.append("[" + "/".join(recalled_words) + "]" if len(recalled_words) > 1 else recalled_words[0])
        else:
            break
            
    print(f"  👦 HYPERION: {' '.join(spoken_sentence)}")

if __name__ == "__main__":
    run_sss_tier_training()
