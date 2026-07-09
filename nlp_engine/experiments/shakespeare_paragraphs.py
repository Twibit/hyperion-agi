import sys
import os
import urllib.request
import re
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.sparse_hippocampus import SparseHippocampus
from nlp_engine.experiments.teacher_curriculum import OrthogonalLanguageEncoder

def run_shakespeare_paragraphs():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Shakespearean Paragraph Generation           ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    brain_dim = 1500000 
    encoder = OrthogonalLanguageEncoder(spike_dim=brain_dim)
    sparse_brain = SparseHippocampus(num_neurons=brain_dim)
    
    # Download tiny shakespeare
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Hyperion-Teacher/4.0'})
        with urllib.request.urlopen(req) as response:
            text = response.read().decode('utf-8')
    except Exception as e:
        print(f"Failed to download Shakespeare dataset: {e}")
        return

    # Case-Sensitive Tokenizer including newlines
    print("[+] Tokenizing Tiny Shakespeare (Case-Sensitive)...")
    tokens = re.findall(r'[A-Za-z]+|[.,!?;:\'"]|\n', text)
    print(f"[+] Total Tokens: {len(tokens)}")
    
    unique_tokens = set(tokens)
    for w in unique_tokens:
        encoder.get_spike(w)
        
    print(f"[+] Unique Vocabulary: {len(unique_tokens)}")
    
    print("[+] 📖 Physically wiring the organism's Hippocampus...")
    # Wire 100k tokens for speed
    limit = min(100000, len(tokens) - 1)
    
    for i in range(limit):
        pre_word = tokens[i]
        post_word = tokens[i+1]
        sparse_brain.stdp_sequence_memorize([encoder.get_spike(pre_word), encoder.get_spike(post_word)])
        if i % 100000 == 0 and i > 0:
            print(f"  -> Wired {i} tokens...")
            
    print("[+] Wiring Complete!")
    
    # GENERATION
    prompts = ["To", "Machine", "Love", "Hate"]
    
    print("\n=== GENERATING PARAGRAPHS NATIVELY ===")
    
    for prompt in prompts:
        print(f"\n🎭 PROMPT: '{prompt}'")
        if prompt not in encoder.word_to_idx:
            # Organism encounters a totally new word!
            encoder.get_spike(prompt) 
            
        test_spike = encoder.get_spike(prompt)
        
        # max_active=3 with temperature allows the organism to hallucinate creatively
        generated_spikes = sparse_brain.sequence_generate(
            test_spike, sequence_length=50, threshold=1, max_active=3, temperature=15.0
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
                word = list(recalled_words)[0]
                spoken_sentence.append(word)
            else:
                break
                
        # Format the paragraph by joining words and handling newlines properly
        paragraph = ""
        for word in spoken_sentence:
            if word == '\n':
                paragraph += '\n  '
            elif word in ['.', ',', '!', '?', ';', ':', "'", '"']:
                paragraph = paragraph.rstrip() + word + ' '
            else:
                paragraph += word + ' '
                
        print(f"  👦 HYPERION: {paragraph}")

if __name__ == "__main__":
    run_shakespeare_paragraphs()
