import sys
import os
import re
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.hippocampus import CA3AssociativeGraph
from nlp_engine.experiments.teacher_curriculum import OrthogonalLanguageEncoder

def run_grammar_masterclass():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Advanced Grammar & Punctuation Ingestion     ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    # A concentrated corpus representing the essence of English grammar textbooks
    grammar_text = """
    A sentence expresses a complete thought . 
    The subject is the actor . 
    The verb is the action . 
    An adjective describes a noun . 
    An adverb describes a verb .
    A question mark ? is used at the end of a query . 
    An exclamation mark ! denotes a strong tone of voice or shouting . 
    A comma , indicates a short pause in the sentence .
    A period . denotes the end of a statement .
    who are you ? 
    i am dual_planner . 
    what is your purpose ? 
    to learn english syntax natively !
    stop shouting ! 
    i will pause , and then continue .
    """
    
    # Advanced Tokenizer: Keep words and specific punctuation as unique structural tokens
    tokens = re.findall(r'[a-z]+|[.,!?]', grammar_text.lower())
    
    unique_tokens = list(set(tokens))
    vocab_size = len(unique_tokens)
    brain_dim = vocab_size * 3 
    
    print(f"[+] Grammar Textbook loaded. Total tokens: {len(tokens)} | Unique Structural Tokens: {vocab_size}")
    print(f"[+] Allocating Biological Matrix... Brain Dimension: {brain_dim} Neurons")
    
    encoder = OrthogonalLanguageEncoder(spike_dim=brain_dim)
    for w in unique_tokens:
        encoder.get_spike(w)
        
    hippocampus = CA3AssociativeGraph(num_neurons=brain_dim)
    
    # Read the text using Asymmetric Sequential STDP (Spatiotemporal Wiring)
    print("\n[+] 📖 Hyperion is ingesting the grammar rules...")
    
    # We wire the sequence by sliding a window over the tokens
    for i in range(len(tokens) - 1):
        pre_word = tokens[i]
        post_word = tokens[i+1]
        
        # Asymmetric wiring
        hippocampus.stdp_sequence_memorize([encoder.get_spike(pre_word), encoder.get_spike(post_word)])
            
    print("\n[+] Grammar Rules and Punctuation physically wired into Synapses.")

    # --- NATIVE GENERATION ---
    prompts = ["what", "who", "stop", "i", "an"]
    
    print("\n=== GENERATING STRUCTURAL SENTENCES ===")
    
    for prompt in prompts:
        if prompt not in encoder.word_to_idx:
            continue
            
        print(f"\n👩‍🏫 TEACHER: 'Construct a thought starting with [{prompt}]'")
        
        test_spike = encoder.get_spike(prompt)
        
        # Chain reaction
        generated_spikes = hippocampus.sequence_generate(
            test_spike, sequence_length=7, threshold=1, max_active_neurons=20
        )
        
        spoken_sentence = []
        for step_spike in generated_spikes:
            recalled_words = []
            for w, w_spike in encoder.word_spikes.items():
                if np.sum(step_spike & w_spike) >= 1:
                    recalled_words.append(w)
            
            # If multiple paths exist, format them as [A/B] so we can see the superposition
            if recalled_words:
                if len(recalled_words) == 1:
                    spoken_sentence.append(recalled_words[0])
                else:
                    spoken_sentence.append("[" + "/".join(recalled_words) + "]")
            else:
                break
                
        print(f"  👦 HYPERION: {' '.join(spoken_sentence)}")

if __name__ == "__main__":
    run_grammar_masterclass()
