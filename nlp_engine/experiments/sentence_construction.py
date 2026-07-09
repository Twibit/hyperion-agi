import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.hippocampus import CA3AssociativeGraph
from nlp_engine.experiments.teacher_curriculum import OrthogonalLanguageEncoder

def run_sentence_construction():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Native Sentence Construction                 ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    brain_dim = 2000
    encoder = OrthogonalLanguageEncoder(spike_dim=brain_dim)
    hippocampus = CA3AssociativeGraph(num_neurons=brain_dim)

    # --- TEACHING SEQUENCES ---
    sentences = [
        "the dog barks loudly",
        "the cat sleeps quietly",
        "a big car drives fast",
        "dual_planner learns english grammar natively"
    ]
    
    print("=== INGESTING SEQUENTIAL GRAMMAR ===")
    for sentence in sentences:
        print(f"  -> Wiring spatiotemporal sequence: '{sentence}'")
        words = sentence.split()
        
        # Convert words to orthogonal spikes
        spikes = [encoder.get_spike(w) for w in words]
        
        # Train the organism to form an asymmetric forward-chain
        for _ in range(5): # Reinforce the pathway
            hippocampus.stdp_sequence_memorize(spikes)
            
    print("\n[+] Neural Pathways Constructed. Axonal delays parameterized.")

    # --- NATIVE GENERATION ---
    prompts = ["the", "a", "dual_planner", "dog"]
    
    print("\n=== GENERATING NATIVE SENTENCES ===")
    print("We cue the organism with a starting word, and the neurons chain-react to speak.")
    
    for prompt in prompts:
        print(f"\n👩‍🏫 TEACHER: 'Start a sentence with [{prompt}]'")
        
        if prompt not in encoder.word_to_idx:
            continue
            
        test_spike = encoder.get_spike(prompt)
        
        # The organism natively chain-reacts forward
        generated_spikes = hippocampus.sequence_generate(
            test_spike, sequence_length=6, threshold=1, max_active_neurons=20
        )
        
        # Decode the spike chain back to words
        spoken_sentence = []
        for step_spike in generated_spikes:
            recalled_words = []
            for w, w_spike in encoder.word_spikes.items():
                if np.sum(step_spike & w_spike) >= 1:
                    recalled_words.append(w)
            
            # Since multiple branches might exist (e.g. 'the' branches to 'dog' or 'cat'),
            # it might output multiple superimposed thoughts at once. 
            spoken_sentence.append("[" + "/".join(recalled_words) + "]" if recalled_words else "...")
            
        print(f"  👦 HYPERION (Spatiotemporal Trajectory): {' -> '.join(spoken_sentence)}")

if __name__ == "__main__":
    run_sentence_construction()
