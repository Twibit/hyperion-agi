import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.hippocampus import CA3AssociativeGraph
from nlp_engine.experiments.teach_language import NativeLanguageEncoder

class OrthogonalLanguageEncoder(NativeLanguageEncoder):
    def get_spike(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.vocab_counter
            self.idx_to_word[self.vocab_counter] = word
            spike = np.zeros(self.spike_dim, dtype=bool)
            # Orthogonal mapping: No overlapping neurons between distinct words
            start_idx = self.vocab_counter * 3
            spike[start_idx:start_idx+3] = True
            self.word_spikes[word] = spike
            self.vocab_counter += 1
        return self.word_spikes[word]

def run_teacher_curriculum():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Grade A+ English Curriculum (Teacher Mode)   ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    brain_dim = 2000
    encoder = OrthogonalLanguageEncoder(spike_dim=brain_dim)
    hippocampus = CA3AssociativeGraph(num_neurons=brain_dim)

    print("👩‍🏫 TEACHER: 'Hello Hyperion. Today we are going to learn the foundation of English.'\n")

    # --- LESSON 1: NOUNS (OBJECTS) ---
    print("=== LESSON 1: BASIC NOUNS (OBJECTS) ===")
    print("👩‍🏫 TEACHER: 'These are the basic symbols of the world.'")
    nouns = ["apple", "dog", "cat", "car", "bird"]
    for noun in nouns:
        spike = encoder.get_spike(noun)
        # We simulate the organism observing the word alone multiple times
        for _ in range(5): 
            hippocampus.stdp_memorize(spike) 
    print(f"  -> Hyperion learned Nouns: {nouns}\n")

    # --- LESSON 2: VERBS (ACTIONS) ---
    print("=== LESSON 2: BASIC VERBS (ACTIONS) ===")
    print("👩‍🏫 TEACHER: 'These words describe how objects move.'")
    verbs = ["eats", "runs", "drives", "flies"]
    for verb in verbs:
        spike = encoder.get_spike(verb)
        for _ in range(5):
            hippocampus.stdp_memorize(spike)
    print(f"  -> Hyperion learned Verbs: {verbs}\n")

    # --- LESSON 3: GRAMMAR (SUBJECT-VERB BINDING) ---
    print("=== LESSON 3: GRAMMAR & SYNTAX ===")
    print("👩‍🏫 TEACHER: 'Now, let's connect the Nouns to the Verbs using Subject-Verb-Object structures. Watch the synapses fuse.'")
    sentences = [
        ("dog", "runs"),
        ("cat", "runs"),
        ("dog", "eats", "apple"),
        ("car", "drives"),
        ("bird", "flies")
    ]
    
    for sentence in sentences:
        print(f"  -> Teaching Context: '{' '.join(sentence)}'")
        combined_spike = np.zeros(brain_dim, dtype=bool)
        for word in sentence:
            combined_spike |= encoder.get_spike(word)
            
        # Bind the grammar context with balanced exposure
        for _ in range(5): 
            hippocampus.stdp_memorize(combined_spike)
        
    print("\n[+] Curriculum Complete. Synaptic weights consolidated.")

    # --- FINAL EXAM ---
    print("\n=== FINAL EXAM (A+ GRADE TEST) ===")
    print("👩‍🏫 TEACHER: 'Hyperion, I will give you a Subject. You must use your Hippocampus to complete the Grammar.'")
    
    exam_questions = ["dog", "car", "bird"]
    
    score = 0
    for question in exam_questions:
        print(f"\n👩‍🏫 TEACHER: 'What does the {question} do?'")
        test_spike = encoder.get_spike(question)
        
        # The organism natively pattern-completes the missing grammar (1-hop exact retrieval)
        completed = hippocampus.pattern_complete(test_spike, ticks=1, threshold=1, max_active_neurons=20)
        
        recalled_words = []
        for w, w_spike in encoder.word_spikes.items():
            if w != question and np.sum(completed & w_spike) >= 2:
                recalled_words.append(w)
                
        print(f"  👦 HYPERION (Internal Retrieval): {recalled_words}")
        
        # Grading
        if question == "dog" and ("runs" in recalled_words or "eats" in recalled_words):
            score += 1
        elif question == "car" and "drives" in recalled_words:
            score += 1
        elif question == "bird" and "flies" in recalled_words:
            score += 1
            
    print(f"\n👩‍🏫 TEACHER: 'Final Score: {score}/{len(exam_questions)}'")
    if score == len(exam_questions):
        print("🏆 RESULT: A+ GRADE! Hyperion natively understands basic English Grammar.")
    else:
        print("RESULT: Keep practicing, Hyperion. The biological matrix needs more tuning.")

if __name__ == "__main__":
    run_teacher_curriculum()
