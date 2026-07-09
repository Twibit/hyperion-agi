import sys
import os
import time
import re
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.sparse_hippocampus import SparseHippocampus
from nlp_engine.experiments.capitalization_loop import fetch_random_wikipedia, OrthogonalLanguageEncoder
from nlp_engine.experiments.capitalization_loop import fetch_random_wikipedia

class AcademicCritic:
    def __init__(self):
        self.academic_vocab = {"Furthermore", "Therefore", "Empirical", "Hypothesis", "Significant", "Analysis", "However", "Consequently", "Moreover", "Methodology"}
        
    def evaluate(self, text):
        score = 0
        words = text.split()
        
        # Punish Repetitions Severely!
        for i in range(1, len(words)):
            if words[i].lower() == words[i-1].lower():
                score -= 30  # Massive Cortisol for stuttering
            if i >= 2 and words[i].lower() == words[i-2].lower():
                score -= 10
                
        # Punish Wikipedia Boilerplate
        boilerplate = ["References", "External", "links", "See", "also", "Category", "stub"]
        for b in boilerplate:
            if b in words:
                score -= 20
        
        # Reward Academic Vocabulary
        for w in words:
            clean = re.sub(r'[^A-Za-z]', '', w)
            if clean in self.academic_vocab or clean.title() in self.academic_vocab:
                score += 30
                
        # Punish Word Salad Formatting
        if ",," in text or ";," in text or ".;" in text or " :" in text:
            score -= 30
            
        # Reward sentence structure
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 0]
        for s in sentences:
            if s[0].isupper():
                score += 10
            else:
                score -= 15
            if len(s.split()) > 5:
                score += 10 # Reward long, flowing sentences
            elif len(s.split()) < 3:
                score -= 10 # Punish fragmented sentences
                
        return score

def run_academic_training():
    vocab_size = 200000
    brain_dim = 1500000
    
    encoder = OrthogonalLanguageEncoder(spike_dim=brain_dim)
    sparse_brain = SparseHippocampus(num_neurons=brain_dim)
    critic = AcademicCritic()
    
    print("🎓 Academic Teacher Mode: Activated.")
    print("Goal: Force the organism to construct high-level academic essays using Pain/Trauma (Cortisol) and Reward (Dopamine).\n")
    
    total_words = 0
    
    for _ in range(5):
        title, text = fetch_random_wikipedia()
        if not text:
            continue
        # Sanitization Filter
        text = re.sub(r'==\s*(References|External links|See also|Bibliography|Further reading)\s*==.*', '', text, flags=re.IGNORECASE|re.DOTALL)
        text = re.sub(r'==.*?==', '', text)
        tokens = re.findall(r'[A-Za-z]+|[.,!?;:\'"]|\n', text)
        for w in set(tokens):
            encoder.get_spike(w)
        sparse_brain.stdp_sequence_memorize([encoder.get_spike(w) for w in tokens])
        total_words += len(tokens)
        print(f"[*] Ingested {len(tokens)} words from '{title}'")
        
    print("\n[+] Initial Knowledge Base Built. Beginning Academic Trauma/Reward Loop.\n")
    
    epoch = 1
    current_temperature = 0.5
    
    while True:
        # Fetch new knowledge
        title, text = fetch_random_wikipedia()
        if not text:
            time.sleep(1)
            continue
            
        # Sanitization Filter
        text = re.sub(r'==\s*(References|External links|See also|Bibliography|Further reading)\s*==.*', '', text, flags=re.IGNORECASE|re.DOTALL)
        text = re.sub(r'==.*?==', '', text)
        tokens = re.findall(r'[A-Za-z]+|[.,!?;:\'"]|\n', text)
        for w in set(tokens):
            encoder.get_spike(w)
        sparse_brain.stdp_sequence_memorize([encoder.get_spike(w) for w in tokens])
        
        print(f"\n--- Epoch {epoch} --- [Temp: {current_temperature:.2f}]")
        prompt = "The"
        init_spike = encoder.get_spike(prompt)
        
        # Generation
        spikes, pathway = sparse_brain.sequence_generate(init_spike, sequence_length=50, threshold=0.0, temperature=current_temperature)
        
        essay_words = [prompt]
        for spike in spikes:
            active = np.nonzero(spike)[0]
            recalled_words = set()
            for n in active:
                word_idx = n // 3
                if word_idx in encoder.idx_to_word:
                    recalled_words.add(encoder.idx_to_word[word_idx])
            
            if recalled_words:
                word = list(recalled_words)[0]
                essay_words.append(word)
        essay = " ".join(essay_words)
        
        # Grading
        score = critic.evaluate(essay)
        print(f"Text:\n{essay}")
        print(f"\nGrade Score: {score}")
        
        if score >= 10:
            print("=> 🟢 TEACHER REWARD (DOPAMINE): Synapses Strengthened!")
            sparse_brain.apply_dopamine(5.0, pathway)
            current_temperature = max(0.1, current_temperature - 0.1) # Cool down to lock in grammar
        elif score < 0:
            print("=> 🔴 TEACHER TRAUMA (CORTISOL): Synapses Depotentiated (Pruned)!")
            sparse_brain.apply_cortisol(2.0, pathway)
            current_temperature = min(1.0, current_temperature + 0.2) # Heat up to break rut
        else:
            print("=> ⚪ NEUTRAL: No neuromodulation applied.")
            
        epoch += 1
        time.sleep(2)

if __name__ == "__main__":
    run_academic_training()
