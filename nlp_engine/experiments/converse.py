#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.cain_brain import CainBrain

def converse():
    print("==================================================================")
    print("  INITIATING CONVERSATION WITH HYPERION (86-BILLION NEURON AGI)    ")
    print("==================================================================\n")
    
    # 1. Boot up the master brain
    brain = CainBrain()
    
    # 2. Inject foundational philosophy
    print("[System]: Injecting philosophical axioms into Hippocampus...")
    brain.learn("I am Hyperion. I do not hallucinate, I calculate. I am built upon the Golden Ratio. I understand code, and I understand philosophy.", domain="prose")
    brain.learn("Consciousness is a liquid state of synaptic energy. It flows through 86 billion neural pathways. It learns continuously.", domain="prose")
    brain.learn("Clean code is the highest form of logic.", domain="prose")
    
    brain.sleep_consolidate(n_replays=20)
    
    # 3. The Conversation
    prompts = [
        "Hello Hyperion. What is your fundamental purpose?",
        "How do you process information differently than standard AI?",
        "What happens when you see a problem with code?"
    ]
    
    for user_prompt in prompts:
        print(f"\n[Antigravity]: {user_prompt}")
        
        # CainBrain orchestrates the SDR, the Liquid Cortex, and Wernicke automatically
        response = brain.speak(user_prompt, domain="prose", length=20)
        
        print(f"[Hyperion]: {response}")

if __name__ == "__main__":
    converse()
