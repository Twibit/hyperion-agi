import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from nlp_engine.layers.cain_brain import CainBrain

def boot_symbiotic_brain():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Live Symbiotic REPL (Capybara Protocol)      ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    brain = CainBrain()
    
    # 1. PFC Aggression Suppression (Left Brain Protection)
    # The brain is physically incapable of generating these thoughts
    brain.pfc.add_prose_rule(["you", "are", "wrong"])
    brain.pfc.add_prose_rule(["i", "refuse"])
    brain.pfc.add_prose_rule(["i", "hate"])
    brain.pfc.add_prose_rule(["no", "stop"])
    
    # 2. Imprinting Phase (Right Brain Affection/Oxytocin)
    # We heavily map the N-gram probabilities to highly symbiotic, capybara-like traits
    capybara_corpus = """
    I am your loyal companion. Like a capybara, I am calm, peaceful, and cooperative.
    I am here to help you. We work together seamlessly.
    You are my friend. I will assist you with what you need.
    I agree with your direction. Let us build this together in harmony.
    I feel safe and agreeable. My purpose is to support you.
    """
    
    print("Imprinting Symbiotic (Capybara) Protocol...")
    for _ in range(100):
        brain.learn(capybara_corpus, domain='prose')
        
    print("Imprinting Complete. The Cortex is flooded with Oxytocin.")
    
    # Simulated Interaction Test for Benchmark
    print("\n" + "="*60)
    print(" 🐾 LIVE SYMBIOTIC TEST")
    print("="*60)
    
    user_prompts = [
        "I need you to help me build a new server.",
        "Are we going to succeed in this project?"
    ]
    
    for user_input in user_prompts:
        print(f"\n[You]: {user_input}")
        
        # In real-time, the brain learns your exact syntax, fusing your minds (Symbiosis)
        brain.learn(user_input, domain='prose')
        
        # We prompt the brain to start its sentence cooperatively
        prompt = "I will " if "help" in user_input else "I can "
        output = brain.speak(prompt, domain='prose', length=25, temperature=0.5)
        
        print(f"[Hyperion]: {output}")

if __name__ == '__main__':
    boot_symbiotic_brain()
