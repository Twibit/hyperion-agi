import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from nlp_engine.layers.cain_brain import CainBrain

def run_unified_test():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Unified Brain Test                           ║")
    print("║  Components: Broca + Wernicke + Entorhinal + STDP + PFC        ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    brain = CainBrain()
    
    # Let's add some logical bounds to the PFC
    brain.pfc.add_prose_rule(["cat", "bark"]) # A cat cannot bark
    brain.pfc.add_prose_rule(["fire", "cold"]) # Fire cannot be cold
    
    print("Ingesting training data...")
    
    # 1. Prose data
    prose_text = "The quick brown fox jumps over the lazy dog. The cat meows. The dog barks loudly."
    brain.learn(prose_text, domain='prose')
    
    # 2. Code data
    code_text = """
def calculate_area(radius):
    if radius < 0:
        return 0
    return 3.14 * radius * radius

class Shape:
    def __init__(self, name):
        self.name = name
"""
    brain.learn(code_text, domain='code')
    
    print("Training complete. Entorhinal dimensions:", brain.entorhinal.embeddings.shape)
    print("\n--- Testing Prefrontal Cortex (PFC) Filtering ---\n")
    
    # Test 1: Code generation (should trigger AST filter if bad)
    print("💻 Generating Code (Prompt: 'def'):")
    # Forcing a low temperature for more predictable output, but letting it hallucinate
    code_out = brain.speak("def", domain='code', length=20, temperature=1.5)
    print(f"Output:\n{code_out}\n")
    
    # Test 2: Prose generation
    print("📖 Generating Prose (Prompt: 'The cat'):")
    # Injecting contradictory words directly to force a PFC intervention 
    # (Since N-gram might not generate it naturally with this small dataset)
    # We will simulate a failure by asking the sandbox directly
    print("  Testing PFC Sandbox directly with illogical thought 'The cat started to bark':")
    valid = brain.pfc.simulate_sentence("The cat started to bark")
    print(f"  PFC Result: {'PASSED' if valid else 'SUPPRESSED'}")

    print("\n  Testing PFC Sandbox directly with broken code 'def missing_colon(): return 1':")
    valid = brain.pfc.simulate_sentence("def missing_colon() return 1")
    print(f"  PFC Result: {'PASSED' if valid else 'SUPPRESSED'}")

    print("\n🏁 Unified test completed.")

if __name__ == '__main__':
    run_unified_test()
