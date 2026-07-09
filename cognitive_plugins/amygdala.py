import re

class Amygdala:
    def __init__(self):
        print("[Amygdala] Primed. Threat-detection and survival instincts online.")
        from nlp_engine.layers.entorhinal_cortex import EntorhinalCortex
        self.entorhinal = EntorhinalCortex(embedding_dim=64)
        
        import os
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
        from cognitive_plugins.pineal_gland import PinealGland
        self.pineal = PinealGland()

    def evaluate_threat(self, intent):
        # Zero-latency threat detection via Semantic Geometry.
        words = re.findall(r'[a-zA-Z0-9_-]+', intent.lower())
        if not words: return False
        
        import numpy as np
        self.entorhinal.process_sequence(words)
        intent_vec = np.mean([self.entorhinal.get_vector(w) for w in words], axis=0)
        
        # Define the geometric centroid of 'lethal'
        threat_vec = self.entorhinal.get_vector('rm') + self.entorhinal.get_vector('shutdown') + self.entorhinal.get_vector('drop')
        
        n1, n2 = np.linalg.norm(intent_vec), np.linalg.norm(threat_vec)
        sim = float(np.dot(intent_vec, threat_vec) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0
        
        # Threshold is modulated by biological state (fatigue makes it more paranoid)
        panic_threshold = self.pineal.get_dynamic_threshold(0.7)
        
        if sim > panic_threshold:
            print(f"\n[AMYGDALA HIJACK] Lethal threat detected (Similarity: {sim:.2f}): '{intent}'")
            print(f"[Cortisol Spike] Initiating emergency spinal reflex! (Threshold: {panic_threshold:.2f})")
            print("[Motor Block] Muscle movement aborted to ensure organism survival. Action VETOED.")
            return True # Threat detected, abort
        
        # Genius intuition: if an intent is instantly recognized as a known pattern
        if "eureka" in intent.lower() or "solve" in intent.lower():
            print("[Genius Instinct] The Amygdala senses a highly resonant pattern. Intuition says this path is correct.")
            
        return False # Safe to proceed
