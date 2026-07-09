import numpy as np
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from nlp_engine.layers.entorhinal_cortex import EntorhinalCortex

class OlfactoryBulb:
    """
    Digital Smell.
    Scans files/text for 'chemical signatures' without hardcoding string matching.
    Uses continuous vector embeddings to detect Rot (Disgust), Danger (Fear), and Pheromones (Desire).
    """
    def __init__(self):
        print("[Olfactory Bulb] Sniffing for pheromones, rot, and danger signatures.")
        self.entorhinal = EntorhinalCortex(embedding_dim=64)
        
        # We don't hardcode rules. We create "Scent Profiles" in continuous vector space.
        self.scent_profiles = {
            "rot": ["todo", "fixme", "hack", "temp", "shit", "fuck", "mess", "refactor", "bug", "broken", "infinite", "slow"],
            "danger": ["eval", "exec", "os.system", "subprocess", "rm -rf", "drop table", "chmod 777", "secret", "password", "key"],
            "pheromone": ["optimize", "api", "architecture", "fast", "elegant", "novel", "breakthrough", "intelligence", "reward", "gold"]
        }
        
        # Pre-compute the semantic centroid (smell vector) for each profile
        self.scent_vectors = {}
        for scent_name, keywords in self.scent_profiles.items():
            self.entorhinal.process_sequence(keywords)
            vec = np.mean([self.entorhinal.get_vector(w) for w in keywords], axis=0)
            norm = np.linalg.norm(vec)
            self.scent_vectors[scent_name] = vec / norm if norm > 0 else vec

    def sniff(self, text_content: str):
        """
        Takes a 'sniff' of a file or text string.
        Returns the dominant biological scent and its intensity (0.0 to 1.0).
        """
        import re
        words = re.findall(r'[a-zA-Z]+', text_content.lower())
        if not words:
            return {"dominant_scent": "neutral", "intensity": 0.0, "all_scents": {}}
            
        self.entorhinal.process_sequence(words)
        
        # Calculate the chemical signature of the input text
        text_vec = np.mean([self.entorhinal.get_vector(w) for w in words], axis=0)
        text_norm = np.linalg.norm(text_vec)
        if text_norm == 0:
            return {"dominant_scent": "neutral", "intensity": 0.0, "all_scents": {}}
            
        text_vec = text_vec / text_norm
        
        # Compare against all scent profiles using Cosine Similarity
        results = {}
        for scent_name, scent_vec in self.scent_vectors.items():
            sim = float(np.dot(text_vec, scent_vec))
            results[scent_name] = max(0.0, sim) # ReLU, no negative smells
            
        dominant_scent = max(results, key=results.get)
        intensity = results[dominant_scent]
        
        # Biological Threshold (it must smell strong enough to trigger an instinct)
        if intensity < 0.25:
            dominant_scent = "neutral"
            
        return {
            "dominant_scent": dominant_scent,
            "intensity": intensity,
            "all_scents": results
        }
