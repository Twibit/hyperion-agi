"""
isomorphism.py - The Universal Concept Manifold
Forces distinct multilingual inputs (e.g. 'water', 'agua', '水') to collapse into the exact same 2048-bit Concept SDR.
"""
import numpy as np
import hashlib
from typing import Dict, List
from nlp_engine.layers.scale86b.sdr import SparseDistributedRepresentation

class IsomorphismConceptMap:
    def __init__(self, sdr_size: int = 2048, sparsity: float = 0.02):
        self.sdr_size = sdr_size
        self.k_wta = int(sdr_size * sparsity) # Top k active bits (e.g., 41)
        
        # In a real system, this is trained via Contrastive Semantic Alignment.
        # Here we simulate the learned manifold mapping by explicitly tying known translations
        # to a shared, deterministic mathematical seed.
        self.concept_clusters = [
            {"water", "agua", "eau", "水"},
            {"fire", "fuego", "feu", "火"},
            {"hello", "hola", "bonjour", "你好"},
            {"world", "mundo", "monde", "世界"},
            {"apple", "manzana", "pomme", "苹果"}
        ]
        
        self.concept_to_sdr_cache: Dict[str, SparseDistributedRepresentation] = {}
        
    def _get_cluster_centroid(self, word: str) -> str:
        """Finds the universal semantic cluster for a word."""
        word = word.lower()
        for cluster in self.concept_clusters:
            if word in cluster:
                # Use the sorted first element of the set as the universal "centroid" key
                return sorted(list(cluster))[0]
        return word # Fallback to the word itself if no translation cluster is known

    def project_to_universal_sdr(self, word: str, region: str = "cortex") -> SparseDistributedRepresentation:
        """
        Projects a language-specific word into the universal concept manifold.
        """
        centroid_key = self._get_cluster_centroid(word)
        
        if centroid_key in self.concept_to_sdr_cache:
            return self.concept_to_sdr_cache[centroid_key]
            
        # Generate a deterministic but pseudo-random sparse representation based on the centroid
        seed = int(hashlib.md5(centroid_key.encode()).hexdigest(), 16) % (2**32 - 1)
        rng = np.random.RandomState(seed)
        
        # k-WTA Binarization: We pick exactly `k` random active indices out of `sdr_size`
        active_indices = np.sort(rng.choice(self.sdr_size, size=self.k_wta, replace=False)).astype(np.int64)
        weights = np.ones(self.k_wta, dtype=np.float32)
        
        sdr = SparseDistributedRepresentation(active_ids=active_indices, weights=weights, region=region)
        self.concept_to_sdr_cache[centroid_key] = sdr
        
        return sdr
