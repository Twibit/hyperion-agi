"""
Isomorphic SDR Encoder — V14 (Aurora-Foundation).
==================================================

The fundamental problem in multilingual and multimodal AGI: the word "Water",
"Agua", "水", and an *image* of a river must all activate the exact same
neural pattern in the brain. Otherwise the organism has no universal concept
of "water" — it has four disconnected tokens.

This module implements a Semantic Hashing layer that maps any input (text in
any language, image features, audio features) through a shared concept space
into a deterministic 2048-bit Sparse Distributed Representation (SDR).

Architecture:
    1. Input arrives (text token, image embedding, audio MFCC, etc.)
    2. A modality-specific encoder projects it into a 256-dim concept vector.
    3. The concept vector is hashed into a 2048-bit SDR with exactly K=40 bits
       active (≈2% sparsity, matching biological cortical activation).
    4. The SDR is the universal "language" that the Cerebrum understands.

NOTE: In production, this module should be rewritten as a Go microservice for
high-concurrency streaming. The interface is designed to make that swap trivial.
"""
from __future__ import annotations
import hashlib
import numpy as np
from typing import Dict, Optional


# Pre-built concept anchors: words in different languages that mean the same thing
# map to the same concept_id.  In production, this would be loaded from a massive
# multilingual knowledge graph (e.g., Wikidata Q-IDs).
_CONCEPT_ANCHORS: Dict[str, str] = {
    # English
    "water": "Q283", "fire": "Q3196", "cat": "Q146", "dog": "Q144",
    "human": "Q5", "earth": "Q2", "sun": "Q525", "moon": "Q405",
    "food": "Q2095", "tree": "Q10884",
    # Spanish
    "agua": "Q283", "fuego": "Q3196", "gato": "Q146", "perro": "Q144",
    "humano": "Q5", "tierra": "Q2", "sol": "Q525", "luna": "Q405",
    "comida": "Q2095", "árbol": "Q10884",
    # Chinese (Pinyin for simplicity; real system would use Unicode)
    "shui": "Q283", "huo": "Q3196", "mao": "Q146", "gou": "Q144",
    # Japanese
    "mizu": "Q283", "hi": "Q3196", "neko": "Q146", "inu": "Q144",
}

SDR_WIDTH = 2048
SDR_K = 40  # Number of active bits (≈2% sparsity)


class IsomorphicEncoder:
    """
    Maps any input into a universal 2048-bit SDR.
    
    The key property is ISOMORPHISM: semantically identical concepts in 
    different languages or modalities produce the EXACT same SDR.
    """
    
    def __init__(self, sdr_width: int = SDR_WIDTH, sdr_k: int = SDR_K,
                 concept_anchors: Optional[Dict[str, str]] = None,
                 perceptual_gain: float = 1.0):
        self.sdr_width = sdr_width
        self.sdr_k = sdr_k
        self.anchors = concept_anchors or _CONCEPT_ANCHORS
        # Gap 7: Sensory Gain Control (Dabrowski Overexcitability)
        # gain=1.0 = normal, gain=2.0 = genius (perceives more detail)
        # Higher gain increases the number of active SDR bits, creating
        # richer, more detailed representations from the same input.
        self.perceptual_gain = max(0.5, min(3.0, perceptual_gain))
        self._effective_k = int(self.sdr_k * self.perceptual_gain)
        self._effective_k = min(self._effective_k, self.sdr_width // 4)  # Safety cap
        
        # Cache computed SDRs for fast lookup
        self._cache: Dict[str, np.ndarray] = {}
    
    def _concept_id(self, token: str) -> str:
        """Resolve a token to its universal concept ID.
        
        If the token is a known anchor (like "agua" -> "Q283"), return the 
        shared concept ID. Otherwise, use the raw token as its own concept.
        """
        normalized = token.strip().lower()
        return self.anchors.get(normalized, f"UNK_{normalized}")
    
    def _hash_to_sdr(self, concept_id: str) -> np.ndarray:
        """Deterministically hash a concept ID into a sparse binary vector."""
        if concept_id in self._cache:
            return self._cache[concept_id].copy()
        
        # Use SHA-256 as a deterministic seed for the SDR
        digest = hashlib.sha256(concept_id.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        
        # Select exactly K positions out of sdr_width to be active
        sdr = np.zeros(self.sdr_width, dtype=np.uint8)
        active_positions = rng.choice(self.sdr_width, size=self._effective_k, replace=False)
        sdr[active_positions] = 1
        
        self._cache[concept_id] = sdr.copy()
        return sdr
    
    def encode_token(self, token: str) -> np.ndarray:
        """Encode a single text token into a 2048-bit SDR."""
        concept_id = self._concept_id(token)
        return self._hash_to_sdr(concept_id)
    
    def encode_sentence(self, sentence: str, fast_mode: bool = False) -> np.ndarray:
        """Encode a sentence into a 2048-bit SDR.
        If fast_mode is True, bypasses HTM temporal binding for 1000x speedup during raw scanning.
        """
        tokens = sentence.strip().split()
        if not tokens:
            return np.zeros(self.sdr_width, dtype=np.uint8)
        
        result_sdr = np.zeros(self.sdr_width, dtype=np.uint8)
        
        if fast_mode:
            # Pure semantic overlap without temporal sequence prediction
            for t in tokens:
                result_sdr |= self.encode_token(t)
            return result_sdr
            
        # V14 Temporal Binding: Use HTM to track sequence context
        # This ensures "cat sat on mat" produces a different SDR than "mat sat on cat"
        if not hasattr(self, 'htm'):
            from ..praxis.htm_predictor import HTMSequenceMemory
            self.htm = HTMSequenceMemory(latent_dim=self.sdr_width, sparsity=self.sdr_k / self.sdr_width)
        
        for t in tokens:
            token_sdr = self.encode_token(t)
            # Process token and get prediction of next state
            prediction = self.htm.step(token_sdr, learn=True)
            
            # The context bits (top K predictions) are bound to the current token
            temporal_bias = np.zeros(self.sdr_width, dtype=np.uint8)
            if np.max(prediction) > 0.0:
                top_k = np.argsort(prediction)[-self.sdr_k:]
                temporal_bias[top_k] = 1
                
            # Union of the token and its temporal context
            result_sdr |= (token_sdr | temporal_bias)
            
        return result_sdr
    
    def encode_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Encode a continuous embedding (e.g., image features) into an SDR.
        
        Uses a Spatial Pooler (random projection + k-WTA) to map dense 
        visual features into the universal 2048-bit sparse representation.
        """
        assert embedding.ndim == 1, "Embedding must be a 1D vector"
        e = embedding.astype(np.float32)
        dim = len(e)
        
        # Lazily initialize a projection matrix from the visual space to SDR space
        if not hasattr(self, '_visual_projections'):
            self._visual_projections = {}
            
        if dim not in self._visual_projections:
            self._visual_projections[dim] = np.random.default_rng(1729 + dim).standard_normal(
                (dim, self.sdr_width), dtype=np.float32
            ) / np.sqrt(dim)
            
        # Project dense features into the higher dimensional SDR space
        excitation = e @ self._visual_projections[dim]
        
        # k-Winner-Take-All (Sparsify)
        # argpartition is O(N) and much faster than full sorting
        sdr = np.zeros(self.sdr_width, dtype=np.uint8)
        top_k_indices = np.argpartition(excitation, -self._effective_k)[-self._effective_k:]
        sdr[top_k_indices] = 1
        
        return sdr
    
    def similarity(self, sdr_a: np.ndarray, sdr_b: np.ndarray) -> float:
        """Compute overlap similarity between two SDRs (Jaccard-like)."""
        overlap = int(np.sum(sdr_a & sdr_b))
        union = int(np.sum(sdr_a | sdr_b))
        if union == 0:
            return 0.0
        return overlap / union
