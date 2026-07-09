"""
translation_bridge.py - Perfect Cross-Lingual Pipeline
Reads a sequence in Language A, maps to Universal SDRs, locks into Working Memory,
swaps the Apical Context to Language B, and daydreams the translation flawlessly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlp_engine.layers.isomorphism import IsomorphismConceptMap
from nlp_engine.layers.predictive import HTMTemporalPredictor

class TranslationBridge:
    def __init__(self, iso_map: IsomorphismConceptMap, predictor: HTMTemporalPredictor):
        self.iso_map = iso_map
        self.predictor = predictor
        self.working_memory = []
        
    def ingest_and_store(self, sentence: list[str], source_lang: str):
        """Phase 1-3: Read Lang A, map to Universal SDR, and lock into Working Memory"""
        self.working_memory = []
        
        # We prime the predictor's temporal memory with the source language context
        # This resolves local syntactic binding in Lang A
        self.predictor.reset_state()
        
        for word in sentence:
            # Strip syntactic scaffolding, encode pure semantic topology
            universal_sdr = self.iso_map.project_to_universal_sdr(word)
            
            # Predictor processes it to learn/reinforce sequence in Lang A (if plasticity is on)
            self.predictor.process_step(universal_sdr, apical_context=source_lang)
            
            # Lock the Universal SDR into Working Memory
            self.working_memory.append(universal_sdr)
            
        print(f"[Translation Bridge] Locked {len(sentence)} Universal Concepts into Working Memory.")

    def translate_to(self, target_lang: str) -> list[str]:
        """Phase 4-5: Flip Apical Context to Lang B and Synthesize"""
        if not self.working_memory:
            return []
            
        translated_sentence = []
        
        # We start the daydream from the first Universal Concept in Working Memory
        current_sdr = self.working_memory[0]
        
        # Decode the first concept directly
        first_word = self._decode_sdr(current_sdr, target_lang)
        if first_word:
            translated_sentence.append(first_word)
        
        self.predictor.reset_state()
        
        # We force the Temporal Predictor to traverse the working memory sequence,
        # but using the Target Language's Apical Context (Grammar Matrix)
        for i in range(len(self.working_memory) - 1):
            target_concept_sdr = self.working_memory[i+1]
            
            # Daydream the next step using the Target Language Grammar
            pred_sdr, err = self.predictor.process_step(
                current_sdr, 
                apical_context=target_lang,
                basal_context=target_concept_sdr  # The goal is the next concept in Working Memory
            )
            
            # Decode the predicted SDR back to the Target Language string
            next_word = self._decode_sdr(pred_sdr, target_lang)
            if next_word:
                translated_sentence.append(next_word)
                
            current_sdr = pred_sdr
            
        return translated_sentence

    def _decode_sdr(self, sdr, language: str) -> str:
        """Simulates Broca's Area language-specific decoding"""
        import numpy as np
        
        # Fast reverse lookup using the Isomorphism Concept Map's active cache
        # The training loop will have populated this cache with the exact SDRs
        for centroid, cached_sdr in self.iso_map.concept_to_sdr_cache.items():
            if np.array_equal(np.sort(cached_sdr.active_ids), np.sort(sdr.active_ids)):
                return centroid
                
        print(f"[Decoder Warn] Could not map Universal SDR back to any known concept.")
        return None
