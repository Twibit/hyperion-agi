"""
thinking_loop.py - The Internal Monologue Engine
Routes the HTM predictor's output back into its own input, simulating an internal 'thought'
before motor execution, guided by a Goal SDR (Prefrontal Cortex).
"""
import numpy as np
from typing import List
from nlp_engine.layers.scale86b.sdr import SparseDistributedRepresentation
from nlp_engine.layers.predictive import HTMTemporalPredictor
from nlp_engine.layers.isomorphism import IsomorphismConceptMap

class PrefrontalCortex:
    def __init__(self):
        self.active_goal_sdr: SparseDistributedRepresentation = None
        self.arousal_decay = 0.95
        
    def set_goal(self, goal_sdr: SparseDistributedRepresentation):
        """Sets the persistent top-down Goal (Basal Context)."""
        self.active_goal_sdr = goal_sdr

class ThinkingLoop:
    def __init__(self, predictor: HTMTemporalPredictor, pfc: PrefrontalCortex, iso_map: IsomorphismConceptMap):
        self.predictor = predictor
        self.pfc = pfc
        self.iso_map = iso_map
        
    def daydream(self, start_concept: str, language: str, max_steps: int = 10) -> List[str]:
        """
        The Internal Monologue Mode.
        - Plasticity is strictly disabled (hallucination shield).
        - The predicted output is routed back as the next sensory input.
        - The sequence is steered by the PFC Goal SDR (Basal Context).
        """
        thought_sequence = []
        current_word = start_concept
        
        # Save original learning rate to prevent hallucinatory memory corruption
        original_lr = self.predictor.learning_rate
        self.predictor.learning_rate = 0.0  # PLASTICITY SUPPRESSED
        
        # Trigger the initial state
        current_sdr = self.iso_map.project_to_universal_sdr(current_word)
        thought_sequence.append(current_word)
        
        self.predictor.reset_state()
        
        for _ in range(max_steps):
            # Pass the Apical Context (Language) and the Basal Context (PFC Goal)
            pred_sdr, _ = self.predictor.process_step(
                current_sdr, 
                apical_context=language,
                basal_context=self.pfc.active_goal_sdr
            )
            
            # Stop if the thought naturally degrades to silence
            if len(pred_sdr.active_ids) == 0:
                break
                
            # Decode the predicted internal thought back into a string
            next_word = None
            
            # Fast reverse lookup into the specific language (Simulating Broca's Area decoding)
            next_word = None
            if language == "EN":
                lang_vocab = ["hello", "world", "water", "apple"]
            elif language == "ES":
                lang_vocab = ["hola", "mundo", "agua", "manzana"]
            else:
                lang_vocab = []
                
            for w in lang_vocab:
                w_sdr = self.iso_map.project_to_universal_sdr(w)
                if np.array_equal(np.sort(w_sdr.active_ids), np.sort(pred_sdr.active_ids)):
                    next_word = w
                    break
                    
            if not next_word:
                break # Thought drifted off the manifold
                
            thought_sequence.append(next_word)
            
            # The Loop: Feed the output back as the next input!
            current_word = next_word
            current_sdr = pred_sdr
            
            # Goal Satisfaction Check: If the generated concept perfectly matches the PFC Goal, the thought is complete.
            if self.pfc.active_goal_sdr is not None:
                intersection = len(np.intersect1d(pred_sdr.active_ids, self.pfc.active_goal_sdr.active_ids))
                if intersection > 30: # Strong overlap with the Goal
                    break
                    
        # Restore plasticity
        self.predictor.learning_rate = original_lr
        
        return thought_sequence
