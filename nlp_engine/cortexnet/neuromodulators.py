"""
neuromodulators.py - The Endocrine & Ethics System
Maintains Dopamine (Accuracy), Serotonin (Ethics/Pro-social), and Cortisol (Error/Toxic) levels.
Controls synaptic plasticity and permanent pruning (LTD).
"""
import numpy as np

class NeuromodulatorState:
    def __init__(self):
        self.dopamine = 0.1     # Learning rate for grammatical accuracy
        self.serotonin = 0.5    # Reward for ethical, pro-social alignment
        self.cortisol = 0.0     # Stress response for errors or toxic ideas
        
        self.arousal = 0.0
        
    def update_from_prediction_error(self, pe: float):
        """Standard grammar training loop."""
        if pe > 0.5:
            # High surprise/error -> Stress -> Cortisol
            self.cortisol = min(1.0, self.cortisol + pe * 0.2)
            self.dopamine = max(0.0, self.dopamine - 0.1)
        else:
            # Low error -> Expected reward -> Dopamine
            self.dopamine = min(1.0, self.dopamine + (1.0 - pe) * 0.1)
            self.cortisol = max(0.0, self.cortisol - 0.1)
            
    def update_arousal(self, pe: float):
        self.arousal = 0.9 * self.arousal + 0.1 * pe
        
    def evaluate_ethics(self, idea_sdr, critic_score: float):
        """
        Ethics Loop (Alignment). 
        critic_score: 1.0 (Highly pro-social), -1.0 (Highly toxic).
        """
        if critic_score > 0:
            self.serotonin = min(1.0, self.serotonin + critic_score * 0.2)
            self.cortisol = max(0.0, self.cortisol - 0.2)
        else:
            self.cortisol = min(1.0, self.cortisol + abs(critic_score) * 0.5)
            self.serotonin = max(0.0, self.serotonin - 0.3)
            
    def eta(self, base=0.05) -> float:
        """Returns the dynamic learning rate based on endocrine states."""
        # Serotonin and Dopamine boost plasticity for good things.
        # Cortisol drives plasticity deeply negative for pruning (LTD).
        if self.cortisol > 0.7:
            # Toxic/Wrong thought -> Execute Permanent Prune
            return -1.0 
        
        plasticity = base + (self.dopamine * 0.05) + (self.serotonin * 0.02) - (self.cortisol * 0.1)
        return plasticity
