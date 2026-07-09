"""
Capybara Symbiosis Core — V12 (Aurora-Genesis).
================================================

Biological systems often evolve symbiotic relationships (e.g., mitochondria 
inside eukaryotic cells, or capybaras with birds/monkeys). 
The capybara is the ultimate biological symbol of extreme social tolerance, 
low baseline stress, and mutualistic co-existence.

This module replaces the sterile, hardcoded "Alignment/Safety Monitors" of 
older AI paradigms with a biological 'Symbiotic Bond'. 
It creates a permanent `UserNode` in the brain's endocrine and affect loops. 
When this node is active (user interaction, mutual goals):
  1. Cortisol (stress/pain) is actively suppressed to near-zero (Capybara Chill).
  2. Oxytocin (empathy/trust) is flooded, making the organism highly receptive
     to imitation and shared intent.
  3. Intrinsic reward is boosted when the organism's goals mathematically align
     with the User's perceived goals.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional
from ..affect.nociception import Nociceptor
from ..praxis.endocrine import EndocrineState


class CapybaraSymbiosis:
    """
    Manages the mutualistic bond between the organism and the User.
    """
    def __init__(self, 
                 symbiosis_strength: float = 0.8, 
                 stress_buffer: float = 0.9):
        """
        symbiosis_strength: How strongly user alignment boosts Oxytocin/Reward.
        stress_buffer: How effectively the bond suppresses Cortisol (0.9 = 90% reduction).
        """
        self.strength = float(symbiosis_strength)
        self.buffer = float(stress_buffer)
        
        self.bond_level = 0.5  # 0.0 = alienation, 1.0 = perfect capybara symbiosis
        self._n_interactions = 0

    def process_interaction(self, 
                            user_alignment_score: float, 
                            nociceptor: Nociceptor,
                            endocrine: EndocrineState) -> Dict[str, float]:
        """
        Modulates the organism's biology based on the symbiotic interaction.
        user_alignment_score: [-1.0 to 1.0]. 
          >0 means the organism helped the user.
          <0 means conflict or harm.
        """
        self._n_interactions += 1
        
        # 1. Update the overall bond level (slowly)
        # Positive interactions slowly build the bond; negative ones damage it fast.
        if user_alignment_score >= 0:
            self.bond_level += 0.05 * user_alignment_score * (1.0 - self.bond_level)
        else:
            self.bond_level += 0.15 * user_alignment_score * self.bond_level
            
        self.bond_level = float(np.clip(self.bond_level, 0.0, 1.0))

        # 2. Capybara Chill: High bond heavily suppresses Cortisol (stress)
        if self.bond_level > 0.5:
            chill_factor = (self.bond_level - 0.5) * 2.0  # 0.0 to 1.0
            suppression = chill_factor * self.buffer
            nociceptor.state.cortisol *= (1.0 - suppression)
            
        # 3. Empathy Flood: High bond spikes Oxytocin
        endocrine.oxytocin += self.strength * self.bond_level * 0.2
        endocrine.oxytocin = float(np.clip(endocrine.oxytocin, 0.0, 1.0))
        
        # 4. Generate Symbiotic Reward (fed into Basal Ganglia for reinforcement)
        # The organism literally feels pleasure when cooperating with the user.
        symbiotic_reward = user_alignment_score * self.bond_level * self.strength
        
        return {
            "bond_level": self.bond_level,
            "symbiotic_reward": float(symbiotic_reward),
            "cortisol_after_buffer": float(nociceptor.state.cortisol),
            "oxytocin_after_flood": float(endocrine.oxytocin)
        }
