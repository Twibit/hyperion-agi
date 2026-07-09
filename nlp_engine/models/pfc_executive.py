import numpy as np
from nlp_engine.core.category import MarkovBlanket

class PrefrontalExecutive:
    """
    The Prefrontal Cortex (PFC): Category Theory Executive Logic.
    Maintains long-term Working Memory and sets the overarching Target Priors
    that the rest of the brain attempts to minimize Free Energy against.
    """
    def __init__(self, dim):
        # Sustained working memory blanket (does not decay rapidly)
        self.working_memory = MarkovBlanket(dim)
        
    def set_macro_policy(self, abstract_goal_vector):
        """
        Translates a high-level goal into a sustained Target Prior 
        that will drive the Basal Ganglia and Chemotaxis engines.
        """
        # Lock the goal into working memory
        self.working_memory.internal_state = abstract_goal_vector
        return self.working_memory.internal_state
        
    def contextualize_error(self, ascending_surprise):
        """
        If the Thalamus routes a massive error up to the PFC, the PFC must
        update its abstract models (Working Memory) to adapt to reality.
        """
        # Slow, long-horizon update of the internal generative model
        # (The PFC learns slower than the Cerebellum, focusing on deep rules rather than fast execution)
        self.working_memory.internal_state -= 0.001 * ascending_surprise
