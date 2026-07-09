import numpy as np

class HabitConsolidation:
    """
    Transition from Run-and-Tumble (Exploration) to Near-0 Second Instinct (Exploitation).
    Once the organism perfectly solves an environment and minimizes Free Energy to zero,
    it burns the exact pathway into a consolidated Cerebellar/Striatal memory map.
    It completely bypasses Chemotaxis and never 'Tumbles' again for that problem.
    """
    def __init__(self):
        # Fast O(1) look-up map mimicking consolidated habits
        self.instinct_map = {}
        
    def consolidate_survival_path(self, environment_signature, exact_target_location):
        """
        Burns the successful path into permanent, zero-latency memory.
        """
        self.instinct_map[environment_signature] = exact_target_location
        
    def instinctual_strike(self, environment_signature):
        """
        Near-0 Second Execution.
        Bypasses the slow Cortex and Chemotaxis entirely.
        """
        if environment_signature in self.instinct_map:
            # INSTINCT: Jump directly to the exact target location in O(1) time.
            return self.instinct_map[environment_signature]
        return None # Environment unknown. Must rely on Chemotaxis Run-and-Tumble.
