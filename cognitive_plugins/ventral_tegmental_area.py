import numpy as np

class VentralTegmentalArea:
    """
    Motivation Engine using Temporal Difference Learning.
    Calculates Reward Prediction Error (RPE) to drive autonomous curiosity.
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.9):
        self.alpha = learning_rate
        self.gamma = discount_factor
        
        # Expected value of learning in general (V(t))
        self.expected_value = 0.5 
        
    def calculate_rpe(self, actual_info_value):
        """
        Calculates the Reward Prediction Error (RPE).
        RPE = Actual Reward + (discount * Future Expected Reward) - Current Expected Reward
        For pure curiosity, reward is the information value.
        """
        # Temporal Difference Error
        rpe = actual_info_value - self.expected_value
        
        # Update our expectation for the future
        self.expected_value = self.expected_value + self.alpha * rpe
        
        # We constrain expected value to biologically plausible limits [0, 1]
        self.expected_value = np.clip(self.expected_value, 0.0, 1.0)
        
        return rpe
        
    def get_dopamine_release(self, rpe):
        """
        Converts RPE into a biological dopamine multiplier.
        If RPE is positive (surprising good knowledge), massive dopamine release (e.g. 1.5x-2.0x).
        If RPE is negative (boring/expected), baseline or suppressed dopamine (e.g. 0.5x-1.0x).
        """
        # Baseline dopamine is 1.0
        dopamine_multiplier = 1.0 + rpe 
        
        # Ensure we don't go negative, and cap massive spikes
        return float(np.clip(dopamine_multiplier, 0.1, 3.0))
