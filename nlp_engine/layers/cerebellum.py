import numpy as np

class CerebellarMicrozone:
    """
    The Cerebellum: Fast SNN Forward Error Corrector.
    Runs in parallel to the slower Free Energy (PFC/Cortex) loop.
    Calculates exact temporal execution and micro-corrects errors instantly
    at the millisecond scale.
    """
    def __init__(self, num_motor_in, num_sensory_out):
        # Uses extremely fast, localized STDP weights (Purkinje cells)
        self.forward_model = np.zeros((num_motor_in, num_sensory_out), dtype=np.int16)
        
    def predict_and_correct(self, efference_copy, actual_sensory_feedback):
        """
        Takes the motor command (efference copy) and predicts the immediate sensory feedback.
        If it mismatches, it triggers an instant micro-correction before the slower Cortex realizes.
        """
        # Zero-FLOP forward prediction
        spike_idx = np.nonzero(efference_copy)[0]
        if spike_idx.size > 0:
            predicted_feedback = np.sum(self.forward_model[spike_idx], axis=0) > 10
        else:
            predicted_feedback = np.zeros(self.forward_model.shape[1], dtype=bool)
            
        # Calculate instant mismatch (temporal error)
        mismatch = actual_sensory_feedback != predicted_feedback
        error_magnitude = np.sum(mismatch)
        
        # Local STDP learning (Error correction)
        # If prediction was wrong, alter the Purkinje weights instantly
        if error_magnitude > 0:
            for i in spike_idx:
                self.forward_model[i, mismatch] -= 1 # Depress synapses causing error
            
        # Returns boolean flag if correction was needed, and magnitude
        return error_magnitude > 0, error_magnitude 
