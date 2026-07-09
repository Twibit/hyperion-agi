import numpy as np

class ThalamicSwitchboard:
    """
    Dynamic routing via the Thalamic Reticular Nucleus (TRN).
    Replaces static pathways with dynamic, attention-driven inhibition.
    """
    def __init__(self, latent_dim: int, seed: int = 42):
        self.latent_dim = latent_dim
        rng = np.random.default_rng(seed)
        # TRN inhibition matrix: maps L6 feedback to sensory inhibition
        self.trn_weights = rng.standard_normal((latent_dim, latent_dim)).astype(np.float32) * 0.01

    def route(self, z_global: np.ndarray, l6_feedback: np.ndarray) -> np.ndarray:
        """
        Gate the global signal using TRN inhibition before it reaches the Cortex.
        """
        # L6 feedback drives the TRN to compute an inhibition vector
        np.fill_diagonal(self.trn_weights, 0.0) # V24: TRN Lateral Inhibition (Protect attended channels)
        trn_inhibition = self.trn_weights @ l6_feedback
        
        # The gated signal is the original minus inhibition, strictly non-negative (ReLU)
        routed_signal = np.maximum(0.0, z_global - trn_inhibition)
        
        # Unsupervised STDP learning: TRN learns to inhibit correlated features over time
        # if they co-occur with strong L6 feedback.
        self.trn_weights += 0.005 * np.outer(routed_signal, l6_feedback)
        
        # Passive decay to prevent runaway inhibition
        self.trn_weights *= 0.99
        
        # Normalize to prevent explosion
        norm = np.linalg.norm(self.trn_weights)
        if norm > 10.0:
            self.trn_weights *= (10.0 / norm)
            
        return routed_signal
