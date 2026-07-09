from typing import Any, Tuple
import numpy as np

class MarkovBlanket:
    """
    Topological boundary enforcing conditional independence.
    Internal states (mu) can only interact with the outside world via:
    - Sensory States (Input)
    - Active States (Output)
    """
    def __init__(self, state_dim: int):
        self.internal_state = np.zeros(state_dim)
        self.sensory_state = np.zeros(state_dim)
        self.active_state = np.zeros(state_dim)

class ParametrizedLens:
    """
    Category Theory Optic / Lens.
    Provides strict compositionality. Instead of a monolithic backprop chain,
    modules are isolated lenses that map internal generative states (View) 
    and update based on precision-weighted errors (Update).
    """
    
    def view(self, blanket: MarkovBlanket) -> np.ndarray:
        """
        Forward Pass: Maps the internal generative model to a prediction
        in the sensory space.
        """
        raise NotImplementedError("Lenses must implement a forward View.")

    def update(self, blanket: MarkovBlanket, precision_weighted_error: np.ndarray) -> np.ndarray:
        """
        Backward Pass: Modifies the internal state based strictly on local
        Free Energy gradients, returning the error to pass to the next lens.
        No continuous auto-diff leaking across boundaries.
        """
        raise NotImplementedError("Lenses must implement an Update mechanism.")

class ComposedLens(ParametrizedLens):
    """
    Associatively composes two lenses (Lens A and Lens B).
    Perfectly routes Active Inference through a nested hierarchy.
    """
    def __init__(self, lens_a: ParametrizedLens, lens_b: ParametrizedLens):
        self.lens_a = lens_a
        self.lens_b = lens_b

    def view(self, blanket: MarkovBlanket) -> np.ndarray:
        # Pass prediction up the hierarchy
        a_out = self.lens_a.view(blanket)
        return self.lens_b.view(a_out)

    def update(self, blanket: MarkovBlanket, precision_weighted_error: np.ndarray) -> np.ndarray:
        # Route precision errors back down locally
        b_err = self.lens_b.update(blanket, precision_weighted_error)
        return self.lens_a.update(blanket, b_err)
