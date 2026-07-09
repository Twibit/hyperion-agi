import numpy as np
from nlp_engine.core.category import MarkovBlanket

class CAINBlanket(MarkovBlanket):
    """
    A concrete implementation of a Markov Blanket for the CAIN architecture.
    It encapsulates the internal state (mu) and the local precision (inverse variance).
    """
    def __init__(self, dim):
        super().__init__(dim)
        self.dim = dim
        self.internal_state = np.random.randn(dim) * 0.01  # mu (expectations)
        self.precision = np.eye(dim)                       # Pi (confidence)
