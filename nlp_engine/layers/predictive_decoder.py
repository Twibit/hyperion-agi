import numpy as np
from nlp_engine.layers.entorhinal_cortex import EntorhinalCortex

class PredictiveDecoder:
    """
    Biological language generation via Predictive Coding.
    Instead of selecting words via an N-gram frequency distribution,
    it maintains a hidden state and predicts the *next semantic vector*.
    It then finds the closest word in the Entorhinal Cortex to that predicted vector.
    """
    def __init__(self, entorhinal: EntorhinalCortex, hidden_dim: int = 64, learning_rate: float = 0.01):
        self.entorhinal = entorhinal
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Recurrent weight matrix
        self.W_rec = np.random.normal(0, 0.1, (hidden_dim, hidden_dim))
        # Input weight matrix
        self.W_in = np.random.normal(0, 0.1, (entorhinal.embedding_dim, hidden_dim))
        
        self.hidden_state = np.zeros(hidden_dim)
        
    def step(self, input_vector: np.ndarray) -> np.ndarray:
        """Update hidden state based on new input vector."""
        self.hidden_state = np.tanh(np.dot(self.hidden_state, self.W_rec) + np.dot(input_vector, self.W_in))
        return self.hidden_state
        
    def predict_next_vector(self) -> np.ndarray:
        """Predicts the next semantic vector based on current hidden state."""
        # For simplicity in this iteration, we assume hidden_dim == embedding_dim
        # and the hidden state IS the predicted next semantic vector.
        return self.hidden_state
        
    def decode_word(self, predicted_vector: np.ndarray, temperature: float = 0.5) -> str:
        """Decodes the predicted vector to the nearest word in Entorhinal Cortex.

        Uses temperature-scaled sampling over cosine similarity rather than a
        hard greedy argmax, so the generator explores and the inner-monologue
        chain avoids collapsing into an exact loop.
        """
        if len(self.entorhinal.token_to_idx) == 0:
            return ""

        toks = list(self.entorhinal.token_to_idx.keys())
        idxs = [self.entorhinal.token_to_idx[t] for t in toks]
        vecs = self.entorhinal.embeddings[idxs]
        target_norm = np.linalg.norm(predicted_vector) + 1e-8
        norms = np.linalg.norm(vecs, axis=1) + 1e-8
        sims = (vecs @ predicted_vector) / (norms * target_norm)

        t = max(1e-3, temperature)
        logits = sims / t
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return toks[int(np.random.choice(len(toks), p=probs))]
        
    def learn_transition(self, current_vector: np.ndarray, next_vector: np.ndarray, reward: float = 1.0) -> float:
        """
        Predictive coding update: adjust weights to minimize prediction error.
        This represents the top-down prediction error correction.
        Returns the magnitude of the prediction error (surprise).
        """
        # Forward pass (done in step, but we recompute for gradients)
        hidden = np.tanh(np.dot(self.hidden_state, self.W_rec) + np.dot(current_vector, self.W_in))
        
        # Error is the difference between what we predicted (hidden) and what actually came next
        # In a real predictive coding network, error flows backwards to update weights.
        error = next_vector - hidden
        error_magnitude = np.linalg.norm(error)
        
        # Simple biologically-plausible delta rule (Hebbian approximation of backprop)
        # Weight update = learning_rate * error * presynaptic_activity * reward
        
        # Update input weights
        dW_in = np.outer(current_vector, error) * self.learning_rate * reward
        self.W_in += dW_in
        
        # Update recurrent weights
        dW_rec = np.outer(self.hidden_state, error) * self.learning_rate * reward
        self.W_rec += dW_rec
        
        return float(error_magnitude)
