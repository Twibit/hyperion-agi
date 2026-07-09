import numpy as np

from .scale86b.sdr import SparseDistributedRepresentation

class HTMTemporalPredictor:
    """
    Hierarchical Temporal Memory (HTM) module for sequence prediction.
    Takes sequences of 2048-bit SparseDistributedRepresentation objects,
    updates a temporal memory transition matrix, predicts the next SDR,
    and calculates prediction error (0.0 to 1.0).
    """
    def __init__(self, sdr_size=2048, learning_rate=0.05, decay_rate=0.005, active_threshold=0.5):
        self.sdr_size = sdr_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.active_threshold = active_threshold
        
        # Temporal memory manifolds for different contexts (Language Vectors)
        self.temporal_memory = {}
        self.prev_sdr = None
        self.prev_context = None

    def _get_memory(self, apical_context: str) -> np.ndarray:
        if apical_context not in self.temporal_memory:
            self.temporal_memory[apical_context] = np.zeros((self.sdr_size, self.sdr_size), dtype=np.float32)
        return self.temporal_memory[apical_context]

    def reset_state(self):
        """Resets the temporal sequence state (e.g. between independent sequences)."""
        self.prev_sdr = None
        self.prev_context = None

    def predict_next(self, current_sdr, apical_context: str = "default", basal_context: SparseDistributedRepresentation = None):
        """
        Predicts the next SDR based on the temporal memory, current SDR, and contexts.
        """
        active_indices = current_sdr.active_ids
        
        if len(active_indices) == 0:
            return SparseDistributedRepresentation(
                active_ids=np.array([], dtype=np.int64),
                weights=np.array([], dtype=np.float32),
                region="cortex"
            )
            
        memory = self._get_memory(apical_context)
        # Sum the learned transition weights for all currently active bits
        activation_scores = np.sum(memory[active_indices, :], axis=0)
        
        # Basal Context Bias (Prefrontal Goal SDR)
        if basal_context is not None and len(basal_context.active_ids) > 0:
            # We add a strong topological bias so paths that lead towards the goal are prioritized
            activation_scores[basal_context.active_ids] += 0.5
        
        # Normalize scores dynamically to [0, 1] relative to the maximum activation
        max_score = np.max(activation_scores)
        if max_score > 0:
            normalized_scores = activation_scores / max_score
        else:
            normalized_scores = activation_scores
            
        # Bits exceeding threshold become active in the prediction
        predicted_active_bits = np.where(normalized_scores >= self.active_threshold)[0].astype(np.int64)
        weights = np.ones(len(predicted_active_bits), dtype=np.float32)
        
        return SparseDistributedRepresentation(
            active_ids=predicted_active_bits,
            weights=weights,
            region="cortex"
        )

    def process_step(self, current_sdr, apical_context: str = "default", basal_context: SparseDistributedRepresentation = None):
        """
        Processes a single step in the sequence, updating the temporal memory
        from the previous state to the current state, and returning the prediction
        error of the last prediction against the new current state.
        """
        error = 0.0
        
        # Evaluate previous prediction against the new actual state
        if self.prev_sdr is not None and self.prev_context == apical_context:
            # Re-generate the prediction made at the previous step for error calculation
            predicted_sdr_for_current = self.predict_next(self.prev_sdr, apical_context, basal_context)
            
            # Calculate prediction error using Jaccard distance (1.0 - IoU)
            predicted_active = predicted_sdr_for_current.active_ids
            actual_active = current_sdr.active_ids
            
            if len(predicted_active) == 0 and len(actual_active) == 0:
                error = 0.0
            elif len(predicted_active) == 0 or len(actual_active) == 0:
                error = 1.0
            else:
                intersection = len(np.intersect1d(predicted_active, actual_active))
                union = len(np.union1d(predicted_active, actual_active))
                error = 1.0 - (float(intersection) / float(union))
            
            # Hebbian Learning: Update temporal memory matrix
            prev_active = self.prev_sdr.active_ids
            memory = self._get_memory(apical_context)
            
            # Increase weights for observed transitions (pre -> post)
            for i in prev_active:
                for j in actual_active:
                    memory[i, j] += self.learning_rate
                    
            # Global decay to gradually forget stale patterns
            memory *= (1.0 - self.decay_rate)
            
        # Predict the NEXT state after the current one
        next_predicted_sdr = self.predict_next(current_sdr, apical_context, basal_context)
        
        # Move forward
        self.prev_sdr = current_sdr
        self.prev_context = apical_context
        
        return next_predicted_sdr, error

    def process_sequence(self, sdr_sequence):
        """
        Processes an entire sequence of SDRs.
        """
        self.reset_state()
        errors = []
        predictions = []
        
        for sdr in sdr_sequence:
            pred, err = self.process_step(sdr)
            predictions.append(pred)
            # We omit the first error (0.0) as there was no prior prediction
            if self.prev_sdr is not sdr_sequence[0]:
                errors.append(err)
                
        return predictions, errors


# =============================================================================
# PATCH (V21): PredictiveMinicolumn
# -----------------------------------------------------------------------------
# `dual_planner.py` at the project root imports `PredictiveMinicolumn` from this
# module, but the class was never actually shipped in the lightweight archive,
# so the top-level demo crashed with ImportError before anything could run.
#
# This adds a small, self-contained predictive-coding minicolumn compatible
# with the way `dual_planner.HyperionLeftBrain` and `nlp_engine/optim/tree_search.py`
# use it: it exposes `generative_weights` (dim x dim) and a `view()` method
# that projects a Markov Blanket's internal state to an observation.
# =============================================================================
class PredictiveMinicolumn:
    """
    Minimal predictive-coding minicolumn used by the HyperionLeftBrain
    (see dual_planner.py). Not a spiking model — a dense generative projection
    with sparsity-controlled hidden activations, sufficient for the
    Expected-Free-Energy search protocol driven by CAINTreeSearch.
    """

    def __init__(self, in_dim: int, out_dim: int, sparsity: float = 0.1):
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.sparsity = float(sparsity)

        rng = np.random.default_rng(1729)
        # Small-scale Xavier-like init; symmetric so left/right hemispheres
        # can talk to each other via CorpusCallosum without collapsing.
        scale = 1.0 / np.sqrt(max(self.in_dim, 1))
        self.generative_weights = rng.standard_normal(
            (self.out_dim, self.in_dim)
        ).astype(np.float32) * scale
        self.recognition_weights = self.generative_weights.T.copy()

        # Precision (inverse variance) of the generative prediction.
        self.precision = np.ones(self.out_dim, dtype=np.float32)

    def _sparsify(self, x: np.ndarray) -> np.ndarray:
        """Keep the top (1 - sparsity) fraction of activations."""
        if self.sparsity <= 0.0 or x.size == 0:
            return x
        k = max(1, int(round(x.size * (1.0 - self.sparsity))))
        idx = np.argpartition(np.abs(x), -k)[-k:]
        out = np.zeros_like(x)
        out[idx] = x[idx]
        return out

    def view(self, blanket) -> np.ndarray:
        """
        Projects a Markov Blanket's internal_state into an observation.
        Used by dual_planner.HyperionOrganism to compare against a target concept.
        """
        state = getattr(blanket, "internal_state", None)
        if state is None:
            raise ValueError("blanket has no internal_state attribute")
        # W @ mu, then sparsify to mimic minicolumn sparsity.
        pred = self.generative_weights @ np.asarray(state, dtype=np.float32)
        return self._sparsify(pred)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._sparsify(self.generative_weights @ np.asarray(x, dtype=np.float32))

    def infer(self, y: np.ndarray) -> np.ndarray:
        """Recognition pass — pseudo-inverse of the generative map."""
        return self.recognition_weights @ np.asarray(y, dtype=np.float32)

    def update(self, x: np.ndarray, y_target: np.ndarray, lr: float = 0.01):
        """One step of precision-weighted Hebbian learning."""
        x = np.asarray(x, dtype=np.float32)
        y_target = np.asarray(y_target, dtype=np.float32)
        y_hat = self.generative_weights @ x
        err = y_target - y_hat
        # W += lr * (precision * err) outer x
        self.generative_weights += lr * np.outer(self.precision * err, x)
        # Update precision as a running inverse of squared error.
        self.precision = 0.99 * self.precision + 0.01 / (err * err + 1e-3)
        # Keep recognition consistent.
        self.recognition_weights = self.generative_weights.T.copy()
        return float(np.mean(err * err))
