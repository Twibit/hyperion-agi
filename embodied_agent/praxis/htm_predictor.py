"""
Hierarchical Temporal Memory (HTM) Sequence Predictor — V11 (Aurora-Praxis).
=============================================================================

Biological brains do not just learn instantaneous state-action mappings. The
neocortex (specifically the Prefrontal Cortex) learns the *temporal sequence*
of patterns, enabling prediction, planning, and anomaly detection over long
horizons.

This HTM module provides the foundational sequence memory. It takes sparse
distributed representations (SDRs) and uses a temporal context matrix to
predict the *next* SDR in a sequence. This provides the top-down bias required
by the Prefrontal Cortex to maintain long-term goal vectors.
"""
from __future__ import annotations
import numpy as np
from typing import Optional

import ctypes

# PATCHED: gracefully fall back to a pure-numpy htm_step when the Rust core
# is unavailable (missing .so, wrong architecture, or HYPERION_FORCE_PY=1).
try:
    from ..cortex.snn_engine import _rust_core, USING_RUST_CORE
    if _rust_core is not None:
        _rust_core.htm_step.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.c_bool,
        ]
        _rust_core.htm_step.restype = None
except Exception:
    _rust_core = None
    USING_RUST_CORE = False


def _htm_step_py(latent_dim, current_sdr, prev_sdr, transitions,
                 predictions, lr, learn):
    """Pure-numpy mirror of hyperion_core::htm_step (see lib.rs)."""
    trans = transitions.reshape(latent_dim, latent_dim)
    decay_factor = 1.0 - (lr * 0.1)
    curr_active = current_sdr > 0
    prev_active = prev_sdr > 0
    if learn and curr_active.any() and prev_active.any():
        # LTP on (prev_active -> curr_active) edges; homeostatic decay on
        # all outgoing edges from previously active rows.
        trans[prev_active] *= decay_factor
        idx_prev = np.where(prev_active)[0]
        idx_curr = np.where(curr_active)[0]
        trans[np.ix_(idx_prev, idx_curr)] += lr
    # Predict: sum outgoing rows for currently-active columns.
    predictions[:] = 0.0
    if curr_active.any():
        predictions[:] = trans[curr_active].sum(axis=0)

class HTMSequenceMemory:
    """
    A biologically inspired sequence memory based on Numenta's HTM theory.
    Tracks transitions between active columns to predict future states.
    Now backed by Rust for V14 Polyglot speed!
    """
    def __init__(self, latent_dim: int, sparsity: float = 0.02, learning_rate: float = 0.1, seed: int = 1729):
        self.latent_dim = latent_dim
        self.sparsity = sparsity
        self.lr = float(learning_rate)
        self.rng = np.random.default_rng(seed)
        
        # Temporal memory matrix: [latent_dim, latent_dim]
        # Needs to be a flat 1D array for Rust C-ABI
        self.transitions = np.zeros((latent_dim * latent_dim,), dtype=np.float32)
        
        self.prev_sdr: Optional[np.ndarray] = None
        
    def step(self, current_sdr: np.ndarray, learn: bool = True) -> np.ndarray:
        """
        Process the current SDR, update sequence memory, and predict the next SDR
        using the blisteringly fast Rust physics engine.
        """
        if current_sdr.dtype != np.uint8:
            current_sdr = (current_sdr > 0.0).astype(np.uint8)
            
        assert current_sdr.dtype == np.uint8 and current_sdr.shape == (self.latent_dim,), "SDR must be uint8 1D array"
        
        if self.prev_sdr is None:
            self.prev_sdr = np.zeros(self.latent_dim, dtype=np.uint8)
            
        prediction_scores = np.zeros(self.latent_dim, dtype=np.float32)
        
        # C-ABI Pointers
        curr_ptr = current_sdr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        prev_ptr = self.prev_sdr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        trans_ptr = self.transitions.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pred_ptr = prediction_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Execute in Rust if available, otherwise numpy.
        if _rust_core is not None:
            _rust_core.htm_step(
                self.latent_dim,
                curr_ptr, prev_ptr, trans_ptr, pred_ptr,
                self.lr, learn,
            )
        else:
            _htm_step_py(
                self.latent_dim,
                current_sdr, self.prev_sdr,
                self.transitions, prediction_scores,
                self.lr, learn,
            )
        
        self.prev_sdr = current_sdr.copy()
            
        # Sparsify prediction (keep top-k)
        k = max(1, int(self.latent_dim * self.sparsity))
        predicted_sdr = np.zeros_like(prediction_scores)
        
        if np.any(prediction_scores > 0):
            # Find the top k elements using argpartition for O(N) performance
            top_k_idx = np.argpartition(prediction_scores, -k)[-k:]
            predicted_sdr[top_k_idx] = prediction_scores[top_k_idx]
            
            # Normalize
            norm = np.linalg.norm(predicted_sdr)
            if norm > 0:
                predicted_sdr /= norm
                
        return predicted_sdr

    def get_anomaly_score(self, current_sdr: np.ndarray, predicted_sdr: np.ndarray) -> float:
        """
        Calculate how surprising the current SDR is, given the previous prediction.
        Score is 0.0 for perfect prediction, 1.0 for completely unexpected.
        """
        if np.sum(predicted_sdr) == 0:
            return 1.0 # Cannot predict, max surprise
            
        # Cosine distance
        norm_c = np.linalg.norm(current_sdr)
        norm_p = np.linalg.norm(predicted_sdr)
        
        if norm_c == 0 or norm_p == 0:
            return 1.0
            
        cos_sim = np.dot(current_sdr, predicted_sdr) / (norm_c * norm_p)
        return float(np.clip(1.0 - cos_sim, 0.0, 1.0))
