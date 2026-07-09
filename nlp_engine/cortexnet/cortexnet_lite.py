"""
CortexNetLite — minimal end-to-end model wiring the new modules together.

This is Phase 1 of the implementation roadmap in arc_proposal.md:

    Input  → SparseEmbed
           → CorticalColumn × N
           → OutputHead
    Train  → FTPTrainer (no backprop)

Drop-in replacement for the `TransformerBlock` stack in
`nlp_engine/layers/transformer_cortex.py`.
"""
import numpy as np

from .cortical_column import CorticalColumn
from .complement_pruner import ComplementPruner
from .ftp_trainer import FTPTrainer


class CortexNetLite:
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 128,
                 n_columns: int = 4,
                 num_concepts: int = 32,
                 window: int = 16,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        # Sparse token embedding (k-WTA on output gives sparse codes downstream)
        self.E = rng.standard_normal((vocab_size, d_model)) * (1.0 / np.sqrt(d_model))
        self.columns = [
            CorticalColumn(d_model, num_concepts, window, seed=seed + 10*i)
            for i in range(n_columns)
        ]
        self.W_out = rng.standard_normal((d_model, vocab_size)) * (1.0 / np.sqrt(d_model))

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pruner = ComplementPruner()

    # -------------------------------------------------------------- forward
    def embed(self, ids: np.ndarray) -> np.ndarray:
        return self.E[ids]                              # (T, d)

    def forward(self, ids: np.ndarray) -> np.ndarray:
        h = self.embed(ids)
        for col in self.columns:
            h = col.forward(h) + h                      # residual
        return h @ self.W_out                           # (T, V) logits

    def predict(self, ids: np.ndarray) -> np.ndarray:
        logits = self.forward(ids)
        x = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    # -------------------------------------------------------------- prune
    def prune_step(self):
        """One C1q/C3 cycle across all learnable dense weights."""
        for col in self.columns:
            for layer in (col.l23_attn,):
                for name in ("W_q", "W_k", "W_v", "W_o"):
                    W = getattr(layer, name)
                    self.pruner.observe(W)
                    new_W, _ = self.pruner.step(W)
                    setattr(layer, name, new_W)
        self.pruner.observe(self.W_out)
        self.W_out, _ = self.pruner.step(self.W_out)

    # -------------------------------------------------------------- info
    def info(self):
        n_params = (self.E.size + self.W_out.size +
                    sum(c.l23_attn.W_q.size * 4 for c in self.columns))
        return {
            "vocab": self.vocab_size,
            "d_model": self.d_model,
            "columns": len(self.columns),
            "approx_params": int(n_params),
        }


def build_ftp_head(d_model: int, vocab_size: int, seed: int = 0) -> FTPTrainer:
    """Helper: standalone FTP-trained classification head that can be plugged
    on top of CortexNetLite for end-to-end training without backprop."""
    return FTPTrainer([d_model, d_model, vocab_size],
                      activations=["tanh", "id"], seed=seed)
