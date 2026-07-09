"""
Cerebellum — fast adaptive forward/inverse predictor (v6).
==========================================================

A small linear forward model + linear inverse model trained online by
LMS.  Provides:

  * `predict(z, a)` — what is z' likely to be?
  * `inverse(z, z_target)` — what action would best take us there?
  * `error(z, a, z_next)` — supervised LMS update, returns scalar err

This is the AI analogue of the cerebellum's supervised internal model
that lets you reach for a cup without overshooting.  Crucially, it runs
*in parallel* with the slower cortical world model and gives an early
"motor smoothness" signal to the basal ganglia.
"""
from __future__ import annotations
import numpy as np


class Cerebellum:
    def __init__(self, latent_dim: int, n_actions: int,
                 lr: float = 5e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.d = int(latent_dim)
        self.na = int(n_actions)
        # action embedding (random fixed)
        self.A = (rng.standard_normal((n_actions, latent_dim)).astype(np.float32)
                  / np.sqrt(latent_dim))
        # forward: z' ≈ W_f @ [z; a_emb]
        self.W_f = (rng.standard_normal((latent_dim, 2 * latent_dim)).astype(np.float32)
                    / np.sqrt(2 * latent_dim))
        # inverse: a_emb ≈ W_i @ [z; z'_target]
        self.W_i = (rng.standard_normal((latent_dim, 2 * latent_dim)).astype(np.float32)
                    / np.sqrt(2 * latent_dim))
        self.lr = float(lr)
        self.err_ema = 0.5
        self.skill = 0.0
        self.steps = 0

    def _emb(self, a: int) -> np.ndarray:
        return self.A[int(a) % self.na]

    def predict(self, z: np.ndarray, a: int) -> np.ndarray:
        x = np.concatenate([z, self._emb(a)]).astype(np.float32)
        return (self.W_f @ x).astype(np.float32)

    def inverse(self, z: np.ndarray, z_target: np.ndarray) -> int:
        x = np.concatenate([z, z_target]).astype(np.float32)
        a_hat = self.W_i @ x
        # nearest action embedding
        return int(np.argmax(self.A @ a_hat))

    def learn(self, z: np.ndarray, a: int, z_next: np.ndarray) -> float:
        x = np.concatenate([z, self._emb(a)]).astype(np.float32)
        z_pred = self.W_f @ x
        e = (z_next - z_pred).astype(np.float32)
        self.W_f += self.lr * np.outer(e, x).astype(np.float32)
        # inverse update — train W_i to recover the action embedding
        xi = np.concatenate([z, z_next]).astype(np.float32)
        a_emb_pred = self.W_i @ xi
        a_emb_true = self._emb(a)
        e_i = a_emb_true - a_emb_pred
        self.W_i += self.lr * np.outer(e_i, xi).astype(np.float32)
        err = float(np.linalg.norm(e))
        self.err_ema = 0.99 * self.err_ema + 0.01 * err
        self.skill = max(0.0, 1.0 - self.err_ema)
        self.steps += 1
        return err

    def stats(self) -> dict:
        return dict(err_ema=self.err_ema, skill=self.skill, steps=self.steps)
