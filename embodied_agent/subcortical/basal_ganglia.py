"""
Basal-ganglia — linear actor-critic, TD(0) updates gated by dopamine.

v3 adds an "alignment penalty" hook: actions whose imagined outcomes
contain extreme negative valence-on-others (provided by the safety layer)
are penalised at policy time.  This is the agentic-safety analogue of
the prefrontal-orbital inhibition of impulsive behaviour.
"""
from __future__ import annotations
import numpy as np


class BasalGanglia:
    def __init__(self, latent_dim: int, n_actions: int, lr: float = 5e-4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W_actor = rng.standard_normal((n_actions, latent_dim)).astype(np.float32) / np.sqrt(latent_dim)
        self.W_critic = rng.standard_normal(latent_dim).astype(np.float32) / np.sqrt(latent_dim)
        self.lr = lr
        self.n_actions = n_actions

    def value(self, z): return float(self.W_critic @ z)

    def policy(self, z, temp: float = 1.0, alignment_penalty: np.ndarray | None = None):
        logits = self.W_actor @ z
        logits = np.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        if alignment_penalty is not None:
            logits = logits - np.nan_to_num(alignment_penalty)
        logits = np.clip(logits / max(temp, 1e-6), -20, 20)
        logits = logits - logits.max()
        p = np.exp(logits)
        p_sum = p.sum()
        if p_sum == 0 or np.isnan(p_sum):
            p = np.ones_like(p) / len(p)
        else:
            p = p / p_sum
        a = int(np.random.choice(self.n_actions, p=p))
        return a, p

    def learn_td(self, z, a, r, z_next, done: bool,
                 dopamine_gain: float = 1.0, discount: float = 0.95) -> float:
        v = self.value(z)
        v_next = 0.0 if done else self.value(z_next)
        td = r + discount * v_next - v
        # Critic update.
        self.W_critic += self.lr * dopamine_gain * td * z
        # Actor: REINFORCE-style with TD as advantage.
        logits = self.W_actor @ z
        p = np.exp(logits - logits.max()); p = p / p.sum()
        grad = -p.copy(); grad[a] += 1.0
        self.W_actor += (self.lr * dopamine_gain * td) * np.outer(grad, z).astype(np.float32)
        return float(td)
