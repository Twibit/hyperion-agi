"""
Ensemble World Model (v3) — adds epistemic uncertainty.

v2 had a single MLP forward model.  v3 keeps an *ensemble* of K small
forward models and reports the disagreement between them as an
uncertainty estimate.  This drives:

  * intrinsic exploration bonus (head into states the ensemble is unsure about)
  * safe planning (refuse imagined rollouts past the horizon where
    uncertainty exceeds threshold)

Both behaviours are necessary properties of a general learner —
without them, the agent either explores blindly or believes its own
hallucinations during dreaming.
"""
from __future__ import annotations
import numpy as np


def _gelu(x): return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


class _SingleWorldNet:
    def __init__(self, latent_dim: int, n_actions: int, hidden: int = 256, seed: int = 0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(latent_dim + n_actions)
        self.W1 = rng.standard_normal((hidden, latent_dim + n_actions)).astype(np.float32) * s
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.standard_normal((latent_dim, hidden)).astype(np.float32) / np.sqrt(hidden)
        self.b2 = np.zeros(latent_dim, dtype=np.float32)
        self.W_r = rng.standard_normal(hidden).astype(np.float32) / np.sqrt(hidden)
        self.n_actions = n_actions
        self.latent_dim = latent_dim

    def _onehot(self, a):
        v = np.zeros(self.n_actions, dtype=np.float32); v[a] = 1.0; return v

    def forward(self, z, a):
        x = np.concatenate([z, self._onehot(a)])
        h = _gelu(self.W1 @ x + self.b1)
        z_pred = self.W2 @ h + self.b2
        r_pred = float(self.W_r @ h)
        return z_pred, r_pred, h

    def learn(self, z, a, z_next, r, lr=1e-3):
        z_pred, r_pred, h = self.forward(z, a)
        eps_z = z_next - z_pred
        eps_r = r - r_pred
        self.W2 += lr * np.outer(eps_z, h).astype(np.float32)
        self.b2 += lr * eps_z
        self.W_r += lr * eps_r * h
        d_h = (h > 0).astype(np.float32) * (self.W2.T @ eps_z + self.W_r * eps_r)
        x = np.concatenate([z, self._onehot(a)])
        self.W1 += lr * np.outer(d_h, x).astype(np.float32)
        self.b1 += lr * d_h
        return float(np.mean(eps_z ** 2) + eps_r ** 2)


class EnsembleWorldModel:
    def __init__(self, latent_dim: int, n_actions: int, k: int = 4,
                 hidden: int = 256, seed: int = 0):
        self.nets = [_SingleWorldNet(latent_dim, n_actions, hidden, seed=seed + i)
                     for i in range(k)]
        self.k = k
        self.n_actions = n_actions
        self.latent_dim = latent_dim

    def forward(self, z, a) -> tuple[np.ndarray, float, float]:
        """Return mean prediction, mean reward, *epistemic uncertainty*."""
        zs, rs = [], []
        for n in self.nets:
            zp, rp, _ = n.forward(z, a)
            zs.append(zp); rs.append(rp)
        Z = np.stack(zs, axis=0)
        z_mean = Z.mean(axis=0)
        # uncertainty = mean variance across ensemble.
        uncertainty = float(Z.var(axis=0).mean())
        r_mean = float(np.mean(rs))
        return z_mean, r_mean, uncertainty

    def learn(self, z, a, z_next, r, lr=1e-3) -> float:
        # Bootstrap: each ensemble member is updated, with a random
        # subsample mask so members diverge.
        losses = []
        for i, n in enumerate(self.nets):
            if np.random.rand() < 0.8:           # 80% inclusion → ensemble diversity
                losses.append(n.learn(z, a, z_next, r, lr=lr))
        return float(np.mean(losses)) if losses else 0.0

    def rollout(self, z0, policy_fn, horizon: int,
                uncertainty_stop: float | None = None) -> list[tuple]:
        traj = []
        z = z0.copy()
        for _ in range(horizon):
            a = policy_fn(z)
            z_next, r, u = self.forward(z, a)
            traj.append((z, a, r, z_next, u))
            # Safety: stop the dream when the ensemble strongly disagrees
            # (= we're hallucinating).
            if uncertainty_stop is not None and u > uncertainty_stop:
                break
            z = z_next
        return traj
