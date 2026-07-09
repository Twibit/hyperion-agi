"""
CfCLayer — Closed-form Continuous-time RNN.

Approximates Liquid Neural Network dynamics with a closed-form (no ODE solver)
update, per arc_proposal.md Component 5:

    h(t + Δt) = A(Δt; θ) · h(t) + B(Δt; θ) · f(x(t); θ)

where A and B are smooth gating functions of elapsed time Δt and the input.
This gives:
    - O(1) per-step cost (no solver)
    - native support for irregularly-sampled inputs (pass Δt per step)
    - parameter efficiency vs LSTM at matched accuracy.

The form used here is the Hasani et al. (2022) CfC closed-form:

    g(t) = σ(W_g x + U_g h + b_g)               # time-decay gate
    h_new = -A_decay · g(t)·h
            + (1 - g(t)) · tanh(W_h x + U_h h + b_h)

with `g(t) = σ(... ) · exp(-τ · Δt)` so longer Δt → more forgetting.
"""
import numpy as np


def _sigmoid(x):
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))


class CfCLayer:
    def __init__(self, d_in: int, d_h: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_in + d_h)
        self.W_g = rng.standard_normal((d_in, d_h)) * s
        self.U_g = rng.standard_normal((d_h, d_h)) * s
        self.b_g = np.zeros(d_h)
        self.W_h = rng.standard_normal((d_in, d_h)) * s
        self.U_h = rng.standard_normal((d_h, d_h)) * s
        self.b_h = np.zeros(d_h)
        # learnable per-neuron membrane time constant τ > 0
        self._raw_tau = np.zeros(d_h)    # softplus -> τ ≈ 0.69
        self.d_h = d_h
        self.reset()

    @property
    def tau(self):
        return np.log1p(np.exp(self._raw_tau)) + 1e-3     # softplus

    def reset(self):
        self.h = np.zeros(self.d_h)

    def step(self, x_t: np.ndarray, dt: float = 1.0) -> np.ndarray:
        g_raw = x_t @ self.W_g + self.h @ self.U_g + self.b_g
        # time-modulated decay: exp(-τ·dt) shrinks gate as dt grows
        g = _sigmoid(g_raw) * np.exp(-self.tau * float(dt))
        cand = np.tanh(x_t @ self.W_h + self.h @ self.U_h + self.b_h)
        self.h = (1.0 - g) * cand + g * self.h
        return self.h

    def forward(self, x_seq: np.ndarray,
                dts: np.ndarray | float = 1.0,
                reset: bool = True) -> np.ndarray:
        if reset:
            self.reset()
        T = x_seq.shape[0]
        if np.isscalar(dts):
            dts = np.full(T, float(dts))
        out = np.empty((T, self.d_h), dtype=x_seq.dtype)
        for t in range(T):
            out[t] = self.step(x_seq[t], dts[t])
        return out
