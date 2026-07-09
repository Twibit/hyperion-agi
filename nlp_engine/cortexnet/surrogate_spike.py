"""
SurrogateSpikeLayer — trainable LIF spiking layer with surrogate gradient.

Replaces the un-trainable `EnergyEfficientSNNLayer` (random int16 weights that
never learn). Adds:

    1. Float weights you can actually train.
    2. Surrogate-gradient firing function (fast-sigmoid) so the layer is
       compatible with downstream learning rules (FTP, BP, etc.).
    3. Delta-sigma temporal sparsity: only the *change* in spike pattern
       since last step is emitted on the bus, on top of spatial sparsity.

Mathematical form (LIF, discrete):

    V_t = β · V_{t-1} + I_t           (I_t = W·s_{t-1})
    s_t = H(V_t - θ)                  (Heaviside fire)
    V_t ← V_t - θ · s_t               (soft reset)

Surrogate gradient (fast-sigmoid, Zenke & Ganguli 2018):

    ∂s/∂V ≈ 1 / (1 + |V - θ| · slope)²
"""
import numpy as np


class SurrogateSpikeLayer:
    def __init__(self, d_in: int, d_out: int,
                 beta: float = 0.9, threshold: float = 1.0,
                 surrogate_slope: float = 25.0, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((d_in, d_out)) * (1.0 / np.sqrt(d_in))
        self.b = np.zeros(d_out)
        self.beta = float(beta)
        self.theta = float(threshold)
        self.slope = float(surrogate_slope)

        self.d_in, self.d_out = d_in, d_out
        self.reset()

    def reset(self):
        self.V = np.zeros(self.d_out)
        self._prev_spike = np.zeros(self.d_out, dtype=np.float32)

    def step(self, x_t: np.ndarray, delta_sigma: bool = True):
        """Single LIF step. Returns (spikes, surrogate_grad)."""
        I = x_t @ self.W + self.b
        self.V = self.beta * self.V + I
        s = (self.V >= self.theta).astype(np.float32)
        # surrogate derivative wrt V (used by external trainers)
        sg = 1.0 / (1.0 + np.abs(self.V - self.theta) * self.slope) ** 2
        # soft reset
        self.V = self.V - self.theta * s

        if delta_sigma:
            # only emit the *change* on the wire (temporal sparsity)
            out = s - self._prev_spike
            self._prev_spike = s
            return out, sg
        return s, sg

    def forward(self, x_seq: np.ndarray, reset: bool = True, delta_sigma: bool = True):
        if reset:
            self.reset()
        T = x_seq.shape[0]
        S = np.empty((T, self.d_out), dtype=np.float32)
        G = np.empty((T, self.d_out), dtype=np.float32)
        for t in range(T):
            s, g = self.step(x_seq[t], delta_sigma=delta_sigma)
            S[t], G[t] = s, g
        return S, G

    def spatial_sparsity(self, spikes: np.ndarray) -> float:
        return float(np.mean(spikes == 0.0))
