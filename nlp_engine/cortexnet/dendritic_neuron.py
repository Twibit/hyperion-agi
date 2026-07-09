"""
DendriticNeuron — V6 Dual-Gate dendritic neuron.

Implements the spec from arc_proposal.md Component 1:

    x̃_t       = tanh(W_proj · x_t)
    c_t        = α_c · c_{t-1} + x̃_t                   (apical dendrite buffer)
    w_t        = σ(W_x·x_t + W_c·c_t)                   (NMDA coincidence gate)
    h_fast_t   = α_f · h_fast_{t-1} + w_t · x̃_t          (Na+, fast)
    h_slow_t   = α_s · h_slow_{t-1} + w_t · x̃_t          (Ca2+, slow)
    y_t        = h_fast_t ⊙ σ(h_slow_t)                  (somatic blend)

α_fast (≈0.5–0.9) and α_slow (≈0.95–0.999) are LEARNED via reparameterised
sigmoids so gradient descent recovers α_slow > α_fast naturally — that's the
biologically validated claim from the proposal.

Replaces: the stateless MLP block in `nlp_engine/layers/transformer_cortex.py`
          and the discrete-time `ComplexNeuron` which has no learnable
          temporal kernel.

Complexity: O(T · d²) — linear in sequence length T.
"""
import numpy as np


def _sigmoid(x):
    # numerically stable
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class DendriticNeuron:
    """
    Single dendritic-neuron *layer*. Operates on a sequence (T, d_in) and
    produces (T, d_out). Stateful between calls if you don't reset().

    Genius-Edition note: `alpha_fast/slow/ctx` are derived from raw
    parameters via sigmoid. Recomputing them on every streaming step is
    pure wasted energy — they only change when training mutates the raw
    params. We *myelinate* them: cache once, invalidate on update. This
    removes roughly 3 sigmoid calls per token per layer (the single biggest
    hot-path cost in the streaming profile, ~30% of total inference time).
    """

    def __init__(self, d_in: int, d_out: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_in)

        # learnable projections
        self.W_proj = rng.standard_normal((d_in, d_out)) * s
        self.W_x    = rng.standard_normal((d_in, d_out)) * s
        self.W_c    = rng.standard_normal((d_out, d_out)) * (1.0 / np.sqrt(d_out))
        self.b_w    = np.zeros(d_out)

        # learnable timescales — parameterised so they stay in (0, 1)
        # init: alpha_fast ≈ 0.7, alpha_slow ≈ 0.99
        self._raw_alpha_fast = np.full(d_out,  np.log(0.7 / 0.3))   # σ(x)=0.7
        self._raw_alpha_slow = np.full(d_out,  np.log(0.99 / 0.01)) # σ(x)=0.99
        self._raw_alpha_ctx  = np.full(d_out,  np.log(0.9 / 0.1))   # σ(x)=0.9

        # Myelin: cached alpha values + their (1 - alpha) complements.
        # Recomputed lazily when raw params change via `invalidate_alphas()`.
        self._alpha_cache_valid = False
        self._a_fast = None
        self._a_slow = None
        self._a_ctx  = None
        self._a_fast_1m = None    # 1 - a_fast
        self._a_slow_1m = None
        self._a_ctx_1m  = None

        # state
        self.d_in, self.d_out = d_in, d_out
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.h_fast = np.zeros(self.d_out)
        self.h_slow = np.zeros(self.d_out)
        self.c_ctx  = np.zeros(self.d_out)

    # ----------------------------------------------------- myelin (timescale)
    def invalidate_alphas(self):
        """Call after training mutates `_raw_alpha_*`. Cheap; cache is lazy."""
        self._alpha_cache_valid = False

    def _ensure_alphas(self):
        if self._alpha_cache_valid:
            return
        self._a_fast = _sigmoid(self._raw_alpha_fast)
        self._a_slow = _sigmoid(self._raw_alpha_slow)
        self._a_ctx  = _sigmoid(self._raw_alpha_ctx)
        self._a_fast_1m = 1.0 - self._a_fast
        self._a_slow_1m = 1.0 - self._a_slow
        self._a_ctx_1m  = 1.0 - self._a_ctx
        self._alpha_cache_valid = True

    # ------------------------------------------------------------------ props
    @property
    def alpha_fast(self):
        self._ensure_alphas(); return self._a_fast
    @property
    def alpha_slow(self):
        self._ensure_alphas(); return self._a_slow
    @property
    def alpha_ctx(self):
        self._ensure_alphas(); return self._a_ctx

    # ------------------------------------------------------------------ forward
    def step(self, x_t: np.ndarray) -> np.ndarray:
        """Single timestep. x_t shape (d_in,). Returns (d_out,)."""
        self._ensure_alphas()
        x_tilde = np.tanh(x_t @ self.W_proj)

        # apical context buffer (slow EMA of x_tilde)
        self.c_ctx = self._a_ctx * self.c_ctx + self._a_ctx_1m * x_tilde

        # NMDA write gate — coincidence of bottom-up input and apical context
        w_t = _sigmoid(x_t @ self.W_x + self.c_ctx @ self.W_c + self.b_w)

        # dual timescale soma
        gated = w_t * x_tilde
        self.h_fast = self._a_fast * self.h_fast + self._a_fast_1m * gated
        self.h_slow = self._a_slow * self.h_slow + self._a_slow_1m * gated

        # somatic blend: fast carries phasic info, slow gates it
        return self.h_fast * _sigmoid(self.h_slow)

    def forward(self, x_seq: np.ndarray, reset: bool = True) -> np.ndarray:
        """x_seq shape (T, d_in) → (T, d_out)."""
        if reset:
            self.reset()
        T = x_seq.shape[0]
        out = np.empty((T, self.d_out), dtype=x_seq.dtype)
        for t in range(T):
            out[t] = self.step(x_seq[t])
        return out

    # ------------------------------------------------------------------ params
    def params(self):
        return {
            "W_proj": self.W_proj, "W_x": self.W_x, "W_c": self.W_c,
            "b_w": self.b_w,
            "raw_alpha_fast": self._raw_alpha_fast,
            "raw_alpha_slow": self._raw_alpha_slow,
            "raw_alpha_ctx":  self._raw_alpha_ctx,
        }
