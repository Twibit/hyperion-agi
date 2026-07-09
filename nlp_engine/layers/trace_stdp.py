import numpy as np

class ContinuousTraceSTDP:
    """
    Continuous Trace Spike-Timing-Dependent Plasticity (STDP) Layer.

    Biologically inspired long-range working memory for token sequences.
    This layer replaces standard N-gram short-sightedness with a continuous,
    biological 'electrical trace' that decays slowly over time.

    PATCH: the original class had no plasticity at all – it stored a decaying
    activity trace but never updated any synaptic weights. Real STDP must
    change W in proportion to pre/post timing. We now keep a (vocab × vocab)
    weight matrix and apply the canonical asymmetric STDP window:

        Δw_{j→i} = A+ · pre_j  if post_i fires (LTP)
                   - A- · post_i if pre_j fires after post (LTD)

    The old behaviour (just a trace vector) is preserved via `forward()` so
    existing call-sites keep working.
    """
    def __init__(self, vocab_size: int, decay_rate: float = 0.95,
                 a_plus: float = 0.005, a_minus: float = 0.0055,
                 pre_tau: float = 20.0, post_tau: float = 20.0,
                 enable_plasticity: bool = True):
        self.vocab_size = vocab_size
        self.decay_rate = decay_rate
        self.traces = np.zeros(vocab_size, dtype=np.float32)

        # --- new: real STDP machinery ---
        self.enable_plasticity = enable_plasticity
        self.W = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        self.pre_trace  = np.zeros(vocab_size, dtype=np.float32)
        self.post_trace = np.zeros(vocab_size, dtype=np.float32)
        self.a_plus, self.a_minus = float(a_plus), float(a_minus)
        self.pre_decay  = float(np.exp(-1.0 / pre_tau))
        self.post_decay = float(np.exp(-1.0 / post_tau))

    def forward(self, spike: np.ndarray) -> np.ndarray:
        """Update working-memory trace AND apply STDP if enabled."""
        # 1. Decay existing traces (forgetting mechanism)
        self.traces = self.traces * self.decay_rate
        self.traces += spike

        # 2. Real STDP weight update
        if self.enable_plasticity:
            self._stdp_update(spike)

        return self.traces

    def _stdp_update(self, spike: np.ndarray):
        s = spike.astype(np.float32)
        # Decay pre/post traces.
        self.pre_trace  *= self.pre_decay
        self.post_trace *= self.post_decay

        # When a neuron fires *now* (s>0), every recent presynaptic trace
        # contributes LTP, and conversely every recent post-trace contributes
        # LTD to inputs that fire later.
        fired = s > 0
        if fired.any():
            # LTP: pre-then-post  →  strengthen W[pre, post]
            self.W[:, fired] += self.a_plus * self.pre_trace[:, None]
            # LTD: post-then-pre  →  weaken   W[pre, post]
            self.W[fired, :] -= self.a_minus * self.post_trace[None, :]

        # Update traces *after* the rule (so simultaneous spikes contribute
        # only via the next step – anti-symmetry).
        self.pre_trace  += s
        self.post_trace += s

        # weight clipping (Brunel & Hakim style)
        np.clip(self.W, -1.0, 1.0, out=self.W)

    def predict(self, current_id: int, top_k: int = 5):
        """Use STDP-learned weights to predict the next-token distribution."""
        scores = self.W[current_id]
        idx = np.argsort(scores)[-top_k:][::-1]
        return idx, scores[idx]

    def reset(self):
        """Clear working memory (but keep learned W)."""
        self.traces.fill(0.0)
        self.pre_trace.fill(0.0)
        self.post_trace.fill(0.0)

    def get_top_traces(self, top_k: int = 5):
        indices = np.argsort(self.traces)[-top_k:][::-1]
        values = self.traces[indices]
        return values, indices
