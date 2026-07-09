"""
CorticalColumn — laminar L4 / L2-3 / L5 stack (Omega Edition).

Composition order (matches mammalian neocortex flow):

    Input
      │
      ▼  Layer 4 (sensory input)
        - DendriticNeuron (fine-grained features)
        - Mexican-hat lateral competition (winner-take-most)
      │
      ▼  Layer 2/3 (association)
        - MANARAttention (memory + local window)
        - DendriticNeuron
      │
      ▼  Layer 5 (output / motor)
        - CfCLayer (continuous-time integration)
        - DendriticNeuron with neuromodulatory gating
      │
    Output

=== Phase-2 / Omega upgrades ===========================================
1. **Multi-channel neuromodulation** — DA (Dopamine, reward salience),
   ACh (Acetylcholine, signal-to-noise), NE (Norepinephrine, arousal/gain),
   5-HT (Serotonin, patience / horizon) — each with its own decay constant
   and gating target. Replaces the single-softmax mix.
2. **Mexican-hat kernel cached at init** — was being rebuilt per call.
3. **Streaming generation API** — `step_with_cache()` for token-by-token
   inference. The CfC and DendriticNeuron layers are already stateful;
   we just expose the MANAR cache to the outside.
4. **State management** — `reset()` cleanly clears recurrent state for the
   start of a new sequence; `snapshot()` / `restore()` allow rollback after
   speculative ("thinking") passes.
"""
import numpy as np

from .dendritic_neuron import DendriticNeuron
from .manar_attention import MANARAttention
from .cfc_layer import CfCLayer


def _build_mexican_hat(d, sigma_e=1.0, sigma_i=3.0):
    idx = np.arange(d)
    diff = (idx[:, None] - idx[None, :]) ** 2
    K = np.exp(-diff / (2 * sigma_e ** 2)) - 0.5 * np.exp(-diff / (2 * sigma_i ** 2))
    # Symmetrize and rescale so it doesn't blow up activations
    return K.astype(np.float32) / (np.linalg.norm(K, ord=2) + 1e-6)


class NeuromodulatoryState:
    """Four-channel neuromodulator state with distinct decay constants.

    Roughly:
        DA (dopamine):     reward / salience — boosts L5 motor output
        ACh (acetylcholine): signal-to-noise — sharpens L4 features
        NE (norepinephrine): arousal / gain  — multiplies everything
        5-HT (serotonin):   patience          — biases toward slow timescale
    """
    __slots__ = ("da", "ach", "ne", "ht",
                 "decay_da", "decay_ach", "decay_ne", "decay_ht")

    def __init__(self, decays=(0.95, 0.80, 0.70, 0.99)):
        self.da, self.ach, self.ne, self.ht = 0.5, 0.5, 0.5, 0.5
        self.decay_da, self.decay_ach, self.decay_ne, self.decay_ht = decays

    def update(self, *, reward: float = 0.0, novelty: float = 0.0,
               arousal: float = 0.0, horizon: float = 0.0):
        self.da  = self.decay_da  * self.da  + (1 - self.decay_da)  * (0.5 + reward)
        self.ach = self.decay_ach * self.ach + (1 - self.decay_ach) * (0.5 + novelty)
        self.ne  = self.decay_ne  * self.ne  + (1 - self.decay_ne)  * (0.5 + arousal)
        self.ht  = self.decay_ht  * self.ht  + (1 - self.decay_ht)  * (0.5 + horizon)
        # clip into [0, 1]
        self.da, self.ach, self.ne, self.ht = (
            float(np.clip(self.da,  0, 1)),
            float(np.clip(self.ach, 0, 1)),
            float(np.clip(self.ne,  0, 1)),
            float(np.clip(self.ht,  0, 1)),
        )

    def gains(self):
        """Per-layer multiplicative gains (L4, L23, L5) under current modulators."""
        g_l4  = 0.5 + 0.5 * self.ach            # ACh sharpens sensory
        g_l23 = 0.5 + 0.5 * self.ne             # NE → broad cortical gain
        g_l5  = 0.5 + 0.5 * self.da             # DA → motor / output drive
        # 5-HT biases the readout toward stability (averaged across layers)
        stability = 0.5 + 0.5 * self.ht
        return g_l4, g_l23, g_l5, stability


class CorticalColumn:
    def __init__(self, d_model: int,
                 num_concepts: int = 32,
                 window: int = 16,
                 causal: bool = False,
                 n_heads: int = 1,
                 top_c: int | None = None,
                 acr_refresh: int = 64,
                 seed: int | None = None):
        self.d_model = d_model
        # Layer 4
        self.l4_neuron = DendriticNeuron(d_model, d_model, seed=seed)
        # Layer 2/3 — Genius-Edition MANAR (multi-head + k-WTA + myelin)
        self.l23_attn   = MANARAttention(d_model, num_concepts, window,
                                          causal=causal,
                                          n_heads=n_heads,
                                          top_c=top_c,
                                          acr_refresh=acr_refresh,
                                          seed=seed)
        self.l23_neuron = DendriticNeuron(d_model, d_model, seed=(seed or 0)+1)
        # Layer 5
        self.l5_cfc     = CfCLayer(d_model, d_model, seed=(seed or 0)+2)
        self.l5_neuron  = DendriticNeuron(d_model, d_model, seed=(seed or 0)+3)
        # Cached Mexican-hat kernel
        self._mex_K = _build_mexican_hat(d_model)
        # Neuromodulators (four channels)
        self.mods = NeuromodulatoryState()

    # --------------------------------------------------------------- helpers
    def _apply_mods(self, l4, l23, l5):
        g4, g23, g5, stability = self.mods.gains()
        out = g4 * l4 + g23 * l23 + g5 * l5
        # 5-HT-controlled exponential moving average against the L23 association
        # tier — high 5-HT = more "patient" / stable output.
        return stability * out + (1 - stability) * l23

    # --------------------------------------------------------------- full-seq
    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        # ---- L4: sensory + lateral competition (cached kernel)
        l4 = self.l4_neuron.forward(x_seq)
        l4 = l4 @ self._mex_K

        # ---- L2/3: memory-augmented association
        l23 = self.l23_attn.forward(l4)
        l23 = self.l23_neuron.forward(l23) + l4         # myelinated skip

        # ---- L5: continuous-time integrator + dendritic shaping
        l5 = self.l5_cfc.forward(l23)
        l5 = self.l5_neuron.forward(l5)

        return self._apply_mods(l4, l23, l5)

    # --------------------------------------------------------------- streaming
    def init_cache(self):
        return self.l23_attn.init_cache()

    def prime_cache(self, x_seq: np.ndarray, cache):
        """Run a full forward pass to prime the L23 attention cache AND
        warm up the dendritic / CfC recurrent state from the prefix."""
        self.l4_neuron.reset()
        self.l23_neuron.reset()
        self.l5_neuron.reset()
        self.l5_cfc.reset()

        # Use the streaming step per token to leave recurrent state correctly
        # set for the next step_with_cache call.
        T = x_seq.shape[0]
        out = np.empty((T, self.d_model), dtype=x_seq.dtype)
        # but also prime the MANAR cache from the full prefix in one shot:
        self.l23_attn.prime_cache(x_seq, cache)
        for t in range(T):
            out[t] = self._step(x_seq[t], cache, append_to_cache=False)
        return out

    def step_with_cache(self, x_t: np.ndarray, cache) -> np.ndarray:
        return self._step(x_t, cache, append_to_cache=True)

    def _step(self, x_t: np.ndarray, cache, *, append_to_cache: bool) -> np.ndarray:
        # L4
        l4 = self.l4_neuron.step(x_t)
        l4 = l4 @ self._mex_K
        # L23 attention (streaming) + dendrite + skip
        if append_to_cache:
            l23_attn = self.l23_attn.step_with_cache(l4, cache)
        else:
            # primed pass: K/V already in cache; we still need an output
            # — use the local + global ctx using the existing cache.
            l23_attn = self.l23_attn.step_with_cache(l4, cache)
        l23 = self.l23_neuron.step(l23_attn) + l4
        # L5
        l5 = self.l5_cfc.step(l23)
        l5 = self.l5_neuron.step(l5)
        return self._apply_mods(l4, l23, l5)

    # --------------------------------------------------------------- state
    def reset(self):
        self.l4_neuron.reset()
        self.l23_neuron.reset()
        self.l5_neuron.reset()
        self.l5_cfc.reset()

    def snapshot(self):
        """Capture all recurrent state (for speculative / thinking passes)."""
        return {
            "l4_fast": self.l4_neuron.h_fast.copy(),
            "l4_slow": self.l4_neuron.h_slow.copy(),
            "l4_ctx":  self.l4_neuron.c_ctx.copy(),
            "l23_fast": self.l23_neuron.h_fast.copy(),
            "l23_slow": self.l23_neuron.h_slow.copy(),
            "l23_ctx":  self.l23_neuron.c_ctx.copy(),
            "l5_fast":  self.l5_neuron.h_fast.copy(),
            "l5_slow":  self.l5_neuron.h_slow.copy(),
            "l5_ctx":   self.l5_neuron.c_ctx.copy(),
            "l5_cfc_h": self.l5_cfc.h.copy(),
            "mods": (self.mods.da, self.mods.ach, self.mods.ne, self.mods.ht),
        }

    def restore(self, snap):
        self.l4_neuron.h_fast = snap["l4_fast"].copy()
        self.l4_neuron.h_slow = snap["l4_slow"].copy()
        self.l4_neuron.c_ctx  = snap["l4_ctx"].copy()
        self.l23_neuron.h_fast = snap["l23_fast"].copy()
        self.l23_neuron.h_slow = snap["l23_slow"].copy()
        self.l23_neuron.c_ctx  = snap["l23_ctx"].copy()
        self.l5_neuron.h_fast  = snap["l5_fast"].copy()
        self.l5_neuron.h_slow  = snap["l5_slow"].copy()
        self.l5_neuron.c_ctx   = snap["l5_ctx"].copy()
        self.l5_cfc.h          = snap["l5_cfc_h"].copy()
        (self.mods.da, self.mods.ach, self.mods.ne, self.mods.ht) = snap["mods"]
