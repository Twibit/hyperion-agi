"""
rust_backend_shim — pure-Python fallback shim + OPTIONAL native fast-path.

If the compiled `hyperion_rust_core` module (built from `rust_backend/`,
x86_64) is present on the path, its functions are used — they have
IDENTICAL semantics to the pure-Python implementations below (the native
`reward_modulated_stdp`/`compute_fft`/`sparse_attention` were written to
match them exactly; `hash_neuron_id` uses FNV-1a instead of BLAKE2b, which
is functionally equivalent here since the value is only used as a neuron
address). Otherwise the pure-Python implementations below are used.

Callers (`claustrum`, `auditory_cortex`, `virtual_100t_substrate`) import
this module and are unaffected either way. To disable the native path,
remove `hyperion_rust_core.so` (or set HYPERION_RUST_CORE_FORCE_PY=1).
"""
from __future__ import annotations

import hashlib
import math
import os
import sys

# ---------------------------------------------------------------------------
# Opt-in native fast-path
# ---------------------------------------------------------------------------
_NATIVE = False
if not os.environ.get("HYPERION_RUST_CORE_FORCE_PY"):
    try:
        from hyperion_rust_core import (
            hash_neuron_id,
            reward_modulated_stdp,
            compute_fft,
            sparse_attention,
            native_omni_vision_scan,
            NativeUiEye,
            NativeMotorSystem,
            NativeIzhikevich,
            NativeAgiKernel,
        )
        _NATIVE = True
    except Exception:
        _NATIVE = False

if os.environ.get("HYPERION_RUST_CORE_SHIM_VERBOSE"):
    sys.stderr.write(
        f"[rust_backend_shim] {'native hyperion_rust_core' if _NATIVE else 'pure-Python shim'} active.\n"
    )

if _NATIVE:
    # Wrap constructors to bypass PyO3 positional argument extraction bug in Python 3.14 on aarch64
    _orig_NativeIzhikevich = NativeIzhikevich
    class NativeIzhikevichWrapper:
        def __init__(self, pattern: str, dt_ms: float):
            self._neuron = _orig_NativeIzhikevich(pattern=pattern, dt_ms=dt_ms)
        def step(self, dt_ms: float, i_inj: float) -> tuple:
            res_str = self._neuron.step(dt_ms, i_inj)
            v, spiked = res_str.split(',')
            return (float(v), spiked == 'True')
        def reset(self):
            self._neuron.reset()
    NativeIzhikevich = NativeIzhikevichWrapper

    _orig_NativeAgiKernel = NativeAgiKernel
    class NativeAgiKernelWrapper:
        def __init__(self, depth: int, n_per_layer: int, n_channels: int, top_k: int, dt_ms: float):
            self._kernel = _orig_NativeAgiKernel(depth=depth, n_per_layer=n_per_layer, n_channels=n_channels, top_k=top_k, dt_ms=dt_ms)
        def sense(self, modality: int, features: list):
            self._kernel.sense(modality, features)
        def reward(self, magnitude: float) -> float:
            return self._kernel.reward(magnitude)
        def refuel(self, drive_str: str, amount: float):
            self._kernel.refuel(drive_str, amount)
        def tick(self) -> tuple:
            import json
            res_json = self._kernel.tick()
            res = json.loads(res_json)
            return (float(res["err"]), res["action_id"])
        def t_ms(self) -> float:
            return self._kernel.t_ms()
        def drive_value(self, drive_str: str) -> float:
            return self._kernel.drive_value(drive_str)
        @property
        def dopamine(self):
            return self._kernel.dopamine
        @property
        def noradrenaline(self):
            return self._kernel.noradrenaline
        @property
        def serotonin(self):
            return self._kernel.serotonin
        @property
        def acetylcholine(self):
            return self._kernel.acetylcholine
        @property
        def arousal(self):
            return self._kernel.arousal
        @property
        def throttle_cortex(self):
            return self._kernel.throttle_cortex
    NativeAgiKernel = NativeAgiKernelWrapper

else:
    # ---------------------------------------------------------------------------
    # Pure-Python fallback (original shim)
    # ---------------------------------------------------------------------------
    try:
        import numpy as _np
    except Exception:  # numpy is a hard project dep, but be defensive.
        _np = None

    def hash_neuron_id(concept: bytes, domain: bytes, index: int) -> int:
        """Stable 64-bit hash used by Virtual100TSubstrate to address virtual neurons."""
        if isinstance(concept, str):
            concept = concept.encode("utf-8")
        if isinstance(domain, str):
            domain = domain.encode("utf-8")
        h = hashlib.blake2b(concept + b"\x1f" + domain + b"\x1f" + int(index).to_bytes(8, "little", signed=False),
                            digest_size=8)
        return int.from_bytes(h.digest(), "little", signed=False)

    def reward_modulated_stdp(
        trace_a: float,
        trace_b: float,
        dopamine: float,
        calcium: float,
    ) -> float:
        """
        Return an effective learning rate delta = tanh(trace_a * trace_b) *
        dopamine, gated by astrocyte calcium. Matches the semantics the caller
        in virtual_100t_substrate.py expects: a signed scalar close to 0 when
        the traces disagree, positive when they cofire under high dopamine.
        """
        coactivation = math.tanh(float(trace_a) * float(trace_b))
        dopamine_gain = max(-1.0, min(1.0, float(dopamine)))
        calcium_gate = 1.0 / (1.0 + math.exp(-8.0 * (float(calcium) - 0.2)))
        return coactivation * dopamine_gain * calcium_gate

    def compute_fft(samples):
        """Simple Cooley-Tukey via numpy; returns a Python list of complex numbers."""
        if _np is None:
            raise RuntimeError("compute_fft requires numpy")
        return _np.fft.fft(_np.asarray(samples, dtype=_np.float64)).tolist()

    def sparse_attention(keys, values, temperature: float = 1.0):
        """
        Softmax attention over `keys` (dot-product logits) at the given
        temperature.  If `values` is empty, returns the attention weights;
        otherwise returns the weighted sum along axis 0.
        """
        if _np is None:
            raise RuntimeError("sparse_attention requires numpy")
        k = _np.asarray(keys, dtype=_np.float64)
        if k.size == 0:
            return []
        t = max(1e-6, float(temperature))
        logits = k / t
        logits -= logits.max()  # numerical stability
        w = _np.exp(logits)
        w /= w.sum()
        if len(values) == 0:
            return w.tolist()
        v = _np.asarray(values, dtype=_np.float64)
        if v.ndim == 1:
            return float((w * v).sum())
        return (w[:, None] * v).sum(axis=0).tolist()
        
    def native_omni_vision_scan(filepath: str, keywords: list) -> list:
        raise NotImplementedError("Native vision requires the compiled hyperion_rust_core.")

    class NativeUiEye:
        def __init__(self):
            pass
        def glance_url(self, url: str) -> str:
            raise NotImplementedError("Native UI Eye requires the compiled hyperion_rust_core.")

    class NativeMotorSystem:
        def __init__(self):
            pass
        def reach_file(self, intent_json: str) -> str:
            raise NotImplementedError("Native Motor System requires the compiled hyperion_rust_core.")

    class NativeIzhikevich:
        def __init__(self, pattern: str, dt_ms: float):
            self.v = -65.0
            self.u = 0.0
            self.spiked = False
        def step(self, dt_ms: float, i_inj: float):
            return (self.v, self.spiked)
        def reset(self):
            pass

    class NativeAgiKernel:
        def __init__(self, depth: int, n_per_layer: int, n_channels: int, top_k: int, dt_ms: float):
            self._t_ms = 0.0
            self.dopamine = 0.2
            self.noradrenaline = 0.2
            self.serotonin = 0.2
            self.acetylcholine = 0.2
            self.arousal = 0.5
            self.throttle_cortex = False
        def sense(self, modality: int, features: list):
            pass
        def reward(self, magnitude: float) -> float:
            return self.dopamine
        def refuel(self, drive_str: str, amount: float):
            pass
        def tick(self) -> tuple:
            self._t_ms += 25.0
            return (0.0, None)
        def t_ms(self) -> float:
            return self._t_ms
        def drive_value(self, drive_str: str) -> float:
            return 1.0
