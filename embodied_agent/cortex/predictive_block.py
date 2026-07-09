"""
Predictive-Coding Cortical Block — V20.

Same Friston-style free-energy update as v3 (predict bottom from top,
back-propagate residual only).  V20 improvements:
  * fp16-in-RAM lazy upcast — halves active-expert RAM.
  * free_energy() scalar for the FreeEnergyMonitor.
  * last_err cache for the MetacognitiveMonitor.
  * fp64 weight init bug fixed (explicit np.float32 scalar).
"""
from __future__ import annotations
import numpy as np
import ctypes
import os
import warnings

_this_dir = os.path.dirname(os.path.abspath(__file__))
_hyperion_dir = os.path.abspath(os.path.join(_this_dir, "..", ".."))
_lib_path = os.path.join(_hyperion_dir, 'hyperion_core', 'target', 'release', 'libhyperion_core.so')

_rust_lib = None
if os.path.exists(_lib_path):
    try:
        _rust_lib = ctypes.CDLL(_lib_path)
        _rust_lib.cortex_predictive_block_step.argtypes = [
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float, ctypes.POINTER(ctypes.c_float)
        ]
    except OSError:
        _rust_lib = None


def _gelu(x: np.ndarray) -> np.ndarray:
    # Physical limit of dendritic potential (reversal potential clamp)
    x_clamped = np.clip(x, -8.0, 8.0)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_clamped + 0.044715 * x_clamped ** 3)))


def _layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean()
    sd = x.std() + eps
    return (x - mu) / sd


class PredictiveBlock:
    def __init__(self, dim_in: int, dim_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # V20 fix: explicit fp32 scalar prevents silent fp64 upcast.
        scale_out = np.float32(1.0 / np.sqrt(dim_out))
        scale_in = np.float32(1.0 / np.sqrt(dim_in))
        self.W_gen = rng.standard_normal((dim_in, dim_out)).astype(np.float32) * scale_out
        self.W_rec = rng.standard_normal((dim_out, dim_in)).astype(np.float32) * scale_in
        self.b_gen = np.zeros(dim_in, dtype=np.float32)
        self.b_rec = np.zeros(dim_out, dtype=np.float32)
        self.state = np.zeros(dim_out, dtype=np.float32)
        # Running variance of prediction error => *precision* (1/var).
        self._err_var = np.array([1.0], dtype=np.float32)
        # V20: cheap scalar caches for MetacognitiveMonitor & FE loss.
        self.last_err = 0.0
        self._fp16_in_ram = False

    # V20: lazy fp16->fp32 upcast on the matmul path only.
    def _W_gen32(self) -> np.ndarray:
        if self._fp16_in_ram and self.W_gen.dtype == np.float16:
            return self.W_gen.astype(np.float32, copy=False)
        return self.W_gen

    def _W_rec32(self) -> np.ndarray:
        if self._fp16_in_ram and self.W_rec.dtype == np.float16:
            return self.W_rec.astype(np.float32, copy=False)
        return self.W_rec

    def predict(self, top: np.ndarray) -> np.ndarray:
        return _gelu(self._W_gen32() @ top + self.b_gen)

    @property
    def precision(self) -> float:
        return 1.0 / (float(self._err_var[0]) + 1e-3)

    # V20: single-scalar free-energy for the FreeEnergyMonitor.
    def free_energy(self) -> float:
        var = float(self._err_var[0]) + 1e-3
        return float(self.last_err * self.last_err / var
                     + 0.5 * np.log(var))

    def step(self, bottom: np.ndarray, top: np.ndarray | None = None,
             lr: float = 3e-4) -> tuple[np.ndarray, float]:
        bottom = np.ascontiguousarray(bottom, dtype=np.float32)

        # Rust path: only when weights are fp32 (skip if fp16-in-RAM).
        if (_rust_lib is not None
                and not self._fp16_in_ram
                and self.W_gen is not None
                and self.W_gen.dtype == np.float32):
            if top is not None:
                top = np.ascontiguousarray(top, dtype=np.float32)
                top_ptr = top.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            else:
                top_ptr = None
            out_err = np.array([0.0], dtype=np.float32)
            _rust_lib.cortex_predictive_block_step(
                self.W_gen.shape[0],
                bottom.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                top_ptr,
                self.W_gen.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.W_rec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.b_gen.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.b_rec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self._err_var.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_float(lr),
                out_err.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
            self.last_err = float(out_err[0])
            return self.state.copy(), self.last_err

        ctx = top if top is not None else self.state
        ctx = np.ascontiguousarray(ctx, dtype=np.float32)
        pred = self.predict(ctx)
        eps = bottom - pred
        err = float(np.mean(eps ** 2))

        # Running variance of the residual.
        self._err_var[0] = 0.99 * self._err_var[0] + 0.01 * err

        delta = _gelu(self._W_rec32() @ eps + self.b_rec)
        delta = _layernorm(delta)
        self.state = 0.9 * self.state + 0.1 * delta

        # Friston's Bounded Precision (gain saturates neuromodulator receptor capacity)
        raw_precision = 1.0 / (float(self._err_var[0]) + 1e-2)
        gain = lr * 2.0 * np.tanh(raw_precision / 10.0)

        d_gen = (gain * np.outer(eps, ctx)).astype(np.float32)
        d_rec = (gain * np.outer(delta, eps)).astype(np.float32)

        # L2 Weight Decay (metabolic maintenance cost of synapses)
        decay = 0.9995
        if self._fp16_in_ram:
            # Upcast-add-downcast: peak fp32 is only the transient delta.
            self.W_gen = (self.W_gen.astype(np.float32, copy=False) * decay
                          + d_gen).astype(np.float16)
            self.W_rec = (self.W_rec.astype(np.float32, copy=False) * decay
                          + d_rec).astype(np.float16)
        else:
            self.W_gen = self.W_gen * decay + d_gen
            self.W_rec = self.W_rec * decay + d_rec

        # Turrigiano Synaptic Scaling: Keep weight norms bounded
        norm_gen = np.linalg.norm(self.W_gen, axis=1, keepdims=True) + 1e-8
        self.W_gen = self.W_gen / np.maximum(1.0, norm_gen)
        
        norm_rec = np.linalg.norm(self.W_rec, axis=1, keepdims=True) + 1e-8
        self.W_rec = self.W_rec / np.maximum(1.0, norm_rec)

        self.b_gen = self.b_gen * decay + gain * eps
        self.b_rec = self.b_rec * decay + gain * delta

        self.last_err = err
        return self.state.copy(), err

    # ---- Paging (V20: keep fp16 in RAM after wake) ----
    def hibernate(self, pager):
        """Quantize to float16 and write to disk, then free the memory."""
        if not hasattr(self, "id") or self.id is None:
            warnings.warn(
                "[PredictiveBlock.hibernate] block has no .id; refusing to "
                "page. Neurogenesis code must set expert.id before appending.",
                RuntimeWarning, stacklevel=2,
            )
            return
        if self.W_gen is None:
            return  # already hibernating
        pager.hibernate_expert(self.id, self.W_gen, self.W_rec)
        self.W_gen = None
        self.W_rec = None
        self._fp16_in_ram = False

    def wake(self, pager):
        """V20: keep fp16 dtype in RAM. Do NOT upcast to fp32."""
        if self.W_gen is not None:
            return  # already awake
        if not hasattr(self, "id") or self.id is None:
            warnings.warn(
                "[PredictiveBlock.wake] block has no .id; cannot restore.",
                RuntimeWarning, stacklevel=2,
            )
            return
        w_gen, w_rec = pager.wake_expert(self.id)
        target_dtype = np.float16 if w_gen.dtype == np.float16 else np.float32
        self.W_gen = np.ascontiguousarray(w_gen, dtype=target_dtype)
        self.W_rec = np.ascontiguousarray(w_rec, dtype=target_dtype)
        self._fp16_in_ram = (target_dtype == np.float16)


class CortexStack:
    """Dense (non-MoE) stack — kept for parity with v2 and small models."""

    def __init__(self, latent_dim: int, n_layers: int, seed: int = 0):
        self.layers = [PredictiveBlock(latent_dim, latent_dim, seed=seed + i)
                       for i in range(n_layers)]

    def forward(self, sensory_z: np.ndarray, lr: float = 3e-4) -> tuple[np.ndarray, float]:
        x = sensory_z
        total_err = 0.0
        for layer in self.layers:
            x, err = layer.step(x, top=None, lr=lr)
            total_err += err
        for i in range(len(self.layers) - 2, -1, -1):
            top_state = self.layers[i + 1].state
            self.layers[i].step(self.layers[i].state, top=top_state, lr=lr * 0.5)
        return self.layers[-1].state.copy(), total_err / max(1, len(self.layers))

    def imagine(self, z_top: np.ndarray) -> np.ndarray:
        x = z_top
        for layer in reversed(self.layers):
            x = layer.predict(x)
        return x

    def precision_profile(self) -> list[float]:
        """Used by the v3 thalamus for inverse-variance attention."""
        return [l.precision for l in self.layers]
