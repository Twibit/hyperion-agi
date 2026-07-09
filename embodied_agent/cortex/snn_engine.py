"""
SNN engine + memristor layer.

CHANGES (patched):
    The original file *hard-imported* a Rust-compiled `.so` at module load
    time and raised `FileNotFoundError` if it was missing.  The archive
    ships an ARM64 `.so`, so on any x86_64 machine (and on the archive as
    distributed) the whole `embodied_agent` package fails to import.

    We now:
      1. Try to load the Rust core if it exists AND its ELF arch matches
         the current interpreter (via a cheap magic-byte check).
      2. If that fails, fall back to a functionally-equivalent pure-numpy
         implementation of `snn_forward` (Leaky-Integrate-and-Fire +
         Hebbian STDP on uint8 memristor weights).
      3. Set `USING_RUST_CORE` for downstream code / tests to introspect.

    Set the env var `HYPERION_FORCE_PY=1` to skip the Rust attempt
    entirely (useful for benchmarking parity).
"""
import os
import sys
import ctypes
import struct
import numpy as np

# ---------------------------------------------------------------------------
# Locate & validate the Rust shared library.
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_aurora_dir = os.path.dirname(_this_dir)
_hyperion_dir = os.path.dirname(_aurora_dir)
_lib_path = os.path.join(
    _hyperion_dir, 'hyperion_core', 'target', 'release', 'libhyperion_core.so'
)

_rust_core = None
USING_RUST_CORE = False


def _elf_arch_matches(path: str) -> bool:
    """Cheap check: does the ELF machine field match this Python's arch?"""
    try:
        with open(path, 'rb') as f:
            head = f.read(20)
        if head[:4] != b'\x7fELF':
            return False
        # e_machine is at offset 18, little-endian u16.
        e_machine = struct.unpack('<H', head[18:20])[0]
        # 0x3e = x86_64, 0xB7 = aarch64, 0x28 = ARM 32
        py_machine = os.uname().machine.lower()
        if 'x86_64' in py_machine or 'amd64' in py_machine:
            return e_machine == 0x3E
        if 'aarch64' in py_machine or 'arm64' in py_machine:
            return e_machine == 0xB7
        if py_machine.startswith('arm'):
            return e_machine == 0x28
        return False
    except Exception:
        return False


if (
    not os.environ.get('HYPERION_FORCE_PY')
    and os.path.exists(_lib_path)
    and _elf_arch_matches(_lib_path)
):
    try:
        _rust_core = ctypes.CDLL(_lib_path)
        _rust_core.snn_forward.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_float,
            ctypes.c_float,
        ]
        _rust_core.snn_forward.restype = None
        USING_RUST_CORE = True
    except Exception as e:  # pragma: no cover
        sys.stderr.write(
            f"[snn_engine] Rust core present but failed to load ({e}); "
            "falling back to pure-Python SNN.\n"
        )
        _rust_core = None
        USING_RUST_CORE = False


# ---------------------------------------------------------------------------
# Pure-python fallback (numpy vectorised).
# Semantics mirror hyperion_core/src/lib.rs::snn_forward exactly.
# ---------------------------------------------------------------------------
def _snn_forward_py(
    n_inputs: int,
    n_outputs: int,
    inputs: np.ndarray,   # uint8 (n_inputs,)
    weights: np.ndarray,  # uint8 (n_outputs, n_inputs)
    potentials: np.ndarray,  # float32 (n_outputs,)
    spikes_out: np.ndarray,  # uint8 (n_outputs,)
    threshold: float,
    leak: float,
) -> None:
    # 1. Leak
    potentials *= leak

    # 2. Excitation = weighted sum of active inputs, weights in [0, 1]
    active_mask = inputs > 0                          # (n_inputs,) bool
    if active_mask.any():
        active_w = weights[:, active_mask].astype(np.float32) / 255.0
        potentials += active_w.sum(axis=1)

    # 3. Fire
    fired = potentials >= threshold
    spikes_out[:] = fired.astype(np.uint8)
    potentials[fired] = 0.0

    # 4. Hebbian STDP on uint8 memristor weights.
    if active_mask.any():
        # LTP on fired outputs
        if fired.any():
            sub = weights[np.ix_(fired, active_mask)].astype(np.int16) + 1
            np.clip(sub, 0, 255, out=sub)
            weights[np.ix_(fired, active_mask)] = sub.astype(np.uint8)
        # LTD on non-fired outputs
        not_fired = ~fired
        if not_fired.any():
            sub = weights[np.ix_(not_fired, active_mask)].astype(np.int16) - 1
            np.clip(sub, 0, 255, out=sub)
            weights[np.ix_(not_fired, active_mask)] = sub.astype(np.uint8)


# ---------------------------------------------------------------------------
class MemristorLayer:
    """LIF memristor layer with Hebbian STDP; Rust-accelerated when possible."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        threshold: float = 1.0,
        leak: float = 0.9,
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.threshold = float(threshold)
        self.leak = float(leak)

        self.weights = np.full((n_outputs, n_inputs), 128, dtype=np.uint8)
        self.potentials = np.zeros(n_outputs, dtype=np.float32)

    def forward(self, spikes_in: np.ndarray) -> np.ndarray:
        assert spikes_in.dtype == np.uint8 and spikes_in.shape == (self.n_inputs,)
        spikes_out = np.zeros(self.n_outputs, dtype=np.uint8)

        if USING_RUST_CORE:
            in_ptr = spikes_in.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            w_ptr = self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            v_ptr = self.potentials.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            out_ptr = spikes_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            _rust_core.snn_forward(
                self.n_inputs, self.n_outputs,
                in_ptr, w_ptr, v_ptr, out_ptr,
                self.threshold, self.leak,
            )
        else:
            _snn_forward_py(
                self.n_inputs, self.n_outputs,
                spikes_in, self.weights, self.potentials, spikes_out,
                self.threshold, self.leak,
            )

        return spikes_out
