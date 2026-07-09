"""
Hyperdimensional Vector Symbolic Architecture (HVSA) — v6 "Omniscient Eye".
===========================================================================

A universal modality-agnostic substrate.  Every percept — text token, image
patch, audio frame, proprioceptive vector — is projected into the same
N-dimensional hypervector space (default N=8192).  Algebraic operations
(bind, bundle, permute) let the system compose symbols compositionally
without ever leaving the same representational manifold.

Key operations:
  * `bind(a, b)`     — element-wise multiplication (associative, invertible)
  * `unbind(c, b)`   — recover `a` from `bind(a, b) = c`
  * `bundle(*hvs)`   — superposition (component-wise mean + sign)
  * `permute(hv, k)` — protected role binding via cyclic shift
  * `cosine(a, b)`   — similarity in the hyperdimensional space

Why this matters for brain mimicry:
  Real neocortex is famously modality-agnostic — the same six-layer column
  processes vision, audition, or somatosensation depending on inputs.
  HVSA gives every cortical sub-system a *common currency*.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, Hashable


def _seed_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hadamard product (the standard VSA bind for bipolar vectors)."""
    return (a * b).astype(np.float32)


def unbind(c: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For bipolar HVs (±1), bind is self-inverse."""
    return (c * b).astype(np.float32)


def bundle(*hvs: np.ndarray) -> np.ndarray:
    """Superposition: component-wise mean then sign-thresholded."""
    if not hvs:
        raise ValueError("bundle() needs at least one hypervector")
    s = np.mean(np.stack(hvs, axis=0), axis=0)
    return np.sign(s).astype(np.float32)


def permute(hv: np.ndarray, k: int = 1) -> np.ndarray:
    return np.roll(hv, k).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((a @ b) / (na * nb))


class HVSASensorium:
    """Modality-agnostic perceptual substrate.

    Maintains a codebook mapping atomic symbols (tokens, patch IDs, etc.)
    to random hypervectors, and a per-modality role vector.  Encoding
    fuses content × role so that downstream cortex can disambiguate
    'red' as a visual feature vs 'red' as a word.
    """

    def __init__(self, dim: int = 8192, seed: int = 1729,
                 latent_dim: int = 512):
        self.dim = int(dim)
        self.latent_dim = int(latent_dim)
        self.rng = _seed_rng(seed)
        # codebook of atomic symbols -> bipolar hypervectors
        self._codebook: Dict[Hashable, np.ndarray] = {}
        # per-modality "role" hypervector (permanent)
        self._modality_hv: Dict[str, np.ndarray] = {}
        # projection from hyperdim -> latent (random but fixed)
        self._proj = (self.rng.standard_normal((latent_dim, dim)).astype(np.float32)
                      / np.sqrt(dim))
        self.codebook_misses = 0
        self.observations = 0

    # -------- internal --------
    def _rand_hv(self) -> np.ndarray:
        return self.rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)

    def _atom(self, sym: Hashable) -> np.ndarray:
        if sym not in self._codebook:
            self._codebook[sym] = self._rand_hv()
            self.codebook_misses += 1
        return self._codebook[sym]

    def _role(self, modality: str) -> np.ndarray:
        if modality not in self._modality_hv:
            self._modality_hv[modality] = self._rand_hv()
        return self._modality_hv[modality]

    # -------- public --------
    def encode(self, value: Any, modality: str = "generic") -> np.ndarray:
        """Project arbitrary input into the latent space via HVSA.

        For text (str): bundle bind(token_hv, position_hv) across tokens.
        For sequences (list/tuple/array): bundle bind(element_hv, pos_hv).
        For scalars/ints/floats: use a binned symbol.
        """
        self.observations += 1
        role = self._role(modality)

        if isinstance(value, str):
            toks = value.split() or [""]
            hvs = []
            for i, t in enumerate(toks[:256]):
                hv = bind(self._atom(("tok", t)), permute(role, i))
                hvs.append(hv)
            content = bundle(*hvs)
        elif isinstance(value, (int, np.integer)):
            content = bind(self._atom(("int", int(value) // 64)), role)
        elif isinstance(value, (float, np.floating)):
            b = int(round(float(value) * 16)) // 8
            content = bind(self._atom(("flt", b)), role)
        elif isinstance(value, dict):
            hvs = [bind(self._atom(("k", k)), self._atom(("v", str(v)[:32])))
                   for k, v in list(value.items())[:32]]
            content = bundle(*hvs) if hvs else role
        elif hasattr(value, "__iter__"):
            try:
                arr = np.asarray(value, dtype=np.float32).reshape(-1)[:256]
                bins = np.clip((arr * 8).astype(np.int32), -64, 64)
                hvs = [bind(self._atom(("seq", int(b))), permute(role, i))
                       for i, b in enumerate(bins)]
            except Exception:
                hvs = [bind(self._atom(("obj", repr(v)[:32])), permute(role, i))
                       for i, v in enumerate(list(value)[:64])]
            content = bundle(*hvs) if hvs else role
        else:
            content = bind(self._atom(("obj", repr(value)[:32])), role)

        # Project hyperdim -> latent_dim and L2-normalise
        z = self._proj @ content
        n = np.linalg.norm(z)
        if n > 1e-9:
            z = z / n
        return z.astype(np.float32)

    def stats(self) -> dict:
        return dict(
            dim=self.dim,
            codebook_size=len(self._codebook),
            codebook_misses=self.codebook_misses,
            modalities_seen=len(self._modality_hv),
            observations=self.observations,
        )
