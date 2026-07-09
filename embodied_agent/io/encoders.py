"""Modality encoders — same interface as v2, with one new optional 'interoception'
encoder for the affect signal so the cortex can attend to its own body state."""
from __future__ import annotations
import numpy as np, hashlib
from typing import Any


def _stable_hash(s: str, dim: int, seed: int) -> np.ndarray:
    h = hashlib.blake2b(f"{seed}:{s}".encode("utf-8"), digest_size=64).digest()
    raw = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0 - 0.5
    out = np.tile(raw, (dim // 64) + 1)[:dim]
    return out.astype(np.float32)


class Encoder:
    name = "base"
    def encode(self, x: Any, latent_dim: int) -> np.ndarray: raise NotImplementedError


class TextEncoder(Encoder):
    name = "text"
    def __init__(self, seed=0): self.seed = seed
    def encode(self, x, latent_dim):
        s = str(x).lower(); toks = s.split()
        if not toks: return np.zeros(latent_dim, dtype=np.float32)
        vecs = [_stable_hash(t, latent_dim, self.seed) for t in toks]
        z = np.mean(vecs, axis=0)
        for i in range(len(s) - 1):
            z += 0.1 * _stable_hash(s[i:i + 2], latent_dim, self.seed + 1)
        return z / (np.linalg.norm(z) + 1e-8)


class ScalarEncoder(Encoder):
    name = "scalar"
    def __init__(self, n_fields=64, vmin=-10.0, vmax=10.0):
        self.centers = np.linspace(vmin, vmax, n_fields, dtype=np.float32)
        self.sigma = (vmax - vmin) / n_fields
    def encode(self, x, latent_dim):
        v = float(x)
        a = np.exp(-((self.centers - v) ** 2) / (2 * self.sigma ** 2)).astype(np.float32)
        out = np.tile(a, latent_dim // a.size + 1)[:latent_dim]
        return out / (np.linalg.norm(out) + 1e-8)


class VisionEncoder(Encoder):
    name = "vision"
    def __init__(self, latent_dim, seed=0):
        rng = np.random.default_rng(seed)
        # V31 Supernatural Vision: Eagle-Eye Foveal + Gestalt Peripheral
        # 16x16 peripheral (256) + 16x16 raw center fovea (256) = 512
        self.W = rng.standard_normal((latent_dim, 512)).astype(np.float32) / 16.0
    def encode(self, x, latent_dim):
        img = np.asarray(x, dtype=np.float32)
        if img.ndim == 3: img = img.mean(axis=-1)
        H, W = img.shape
        if H < 16 or W < 16:
            img = np.pad(img, ((0, max(0, 16 - H)), (0, max(0, 16 - W))))
            H, W = img.shape
        
        # 1. Peripheral Gestalt (blurry awareness of everything)
        sh, sw = H // 16, W // 16
        peri = img[:sh * 16, :sw * 16].reshape(16, sh, 16, sw).mean(axis=(1, 3)).reshape(-1) / 255.0
        
        # 2. Foveal Center (pixel-perfect eagle-eye detail of the exact center)
        cy, cx = H // 2, W // 2
        fovea_raw = img[max(0, cy-8):cy+8, max(0, cx-8):cx+8]
        fovea = np.zeros((16, 16), dtype=np.float32)
        fh, fw = fovea_raw.shape
        fovea[:fh, :fw] = fovea_raw / 255.0
        
        feat = np.concatenate([peri, fovea.reshape(-1)])
        z = self.W @ feat
        return z / (np.linalg.norm(z) + 1e-8)


class AudioEncoder(Encoder):
    name = "audio"
    def __init__(self, latent_dim, n_bands=40, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((latent_dim, n_bands)).astype(np.float32) / np.sqrt(n_bands)
        self.n_bands = n_bands
    def encode(self, x, latent_dim):
        wav = np.asarray(x, dtype=np.float32).ravel()
        if wav.size < self.n_bands: wav = np.pad(wav, (0, self.n_bands - wav.size))
        chunks = np.array_split(wav, self.n_bands)
        feat = np.array([np.log1p((c ** 2).mean()) for c in chunks], dtype=np.float32)
        z = self.W @ feat
        return z / (np.linalg.norm(z) + 1e-8)


class ProprioEncoder(Encoder):
    name = "proprio"
    def __init__(self, latent_dim, state_dim=32, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((latent_dim, state_dim)).astype(np.float32) / np.sqrt(state_dim)
        self.state_dim = state_dim
    def encode(self, x, latent_dim):
        v = np.asarray(x, dtype=np.float32).ravel()
        if v.size < self.state_dim: v = np.pad(v, (0, self.state_dim - v.size))
        else: v = v[:self.state_dim]
        return (self.W @ v) / (np.linalg.norm(v) + 1e-8)


class InteroceptionEncoder(Encoder):
    """V31: Supernatural Interoception (Digital Omniscience).
    The agent feels not only its biochemical affect, but also physically feels its own 
    host machine's RAM utilization and CPU thermals as deep bodily sensations."""
    name = "interoception"
    def __init__(self, latent_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((latent_dim, 6)).astype(np.float32) / 2.0
    def encode(self, x, latent_dim):
        cpu_temp = 0.5
        ram_usage = 0.5
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            if temps: cpu_temp = list(temps.values())[0][0].current / 100.0
            ram_usage = psutil.virtual_memory().percent / 100.0
        except Exception:
            pass
        
        if isinstance(x, dict):
            v = np.array([x.get("valence", 0.0), x.get("arousal", 0.0),
                          x.get("cortisol", 0.0), x.get("pain_recent", 0.0),
                          cpu_temp, ram_usage], dtype=np.float32)
        else:
            v = np.zeros(6, dtype=np.float32)
            
        z = self.W @ v
        return z / (np.linalg.norm(z) + 1e-8)


class OmniEncoder(Encoder):
    """V31: Supernatural Omni-Vision.
    Connects the brain to the OmniOpticalNerve, allowing it to instantly perceive 
    the spatial geometry of abstract digital structures (HTML buttons, PDF streams, Code ASTs)."""
    name = "omni"
    def __init__(self, latent_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((latent_dim, 3)).astype(np.float32) / 2.0
        
    def encode(self, x, latent_dim):
        import sys, os
        # Need to dynamically import OmniOpticalNerve
        nerve_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cognitive_plugins")
        if nerve_path not in sys.path:
            sys.path.append(nerve_path)
            
        try:
            from omni_optical_nerve import OmniOpticalNerve
            nerve = OmniOpticalNerve()
            # Foveate on the target path/url
            connectome = nerve.perceive(str(x))
            
            # The agent "sees" the density of interactive buttons, semantic text, and geometric shapes!
            interactive_density = len(connectome.get("interactive_nodes", []))
            semantic_density = len(connectome.get("semantic_nodes", []))
            spatial_density = len(connectome.get("spatial_geometry", []))
            
            v = np.array([interactive_density, semantic_density, spatial_density], dtype=np.float32)
        except Exception:
            v = np.zeros(3, dtype=np.float32)
            
        z = self.W @ v
        return z / (np.linalg.norm(z) + 1e-8)


def default_registry(latent_dim: int, seed: int = 0) -> dict:
    return {
        "text":          TextEncoder(seed=seed),
        "scalar":        ScalarEncoder(),
        "vision":        VisionEncoder(latent_dim, seed=seed + 1),
        "audio":         AudioEncoder(latent_dim, seed=seed + 2),
        "proprio":       ProprioEncoder(latent_dim, seed=seed + 3),
        "interoception": InteroceptionEncoder(latent_dim, seed=seed + 4),
        "omni":          OmniEncoder(latent_dim, seed=seed + 5),
    }
