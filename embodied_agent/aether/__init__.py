"""
Aurora-Aether (V13) — Distributed/Lazy Architecture Layer.
"""
from .lazy_loader import get_aether_store, AetherDiskStore

__all__ = ["get_aether_store", "AetherDiskStore"]
