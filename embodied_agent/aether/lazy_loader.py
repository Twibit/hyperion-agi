"""
Aurora-Aether (V13) — Distributed / Lazy Architecture.
======================================================

The biological brain does not load 100 Trillion synapses into "active RAM" 
simultaneously. It is physically distributed. 
In AI engineering, the primary bottleneck preventing infinite scaling is 
VRAM/RAM limits: even if only 2% of the Mixture-of-Experts fire, 100% of 
the parameters must traditionally be loaded into memory.

Aether solves this by implementing Lazy Activation. It uses the organism's 
Thalamic priority map and Metabolic budget to physically evict "sleeping" 
cortical regions from RAM to SSD (Cold Storage), and load them back only 
when the organism specifically needs them.

This allows Aurora to scale infinitely on consumer hardware.
"""
from __future__ import annotations
import os
import pickle
from typing import Any, Dict


class AetherDiskStore:
    """
    Virtual Memory manager for neural tissue. 
    Swaps inactive brain regions to disk to save RAM.
    
    WARNING: Currently uses `pickle` for serialization, which is unsafe 
    for untrusted data and fragile across refactors. In a production 
    environment, this should be replaced with `numpy.savez` paired with 
    a structured metadata file.
    """
    def __init__(self, cache_dir: str = "/tmp/aurora_aether", instance_id: str = "default"):
        self.cache_dir = os.path.join(cache_dir, instance_id)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Track which regions are currently on disk
        self._on_disk: Dict[str, bool] = {}
        self._n_swaps_out = 0
        self._n_swaps_in = 0

    def sleep_region(self, region_name: str, payload: Any) -> None:
        """Serialize a massive object (like an MoECortexStack) to disk and clear RAM."""
        if self._on_disk.get(region_name, False):
            return  # Already asleep
            
        path = os.path.join(self.cache_dir, f"{region_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(payload, f)
            
        self._on_disk[region_name] = True
        self._n_swaps_out += 1

    def wake_region(self, region_name: str) -> Any:
        """Load an object from disk back into RAM."""
        path = os.path.join(self.cache_dir, f"{region_name}.pkl")
        if not os.path.exists(path) or not self._on_disk.get(region_name, False):
            return None
            
        with open(path, "rb") as f:
            payload = pickle.load(f)
            
        self._on_disk[region_name] = False
        self._n_swaps_in += 1
        return payload
        
    def is_asleep(self, region_name: str) -> bool:
        return self._on_disk.get(region_name, False)
        
    def stats(self) -> Dict[str, Any]:
        return {
            "regions_on_disk": sum(self._on_disk.values()),
            "total_swaps_out": self._n_swaps_out,
            "total_swaps_in": self._n_swaps_in,
            "cache_dir": self.cache_dir
        }

# Registry of Aether stores keyed by instance_id.
# Avoids cross-contamination when multiple organisms exist in the same process.
_aether_registry: Dict[str, AetherDiskStore] = {}

def get_aether_store(instance_id: str = "default") -> AetherDiskStore:
    if instance_id not in _aether_registry:
        _aether_registry[instance_id] = AetherDiskStore(instance_id=instance_id)
    return _aether_registry[instance_id]
