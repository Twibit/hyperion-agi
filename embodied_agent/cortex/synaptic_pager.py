"""
Synaptic Pager (Astrocytic Memory Management)
=============================================

To achieve brain-scale parameter counts (100 Trillion synapses) on consumer hardware,
we must exploit the fact that biological brains are intensely sparse. At any given moment,
99% of neural clusters (MoE Experts) are completely inactive.

This system monitors expert usage. If an expert is not routed to for a set number of steps,
its Synaptic Weights (W_gen, W_rec) are quantized to float16 and flushed to a memory-mapped 
file on disk. 

When the Thalamic Router calls upon the dormant expert, it is instantly paged back into 
VRAM/RAM and restored to float32 for peak precision during the forward pass.
"""
import os
import numpy as np
from typing import Dict, Any

class SynapticPager:
    def __init__(self, cache_dir: str = "/tmp/hyperion_synapse_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.registry = {} # expert_id -> metadata
        self.step_counter = 0

    def tick(self):
        self.step_counter += 1

    def touch(self, expert_id: str):
        """Mark an expert as actively used this step."""
        if expert_id not in self.registry:
            self.registry[expert_id] = {"last_used": self.step_counter, "state": "RAM"}
        else:
            self.registry[expert_id]["last_used"] = self.step_counter

    def should_hibernate(self, expert_id: str, max_idle: int = 50) -> bool:
        if expert_id not in self.registry:
            return False
        if self.registry[expert_id]["state"] == "DISK":
            return False
        return (self.step_counter - self.registry[expert_id]["last_used"]) > max_idle

    def evict(self, expert_id: str):
        """Permanently remove an expert from the cache (disk + registry)."""
        path = os.path.join(self.cache_dir, f"{expert_id}.npz")
        if os.path.exists(path):
            os.remove(path)
        self.registry.pop(expert_id, None)

    def hibernate_expert(self, expert_id: str, W_gen: np.ndarray, W_rec: np.ndarray) -> bool:
        """Compress to float16 and save to disk, freeing RAM."""
        path = os.path.join(self.cache_dir, f"{expert_id}.npz")
        
        # 1. Quantize to Float16 to halve disk I/O bandwidth
        W_gen_f16 = W_gen.astype(np.float16)
        W_rec_f16 = W_rec.astype(np.float16)
        
        # 2. Save compressed
        np.savez_compressed(path, W_gen=W_gen_f16, W_rec=W_rec_f16)
        
        self.registry[expert_id]["state"] = "DISK"
        return True

    def wake_expert(self, expert_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Load from disk and dequantize back to float32 for peak performance."""
        path = os.path.join(self.cache_dir, f"{expert_id}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert {expert_id} not found in cache.")
            
        data = np.load(path)
        # Dequantize back to f32 for the math engine
        W_gen = data["W_gen"].astype(np.float32)
        W_rec = data["W_rec"].astype(np.float32)
        
        self.registry[expert_id]["state"] = "RAM"
        self.registry[expert_id]["last_used"] = self.step_counter
        return W_gen, W_rec
