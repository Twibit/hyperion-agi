import sys
import os
import time
from embodied_agent.organism import AuroraOmnia
from embodied_agent.config import AuroraOmniaConfig
from cognitive_plugins.omni_optical_nerve import OmniOpticalNerve
from adaptive_constants import (
    WIKTIONARY_PATH, INGESTION_DEFAULT_DRIVE, INGESTION_SLEEP_INTERVAL,
    ADAPTIVE_TIER, ADAPTIVE_LR_CORTEX
)

print("==================================================")
print(" OMNISCIENT LIFELONG LEARNING PIPELINE ")
print("==================================================\n")

cfg = AuroraOmniaConfig()
cfg.thalamic_router_enable = True
cfg.genius_enable = True
cfg.foundation_enable = True
cfg.tier = ADAPTIVE_TIER
cfg.lr_cortex = ADAPTIVE_LR_CORTEX
print(f"[System] Brain tier auto-selected: '{ADAPTIVE_TIER}' with adaptive learning rate {ADAPTIVE_LR_CORTEX}")
org = AuroraOmnia(cfg)
eye = OmniOpticalNerve()

file_path = WIKTIONARY_PATH

print("[System] Initiating Omniscient Eye background ingestion...")
drive = INGESTION_DEFAULT_DRIVE

definitions_learned = 0
start_time = time.time()

def biological_ingestion_stream(node):
    """
    O(1) Memory Stream Callback:
    As soon as the eye foveates a node, it is instantly piped to the cortex,
    learned, and destroyed from memory. No lists, no RAM bloat.
    """
    global definitions_learned, start_time
    org.ingest({"text": node["text"]})
    definitions_learned += 1
    
    # Adaptive Memory Consolidation (Sleep cycle to prevent biological RAM overload)
    if definitions_learned % INGESTION_SLEEP_INTERVAL == 0:
        org.sleep()
        elapsed = time.time() - start_time
        print(f"  [Cortex] Absorbed {definitions_learned:,} concepts in {elapsed:.1f}s. REM cycle complete.")

print(f"\n[System] Eye and Cortex synchronized. Initiating Gigabit Rust Vision Pipeline.")
eye.perceive(file_path, active_drive_keywords=drive, on_foveate_callback=biological_ingestion_stream, skim_stride=1)

print(f"\n[System] Background process terminating. Total highly salient concepts learned: {definitions_learned:,}")
