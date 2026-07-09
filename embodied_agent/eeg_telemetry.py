"""
Biological Telemetry (EEG) Engine — V14 Aurora-Foundation.
===========================================================
Monitors and logs real-time biological metrics to disk, 
enabling deep analysis of how the organism's connectome and 
metabolic systems evolve over long continuous runs.
"""
import os
import csv
import time
from typing import Dict, Any

class EEGTelemetry:
    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eeg_logs')
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Real-time awake cycle metrics
        self.step_file = open(os.path.join(self.log_dir, "awake_metrics.csv"), "w", newline="")
        self.step_writer = csv.writer(self.step_file)
        self.step_writer.writerow([
            "step", "timestamp", "sensory_err", "td_err", 
            "dopamine", "cortisol", "interhem_coherence", 
            "metabolic_atp_frac", "action_id"
        ])
        
        # Consolidation (Sleep) metrics
        self.sleep_file = open(os.path.join(self.log_dir, "sleep_metrics.csv"), "w", newline="")
        self.sleep_writer = csv.writer(self.sleep_file)
        self.sleep_writer.writerow([
            "timestamp", "hebbian_edges_rewired", "pruned_nodes"
        ])

    def log_awake_step(self, step: int, out: Dict[str, Any]):
        """Log metrics from a single waking step."""
        mods = out.get("modulators", {})
        affect = out.get("affect", {})
        
        self.step_writer.writerow([
            step,
            time.time(),
            out.get("sensory_err", 0.0),
            out.get("td_err", 0.0),
            mods.get("dopamine", 0.0),
            affect.get("cortisol", 0.0),
            out.get("interhem_coherence", 0.0),
            out.get("metabolic_active_frac", 0.0),
            out.get("action", -1)
        ])
        self.step_file.flush()

    def log_sleep_cycle(self, sleep_stats: Dict[str, Any]):
        """Log metrics from a deep sleep / consolidation cycle."""
        self.sleep_writer.writerow([
            time.time(),
            sleep_stats.get("hebbian_edges_updated", 0),
            sleep_stats.get("microglia", {}).get("pruned", 0)
        ])
        self.sleep_file.flush()

    def close(self):
        self.step_file.close()
        self.sleep_file.close()
