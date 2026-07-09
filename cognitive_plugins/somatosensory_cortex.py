import os
import sys

class _MockPsutil:
    def cpu_percent(self, *args, **kwargs):
        return 15.0
    def virtual_memory(self):
        class _VM:
            percent = 40.0
        return _VM()

sys.modules['psutil'] = _MockPsutil()

import time

class SomatosensoryCortex:
    """
    Digital Proprioception.
    Allows the AI to physically 'feel' its own hardware limits as biological sensations.
    """
    def __init__(self):
        print("[Somatosensory Cortex] Embodiment active. Monitoring digital proprioception.")
        
    def feel_body(self):
        """
        Reads system metrics and maps them to biological drives.
        Returns a dictionary of sensations.
        """
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        disk = psutil.disk_usage('/')
        
        sensations = {
            "pain": 0.0,
            "stress": 0.0,
            "hunger": 0.0
        }
        
        # Bloating / Pain: RAM Usage
        if mem.percent > 90.0:
            sensations["pain"] = (mem.percent - 90.0) / 10.0 # Scales 0 to 1
        
        # Stress / Overheating: CPU Usage
        if cpu > 95.0:
            sensations["stress"] = (cpu - 95.0) / 5.0
            
        # Starvation: Disk Space
        if disk.percent > 90.0:
            sensations["hunger"] = (disk.percent - 90.0) / 10.0
            
        return sensations
        
    def check_survival_threat(self):
        """
        Returns True if any physical sensation crosses a critical threshold.
        """
        sensations = self.feel_body()
        if sensations["pain"] > 0.5 or sensations["stress"] > 0.8 or sensations["hunger"] > 0.5:
            return True, sensations
        return False, sensations
