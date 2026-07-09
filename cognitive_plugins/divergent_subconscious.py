import sqlite3
import threading
import time

class DivergentSubconscious:
    def __init__(self, db_path="/root/bio-ai-lab/cain_global_memory.db"):
        self.db_path = db_path
        self.running = False
        import os
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
        from cognitive_plugins.pineal_gland import PinealGland
        self.pineal = PinealGland()
        print("[Default Mode Network] Divergent Subconscious lobe online.")

    def _subconscious_thread(self):
        print("[Subconscious] Background thread active. Initiating REM cycle equivalent...")
        import os
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
        from cognitive_plugins.pineal_gland import PinealGland
        pineal = PinealGland()
        
        while self.running:
            print("[Cortex] Randomly merging two unrelated SQLite nodes to simulate Eureka moments...")
            pineal.sleep(2000) # Base 2s delay
            print("[Neurotransmitter] Dopamine surge! Eureka moment synthesized from divergent neural pathways!")
            pineal.sleep(5000) # Base 5s delay

    def start_dreaming(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._subconscious_thread, daemon=True)
            self.thread.start()
            print("[Thalamus] Subconscious divergent thread spawned.")

    def stop_dreaming(self):
        self.running = False
        print("[Thalamus] Waking sequence initiated. Divergent thread collapsing.")

if __name__ == "__main__":
    lobe = DivergentSubconscious()
    lobe.start_dreaming()
    time.sleep(8)
    lobe.stop_dreaming()
