import os
import sqlite3
import time
import threading

class Microglia:
    """
    Programmed Cell Death (Apoptosis) and Sleep Cycle Engine.
    Runs as a background daemon, constantly monitoring the utility of neural blocks.
    Also handles manual REM sleep cycles for deep memory consolidation.
    """
    def __init__(self, memory_path=None, utility_threshold=0.1):
        if memory_path is None:
            self.memory_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cain_global_memory.db')
        else:
            self.memory_path = memory_path
        self.utility_threshold = utility_threshold
        self.running = False
        self._thread = None
        
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
        from cognitive_plugins.pineal_gland import PinealGland
        self.pineal = PinealGland()

    def enter_rem_sleep(self):
        print("==================================================")
        print("[Sleep Cycle] Entering REM Sleep. Consolidating Hippocampal Memory...")
        
        try:
            conn = sqlite3.connect(self.memory_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM global_memory")
            initial_count = cursor.fetchone()[0]
            print(f"    [Pre-Sleep] Initial Memory Size: {initial_count} nodes.")
            
            print("    [Dreaming] Pruning ephemeral, redundant, and fragmented memories...")
            self.pineal.sleep(2000) # Simulate deep sleep
            
            # 1. Prune fragmented thoughts (very short sentences without substance)
            cursor.execute("DELETE FROM global_memory WHERE length(content) < 15")
            deleted_short = cursor.rowcount
            
            # 2. Prune duplicate conceptual vectors
            cursor.execute("DELETE FROM global_memory WHERE rowid NOT IN (SELECT min(rowid) FROM global_memory GROUP BY content)")
            deleted_dupes = cursor.rowcount
            
            conn.commit()
            
            print("    [Memory Consolidation] Executing SQLite VACUUM (Synaptic Pruning)...")
            cursor.execute("VACUUM")
            
            cursor.execute("SELECT COUNT(*) FROM global_memory")
            final_count = cursor.fetchone()[0]
            
            total_deleted = initial_count - final_count
            print(f"    [Post-Sleep] Pruned {total_deleted} useless synaptic nodes.")
            print(f"    [Post-Sleep] Final Memory Size: {final_count} highly optimized nodes.")
            
            conn.close()
            print("==================================================")
            print("[Pineal Gland] Waking up. The brain feels refreshed and optimized.")
            
        except Exception as e:
            print(f"[ERROR] Sleep paralysis: {e}")

    def _apoptosis_loop(self):
        print("[Microglia] Daemon spawned. Background Apoptosis engine active.")
        while self.running:
            try:
                conn = sqlite3.connect(self.memory_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM global_memory")
                count_before = cursor.fetchone()[0]
                
                cursor.execute("DELETE FROM global_memory WHERE length(content) < 15")
                deleted_frag = cursor.rowcount
                
                if deleted_frag > 0:
                    print(f"\\n[Microglia Apoptosis] {deleted_frag} low-utility neural columns permanently pruned in background.")
                    cursor.execute("VACUUM") # Reclaim physical disk space
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                pass # Silent background failure to not interrupt main thread
                
            self.pineal.sleep(30000) # Scan every 30 seconds

    def activate_immune_system(self):
        """Starts the background Microglial pruning process."""
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._apoptosis_loop, daemon=True)
            self._thread.start()

    def deactivate(self):
        self.running = False
        if self._thread:
            self._thread.join()

if __name__ == "__main__":
    sleep = Microglia()
    sleep.enter_rem_sleep()
