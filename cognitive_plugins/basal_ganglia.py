from collections import deque
import time

class BasalGanglia:
    def __init__(self, max_sequence=3):
        # The Basal Ganglia sequences actions. Capped at 3 to prevent cognitive bloat (Idea 5)
        self.action_queue = deque(maxlen=max_sequence)
        import os
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
        from cognitive_plugins.pineal_gland import PinealGland
        self.pineal = PinealGland()
        
        print("[Basal Ganglia] Initialized Pre-Motor Buffer for action sequencing.")

    def queue_action(self, action_intent):
        print(f"[Premotor Cortex] Formulating intent: {action_intent}")
        if len(self.action_queue) == self.action_queue.maxlen:
            print("[Cognitive Bottleneck] Sequence queue is full. Suppressing new intents.")
            return False
            
        self.action_queue.append(action_intent)
        print(f"[Basal Ganglia] Action queued. Current sequence length: {len(self.action_queue)}")
        return True

    def release_sequence(self, motor_cortex_callback):
        print("==================================================")
        print("[SMA] Supplementary Motor Area authorizing sequence execution...")
        
        results = []
        while self.action_queue:
            action = self.action_queue.popleft()
            print(f"  -> [Spinal Cord] Firing action signal: {action}")
            
            # Execute and gather sensory return
            sensory_feedback = motor_cortex_callback(action)
            results.append(sensory_feedback)
            
            self.pineal.sleep(500) # Biological synaptic delay
            
        print("[Basal Ganglia] Sequence complete. Awaiting environmental consequences.")
        print("==================================================")
        return results

if __name__ == "__main__":
    bg = BasalGanglia()
    bg.queue_action("Scan database")
    bg.queue_action("Summarize nodes")
    bg.release_sequence(lambda x: print(f"Mock execution of {x}"))
