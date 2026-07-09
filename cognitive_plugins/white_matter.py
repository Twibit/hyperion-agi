import asyncio
import threading
import time

class WhiteMatterTract:
    """
    Axonal Delays and Subconscious Processing.
    Allows the Prefrontal Cortex to offload complex, multi-step thoughts into a
    background asynchronous loop. Simulates biological Axonal Delays where thoughts
    take physical time to travel across the brain, resulting in a sudden "Eureka" 
    moment when they complete, all while leaving the main thread at 0ms latency.
    """
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self.running = False

    def _start_loop(self):
        print("[White Matter] Asynchronous tract established. Subconscious is active.")
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def boot(self):
        if not self.running:
            self.running = True
            self._thread.start()

    async def _axonal_transmission(self, thought_function, args, delay_ms=2000):
        """
        Simulates the physical time it takes for a complex signal to cross the brain.
        """
        import os
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
        from cognitive_plugins.pineal_gland import PinealGland
        pineal = PinealGland()
        
        # Artificial biological delay (subconscious processing)
        await pineal.async_sleep(delay_ms)
        
        # Execute the heavy thought
        result = thought_function(*args)
        
        # Inject the Eureka moment back
        print(f"\\n[Subconscious Eureka!] Background thought resolved: {result}")
        return result

    def queue_subconscious_thought(self, thought_function, *args, delay_ms=2000):
        """
        Offloads a thought to the subconscious. Returns immediately (0ms delay).
        """
        if not self.running:
            print("[White Matter] Error: Tract not booted.")
            return
            
        asyncio.run_coroutine_threadsafe(
            self._axonal_transmission(thought_function, args, delay_ms),
            self.loop
        )
        print("[Basal Ganglia] Thought offloaded to subconscious white matter. Main thread free.")

    def shutdown(self):
        if self.running:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join()
            self.running = False
