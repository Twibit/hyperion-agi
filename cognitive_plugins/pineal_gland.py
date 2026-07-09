import time
import math

class PinealGland:
    """
    Biological Clock (Circadian Rhythm) & Delay Manager.
    Replaces all arbitrary time.sleep() calls across the architecture.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PinealGland, cls).__new__(cls)
            cls._instance.melatonin = 0.0
            cls._instance.atp_level = 100.0
            cls._instance.last_tick = time.time()
        return cls._instance
        
    def tick(self):
        """Update biological hormones over time."""
        now = time.time()
        elapsed = now - self.last_tick
        self.last_tick = now
        
        # ATP depletes as time goes on, Melatonin rises
        self.atp_level = max(10.0, self.atp_level - (elapsed * 0.5))
        self.melatonin = min(1.0, self.melatonin + (elapsed * 0.01))
        
    def reset_atp(self):
        """Restore energy after a task completes."""
        self.atp_level = 100.0
        self.melatonin = max(0.0, self.melatonin - 0.2)

    def sleep(self, base_ms: int):
        """
        Dynamically sleep based on biological energy.
        If ATP is low, delays increase.
        """
        self.tick()
        
        # Calculate dynamic delay
        # At 100 ATP, multiplier is 1.0. At 10 ATP, multiplier is 2.5
        fatigue_multiplier = 1.0 + (100.0 - self.atp_level) / 60.0
        
        # Add melatonin drowsiness
        drowsiness = 1.0 + self.melatonin
        
        final_delay_ms = base_ms * fatigue_multiplier * drowsiness
        time.sleep(final_delay_ms / 1000.0)
        
    async def async_sleep(self, base_ms: int):
        import asyncio
        self.tick()
        fatigue_multiplier = 1.0 + (100.0 - self.atp_level) / 60.0
        drowsiness = 1.0 + self.melatonin
        final_delay_ms = base_ms * fatigue_multiplier * drowsiness
        await asyncio.sleep(final_delay_ms / 1000.0)

    def get_dynamic_threshold(self, base_val: float) -> float:
        """Modulate thresholds based on biological state."""
        self.tick()
        fatigue = (100.0 - self.atp_level) / 100.0
        return base_val * (1.0 + fatigue)
