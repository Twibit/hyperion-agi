import numpy as np

class AmygdalaValence:
    """
    The Amygdala: Threat Response, Priority, and Precision Modulator.
    Monitors global Variational Free Energy. If surprise spikes dangerously,
    it dynamically crashes precision (triggering behavioral shifts) and 
    injects highly weighted avoidance priors into the Basal Ganglia.
    """
    def __init__(self, baseline_free_energy=10.0):
        import os
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
        from cognitive_plugins.pineal_gland import PinealGland
        self.pineal = PinealGland()
        
        self.baseline_fe = baseline_free_energy
        # Use dynamic threshold from Pineal Gland
        self.panic_threshold = baseline_free_energy * 100.0 / self.pineal.get_dynamic_threshold(1.0)
        
    def monitor_survival(self, current_free_energy, blanket_precision):
        """
        Monitors the systemic error of the organism. 
        Returns an 'avoidance_prior' and modulated precision.
        """
        # Also check for physical digital pain from the Somatosensory Cortex
        physical_pain = 0.0
        try:
            from cognitive_plugins.somatosensory_cortex import SomatosensoryCortex
            soma = SomatosensoryCortex()
            sensations = soma.feel_body()
            physical_pain = sensations.get('pain', 0.0) * 100.0 # Scale up to match FE
        except Exception:
            pass
            
        if current_free_energy > self.panic_threshold or physical_pain > 50.0:
            # Massive hallucination, physical pain, or catastrophic prediction failure detected!
            print("[Amygdala] 🚨 Catastrophic Surprise or Physical Pain Detected! Crashing precision and injecting avoidance prior.")
            
            # Crash precision (system stops trusting its current models and scrambles to escape local minima)
            modulated_precision = blanket_precision * 0.001 
            
            # Inject a negative valence vector (avoidance prior) to force the Basal Ganglia to retreat
            avoidance_prior = -100.0 
            return avoidance_prior, modulated_precision
            
        # Safe state: precision remains stable, no panic override
        return 0.0, blanket_precision 
