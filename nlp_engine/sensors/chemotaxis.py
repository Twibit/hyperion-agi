import numpy as np

class PeripheralSniffer:
    """
    Low-Energy, Broad Sensor (The Peripheral Vision / Olfaction).
    Executes rapid index checks (regex, AST skeletons) to identify 'Epistemic Hotspots'
    without semantically reading full files or wasting energy.
    """
    def sniff_environment(self, environment_map, target_prior):
        """
        Calculates the preliminary Epistemic Value gradient of the environment.
        In a real codebase, this runs grep/ast logic. 
        Here, it returns a gradient mapping of the potential hotspots.
        """
        # Mock gradient detection based on target correlation
        hotspots = []
        for location in environment_map:
            # Scent strength represents Expected Information Gain (Epistemic Value)
            scent_strength = np.random.rand() * np.mean(np.abs(target_prior))
            hotspots.append((location, scent_strength))
        return hotspots

class FovealReader:
    """
    High-Acuity, High-Energy Sensor (The Fovea / Mouth).
    Only activates at precise coordinates. Ingests deep semantic data natively 
    into SNN spikes for the Left Brain to evaluate.
    """
    def ingest_data(self, precise_location):
        """Extracts dense data into sparse spikes."""
        # Outputs a spike train representing the data at the foveated location
        return np.random.rand(1, 16) > 0.95 

class ChemotaxisEngine:
    """
    Bacterial Run-and-Tumble Active Sensing Engine.
    Follows epistemic gradients (Information). If the scent gets stronger, it 'Runs'.
    If it weakens (a dead-end file), it 'Tumbles' (random leap to escape local minima).
    """
    def __init__(self):
        self.sniffer = PeripheralSniffer()
        self.fovea = FovealReader()
        self.current_gradient = 0.0
        
    def active_forage(self, environment_map, target_prior):
        """The run-and-tumble loop."""
        
        # 1. Tumble (Sniff the broad environment for hotspots)
        hotspots = self.sniffer.sniff_environment(environment_map, target_prior)
        
        # 2. Identify the strongest Epistemic Gradient
        best_location, expected_information = max(hotspots, key=lambda x: x[1])
        
        # 3. Active Sensing Evaluation
        if expected_information > self.current_gradient:
            # The scent is stronger! Enter "Run" mode (Gradient climbing)
            # Saccadic suppression: Ignore all other files, jump instantly to the target
            self.current_gradient = expected_information
            spikes = self.fovea.ingest_data(best_location)
            action = f"RUN (Targeted Saccade to {best_location})"
            return spikes, action, best_location
            
        else:
            # The scent went cold (Dead end/local minima). Enter "Tumble" mode.
            self.current_gradient *= 0.1 # Rapid decay of interest
            random_leap = np.random.choice(environment_map)
            action = f"TUMBLE (Stochastic leap to {random_leap} to escape dead end)"
            return None, action, random_leap
