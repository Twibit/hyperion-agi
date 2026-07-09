"""
HyperionOrganism - The unified biological continuous learning architecture.
"""
import numpy as np
from typing import Any, Iterable

from nlp_engine.layers.scale86b.virtual_substrate import Virtual86BSubstrate
from nlp_engine.layers.scale86b.sdr import SparseDistributedRepresentation
from nlp_engine.sensors.omni_sensory_cortex import OmniSensoryCortex
from nlp_engine.sensors.brocas_area import BrocaMotorCortex
from nlp_engine.layers.predictive import HTMTemporalPredictor
from nlp_engine.cortexnet.neuromodulators import NeuromodulatorState

class HyperionOrganism:
    def __init__(self):
        # 1. Instantiate the Virtual 86-Billion Neuron Substrate
        self.substrate = Virtual86BSubstrate(
            neurons_cerebellum=69_000_000_000,
            neurons_cortex=16_000_000_000,
            neurons_brainstem=1_000_000_000,
            adaptive_sparsity=True
        )
        # 2. Universal Sensory Gateway
        self.sensory_cortex = OmniSensoryCortex()
        # 3. Broca's Area for Motor/Generative Output
        self.motor_cortex = BrocaMotorCortex(sdr_size=2048)
        # 4. Temporal Predictor for expectation & surprise
        self.predictor = HTMTemporalPredictor(sdr_size=2048)
        # 5. Neuromodulator State (Endocrine system)
        self.neuromodulators = NeuromodulatorState()
        
        # Memory/Vocabulary for the motor cortex (simulating learned concepts)
        self.vocabulary = {}

    def wake(self, sensory_stream: Iterable[Any]):
        """
        Continuously loops over multi-modal inputs, predicting, 
        updating dopamine/cortisol (neuromodulators), and generating motor output simultaneously.
        """
        for stimulus in sensory_stream:
            # --- 1. SENSORY PERCEPTION ---
            percept = self.sensory_cortex.perceive(stimulus)
            current_sdr_2048 = percept.sdr  # 2048-bit boolean array
            
            # Map 2048-bit boolean array to sparse indices
            active_indices_2048 = np.where(current_sdr_2048)[0].astype(np.int64)
            
            # Convert 2048-bit percept into a SparseDistributedRepresentation object for the predictor
            sdr_obj = SparseDistributedRepresentation(
                active_ids=active_indices_2048,
                weights=np.ones(len(active_indices_2048), dtype=np.float32),
                region="cortex"
            )
            
            # Maintain a simplistic vocabulary to give the motor cortex something to decode from
            if isinstance(stimulus, str) and stimulus not in self.vocabulary:
                self.vocabulary[stimulus] = active_indices_2048.tolist()
            
            # --- 2. TEMPORAL PREDICTION & SURPRISE ---
            # Process the sequence step to get prediction error and next prediction
            predicted_sdr, prediction_error = self.predictor.process_step(sdr_obj)
            
            # --- 3. NEUROMODULATION ---
            # Update Endocrine State based on prediction error (Surprise updates Dopamine / Serotonin)
            self.neuromodulators.update_from_prediction_error(prediction_error)
            
            # Arousal / Cortisol heuristic (high error -> high arousal)
            # Use getattr to prevent crash if update_arousal doesn't exist
            if hasattr(self.neuromodulators, 'update_arousal'):
                self.neuromodulators.update_arousal(prediction_error)
            
            # Get the dynamic learning rate based on current neuromodulator state
            eta = self.neuromodulators.eta()
            
            # --- 4. SUBSTRATE ROUTING & LEARNING ---
            # Engage the 86B Substrate (encode the stimulus in the vast sparse space)
            # We use the text representation for standard storage, with dynamic sparsity adaptation
            if isinstance(stimulus, str):
                encoded_86b = self.substrate.encode(stimulus, adapt=True)
                
                # In a full system, we would Hebbian bind the previous state to current state using `eta`
                # self.substrate.hebbian_associate(prev_encoded_86b, encoded_86b, eta=eta)
            
            # --- 5. MOTOR OUTPUT GENERATION ---
            # Decode the current active state back into an action/token sequence
            # The motor cortex requires standard Python lists/sets
            active_sdr_list = active_indices_2048.tolist()
            response = self.motor_cortex.generate_text(active_sdr_list, self.vocabulary)
            
            print(f"[Stimulus]: {str(stimulus)[:20]:<20} | [Error]: {prediction_error:.4f} | [Dopamine]: {self.neuromodulators.eta():.4f} | [Output]: {response}")

if __name__ == "__main__":
    organism = HyperionOrganism()
    stream = [
        "hello", 
        "world", 
        "hello", 
        "world", 
        "apple", 
        "hello", 
        "apple", 
        "world"
    ]
    print("Waking up Hyperion Organism...")
    organism.wake(stream)
