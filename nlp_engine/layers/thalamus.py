import numpy as np
from nlp_engine.layers.snn_core import EnergyEfficientSNNLayer
from nlp_engine.core.fep_math import FreeEnergyMath

class ThalamicRouter:
    """
    The Thalamus: The Brain's Master Attention Gateway.
    It routes sensory spike trains to the Cortex, dynamically modulating
    synaptic thresholds based on top-down precision (attention).
    """
    def __init__(self, num_in, num_out, batch_size=1, base_threshold=100):
        # The Thalamus uses the extreme energy-efficiency SNN layer
        self.snn = EnergyEfficientSNNLayer(num_in, num_out, batch_size, threshold=base_threshold)
        self.base_threshold = base_threshold
        
    def route_attention(self, in_spikes, precision):
        """
        Gates the flow of information based on top-down expectation.
        - High Precision (Salient info): Lowers the threshold (Hyper-focus, signal passes easily).
        - Low Precision (Noise): Raises the threshold (Ignores the input, saves energy).
        """
        # 1. Ensure precision is strictly positive and mathematically stable
        safe_precision = FreeEnergyMath.softplus(precision)
        
        # 2. Dynamic Threshold Modulation (The physical mechanism of Attention)
        # Assuming safe_precision is roughly ~1.0 natively. 
        # Higher precision drops the integer threshold.
        dynamic_threshold = np.maximum(5, (self.base_threshold / (safe_precision + 1e-4)).astype(np.int32))
        
        # 3. Apply the gate
        self.snn.threshold = dynamic_threshold
        
        # 4. Route the signal via Zero-FLOP integer accumulation
        return self.snn.forward(in_spikes)
