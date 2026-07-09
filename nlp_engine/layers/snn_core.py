import numpy as np

class EnergyEfficientSNNLayer:
    """
    Zero-FLOP Event-Driven Spiking Layer for the CAIN Foundation.
    Achieves extreme energy efficiency by replacing dense Matrix Multiplications
    with sparse integer accumulation.
    """
    def __init__(self, num_in, num_out, batch_size=1, threshold=100):
        # Integer weights avoid floating-point ALU energy costs
        self.weights = np.random.randint(-10, 10, size=(num_in, num_out), dtype=np.int16)
        
        # Pre-allocated membrane potentials (int32) for zero-allocation runtime
        self.membrane = np.zeros((batch_size, num_out), dtype=np.int32)
        self.threshold = threshold

    def forward(self, in_spikes):
        """
        in_spikes: boolean or binary array of shape (batch_size, num_in)
        """
        # Efficient Integer Leak decay using bitshift (~87.5% retention)
        self.membrane -= (self.membrane >> 3)

        # Event-Driven Masking: Find exactly which neurons fired.
        batch_idx, spike_idx = np.nonzero(in_spikes)

        # Zero-FLOP Sparse Accumulation
        if spike_idx.size > 0:
            # np.add.at natively accumulates weight rows into the membrane potentials
            # WITHOUT performing any dense matrix multiplications (Zero FLOPs).
            np.add.at(self.membrane, batch_idx, self.weights[spike_idx])

        # Firing Generation
        out_spikes = self.membrane >= self.threshold

        # Soft Reset: Retain residual voltage for asynchronous stability
        self.membrane[out_spikes] -= self.threshold

        return out_spikes
