import numpy as np

def _sigmoid(x):
    x_c = np.clip(x, -10.0, 10.0)
    return np.where(x_c >= 0, 1.0 / (1.0 + np.exp(-x_c)), np.exp(x_c) / (1.0 + np.exp(x_c)))

def _quantize_symmetric(matrix_f16: np.ndarray) -> tuple[np.ndarray, np.float16]:
    """Quantizes a float16 matrix to int8 using symmetric scaling."""
    max_val = np.max(np.abs(matrix_f16))
    if max_val == 0:
        return np.zeros_like(matrix_f16, dtype=np.int8), np.float16(1.0)
    scale = max_val / 127.0
    matrix_int8 = np.clip(np.round(matrix_f16 / scale), -127, 127).astype(np.int8)
    return matrix_int8, np.float16(scale)

def _dequantize_symmetric(matrix_int8: np.ndarray, scale: np.float16) -> np.ndarray:
    """Dequantizes an int8 matrix back to float16."""
    return (matrix_int8.astype(np.float16) * scale)

class ComplexPyramidalMemoryNeuron:
    """
    Biological Linear Attention (Complex Pyramidal Memory Neuron).
    
    Upgrades:
    1. Dual-Compartment: Separates Basal (sensory input) from Apical (recurrent context/query)
    2. Epigenetic Metaplasticity (E_t): Dynamic decay based on synaptic usage.
    3. Neuromodulatory Gating: Dopamine controls the learning rate explicitly.
    """
    def __init__(self, d_in: int, d_k: int = 86, d_v: int = 86, 
                 lambda_base: float = 0.90, prune_threshold: float = 0.05, seed: int | None = None):
        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v
        self.lambda_base = np.float16(lambda_base)
        self.prune_threshold = np.float16(prune_threshold)
        
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_in)
        
        # Basal Dendrites (Sensory Input -> Key, Value)
        self.W_k = (rng.standard_normal((d_in, d_k)) * s).astype(np.float16)
        self.W_v = (rng.standard_normal((d_in, d_v)) * s).astype(np.float16)
        
        # Apical Dendrites (Context Input -> Query)
        self.W_q = (rng.standard_normal((d_in, d_k)) * s).astype(np.float16)
        
        # Somatic Readout
        self.W_r = (rng.standard_normal((d_v, d_in)) * s).astype(np.float16)
        self.W_x = (rng.standard_normal((d_in, d_in)) * s).astype(np.float16)
        self.b_out = np.zeros(d_in, dtype=np.float16)
        
        self.reset()
        
    def reset(self):
        # M_t: Dendritic Memory Matrix (INT8 Quantized, ~7.3 KB)
        self.M_t_int8 = np.zeros((self.d_v, self.d_k), dtype=np.int8)
        self.M_t_scale = np.float16(1.0)
        
        # E_t: Epigenetic State Matrix (INT8 Quantized, ~7.3 KB)
        # Tracks synaptic durability. High E_t means the synapse resists decay.
        self.E_t_int8 = np.zeros((self.d_v, self.d_k), dtype=np.int8)
        self.E_t_scale = np.float16(1.0)
        
        # BCM Theory Sliding Threshold
        self.theta_M = np.float16(0.0)
        
        # Recurrent state for apical querying
        self.x_prev = np.zeros(self.d_in, dtype=np.float16)
        
    def step(self, x_basal: np.ndarray, x_apical: np.ndarray = None, dopamine: float = 1.0) -> np.ndarray:
        """
        Processes an input with biological mechanics.
        x_basal: The immediate sensory input to the neuron.
        x_apical: The top-down context. If None, it recurrently uses x_prev.
        dopamine: Modulates learning rate. 0.0 = no learning, 1.0 = normal learning, >1.0 = hyper-plasticity.
        """
        x_basal_f16 = x_basal.astype(np.float16)
        if x_apical is None:
            x_apical_f16 = self.x_prev
        else:
            x_apical_f16 = x_apical.astype(np.float16)
            
        # 1. Dendritic Arbor Mapping
        # Basal (Sensory) -> Key, Value
        k_t = _sigmoid(x_basal_f16 @ self.W_k)
        v_t = np.tanh(x_basal_f16 @ self.W_v)
        
        # Apical (Context) -> Query
        q_t = _sigmoid(x_apical_f16 @ self.W_q)
        
        # Dequantize from INT8
        M_t_f16 = _dequantize_symmetric(self.M_t_int8, self.M_t_scale)
        E_t_f16 = _dequantize_symmetric(self.E_t_int8, self.E_t_scale)
        
        # 2. Predictive Coding (Surprisal Error)
        # Only learn if apical context failed to predict the basal key
        epsilon_k = k_t - q_t
        
        # 3. BCM Theory (Homeostatic Stability)
        # Calculate excitation magnitude
        activity = np.mean(np.abs(x_basal_f16))
        phi = activity * (activity - self.theta_M)
        self.theta_M = np.float16(0.99 * self.theta_M + 0.01 * (activity ** 2))
        
        # 4. Epigenetic Metaplasticity
        E_t_sig = _sigmoid(E_t_f16)
        Lambda_t = self.lambda_base + (1.0 - self.lambda_base) * E_t_sig
        
        # 5. Trinity Update (Write)
        learning_signal = np.float16(dopamine * phi)
        trinity_update = np.outer(v_t, epsilon_k).astype(np.float16)
        
        # Epigenetics harden strictly based on absolute learning magnitude
        E_t_f16 = (E_t_f16 * np.float16(0.99)) + (np.abs(trinity_update) * np.float16(0.1 * np.abs(learning_signal)))
        
        # Memory updates bidirectionally (LTP/LTD) depending on BCM and PC Error
        M_t_f16 = (M_t_f16 * Lambda_t) + (trinity_update * learning_signal)
        
        # Synaptic Pruning (Ultra-Compression)
        M_t_f16[np.abs(M_t_f16) < self.prune_threshold] = 0.0
        
        # Re-Quantize to INT8
        self.M_t_int8, self.M_t_scale = _quantize_symmetric(M_t_f16)
        self.E_t_int8, self.E_t_scale = _quantize_symmetric(E_t_f16)
        
        # 6. Continuous Hopfield Dynamics (Read)
        # Sharp Tanh creates a strong biological attractor basin
        r_t = np.tanh((M_t_f16 @ q_t) * 5.0)
        
        # 6. Somatic Output Blend
        raw_y = np.clip(r_t @ self.W_r + x_basal_f16 @ self.W_x + self.b_out, -10.0, 10.0)
        y_t = _sigmoid(raw_y)
        
        # Store for next recurrence
        self.x_prev = x_basal_f16
        
        return y_t.astype(x_basal.dtype)

    def forward(self, x_seq: np.ndarray, reset: bool = True, dopamine: float = 1.0) -> np.ndarray:
        if reset:
            self.reset()
        T = x_seq.shape[0]
        out = np.empty((T, self.d_in), dtype=x_seq.dtype)
        for t in range(T):
            out[t] = self.step(x_seq[t], dopamine=dopamine)
        return out
        
    def saturation(self) -> float:
        """Returns the percentage of Epigenetic synapses that are nearly frozen/saturated.
        If > 0.90, this neuron has reached its biological limit and requires Neurogenesis (Mitosis).
        """
        abs_E = np.abs(self.E_t_int8)
        return float(np.sum(abs_E > 115) / abs_E.size)
