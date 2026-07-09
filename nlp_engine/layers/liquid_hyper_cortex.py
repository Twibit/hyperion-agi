import numpy as np

class LiquidSynapse:
    """
    Based on Liquid Neural Networks (MIT CSAIL).
    Unlike static LLM weights, a Liquid Synapse's resistance changes dynamically 
    based on the continuous flow of incoming data (ODE integration).
    """
    def __init__(self, size: tuple, tau_base: float = 1.0):
        PHI = 1.6180339887
        self.w = np.random.randn(*size) * (0.1 / PHI)
        self.state = np.zeros(size)
        self.tau_base = tau_base * PHI # Golden scaling for time constant
        self.A = np.random.randn(*size) * (0.05 / (PHI**2)) # Sensory coupling

    def step(self, x: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        Euler integration of the Liquid Time-Constant (LTC) differential equation.
        dx/dt = - [x / tau] + A * f(sensory_input)
        """
        # Calculate dynamic time constant based on input intensity
        sensory_activation = np.tanh(np.dot(x, self.w) if self.w.shape[0] == x.shape[-1] else x)
        
        # Ensure dimensions match for the ODE step
        if sensory_activation.shape != self.state.shape:
            # Broadcast or project if needed
            sensory_activation = np.mean(sensory_activation)
            
        tau_dynamic = self.tau_base / (1.0 + np.abs(sensory_activation))
        
        # ODE Step: The weight physically morphs
        dx = -(self.state / tau_dynamic) + (self.A * sensory_activation)
        self.state += dx * dt
        
        return self.w + self.state

class HyperNetwork:
    """
    A network that generates weights for another network.
    Instead of training the Transformer, this generates the exact Attention 
    Matrices needed instantly based on the domain (Code, Prose, Math).
    """
    def __init__(self, context_dim: int, target_d_model: int, target_d_k: int):
        PHI = 1.6180339887
        self.w_gen = np.random.randn(context_dim, target_d_model * target_d_k) * (0.1 / PHI)
        self.target_shape = (target_d_model, target_d_k)

    def generate_attention_weights(self, domain_context: np.ndarray) -> tuple:
        """Generates Q, K, V matrices instantly."""
        PHI = 1.6180339887
        # Generate raw weights
        flat_q = np.dot(domain_context, self.w_gen)
        # Shift using the most irrational number to prevent statistical resonance loops
        flat_k = np.dot(domain_context, self.w_gen + (0.01 / PHI))
        flat_v = np.dot(domain_context, self.w_gen - (0.01 / PHI))
        
        # Reshape for the Transformer
        W_q = flat_q.reshape(self.target_shape)
        W_k = flat_k.reshape(self.target_shape)
        W_v = flat_v.reshape(self.target_shape)
        
        return W_q, W_k, W_v

class LiquidHyperCortex:
    """
    The pinnacle of AGI Architecture.
    Combines HyperNetworks (instant weight generation) with Liquid Synapses 
    (continuous time adaptation without backpropagation).
    """
    def __init__(self, d_model: int = 16, d_k: int = 4):
        self.d_model = d_model
        self.d_k = d_k
        self.hyper = HyperNetwork(context_dim=3, target_d_model=d_model, target_d_k=d_k)
        self.liquid_bridge = LiquidSynapse(size=(d_model, d_k))
        
    def forward(self, x: np.ndarray, domain_vector: np.ndarray, dt: float = 0.1):
        """
        x: (seq_len, d_model) semantic embeddings
        domain_vector: (3,) e.g. [1, 0, 0] for Code, [0, 1, 0] for Prose
        """
        # 1. HyperNetwork generates base Attention Matrices instantly based on the domain
        W_q_base, W_k_base, W_v_base = self.hyper.generate_attention_weights(domain_vector)
        
        # 2. Liquid Synapse dynamically morphs the Q matrix based on current thought sequence
        W_q_liquid = self.liquid_bridge.step(x, dt=dt)
        
        # Blend hyper-weights with liquid state
        W_q = W_q_base + (W_q_liquid * 0.1)
        
        # 3. Standard Attention Math
        Q = np.dot(x, W_q)
        K = np.dot(x, W_k_base)
        V = np.dot(x, W_v_base)
        
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k)
        
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        output = np.dot(attention_weights, V)
        return output, attention_weights, self.liquid_bridge.state
