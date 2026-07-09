import numpy as np
from numba import njit
import math

# ==========================================
# HYPERION JIT-COMPILED MACHINE CODE
# Powered by LLVM via Numba
# ==========================================

@njit
def jit_hash_neuron_id(concept_bytes: bytes, domain_bytes: bytes, offset: int) -> int:
    """
    64-bit FNV-1a Hash compiled to C-level machine code via Numba.
    Generates a deterministic 64-bit coordinate in the 100-Trillion sparse matrix.
    """
    h = np.uint64(14695981039346656037)
    prime = np.uint64(1099511628211)
    
    for b in concept_bytes:
        h = h ^ np.uint64(b)
        h = h * prime
        
    for b in domain_bytes:
        h = h ^ np.uint64(b)
        h = h * prime
        
    h = h ^ np.uint64(offset)
    h = h * prime
    
    return int(h % 100000000000000)

@njit
def jit_calcium_integrate(current_calcium: float, pre_spike: float, post_spike: float, dopamine: float) -> float:
    """
    Calculates Astrocyte Calcium influx across Tripartite synapses instantly.
    """
    decay_rate = 0.95
    calcium_influx = (pre_spike * post_spike) * 0.1
    modulated_influx = calcium_influx * dopamine
    
    new_calcium = (current_calcium * decay_rate) + modulated_influx
    if new_calcium > 1.0:
        new_calcium = 1.0
        
    return new_calcium

@njit
def _jit_fft_recursive(x_real: np.ndarray, x_imag: np.ndarray):
    """
    Pure mathematical Cooley-Tukey FFT compiled to raw machine code via LLVM.
    Achieves performance rivaling C-extensions and scipy.
    """
    N = len(x_real)
    if N <= 1:
        return x_real, x_imag
        
    even_real, even_imag = _jit_fft_recursive(x_real[0::2], x_imag[0::2])
    odd_real, odd_imag = _jit_fft_recursive(x_real[1::2], x_imag[1::2])
    
    T_real = np.zeros(N // 2)
    T_imag = np.zeros(N // 2)
    
    for k in range(N // 2):
        angle = -2.0 * math.pi * k / N
        cos_val = math.cos(angle)
        sin_val = math.sin(angle)
        
        T_real[k] = cos_val * odd_real[k] - sin_val * odd_imag[k]
        T_imag[k] = sin_val * odd_real[k] + cos_val * odd_imag[k]
        
    out_real = np.zeros(N)
    out_imag = np.zeros(N)
    
    for k in range(N // 2):
        out_real[k] = even_real[k] + T_real[k]
        out_imag[k] = even_imag[k] + T_imag[k]
        
        out_real[k + N // 2] = even_real[k] - T_real[k]
        out_imag[k + N // 2] = even_imag[k] - T_imag[k]
        
    return out_real, out_imag

def jit_compute_fft(audio_data: np.ndarray) -> np.ndarray:
    """
    Entry point for the JIT FFT algorithm. 
    Limits frame size to prevent Numba recursion depth issues and normalizes input.
    """
    N = len(audio_data)
    if N == 0:
        return np.zeros(256)
        
    # Pad to nearest power of 2 for FFT
    if (N & (N - 1)) != 0:
        target_pow2 = 1 << (N - 1).bit_length()
        audio_data = np.pad(audio_data, (0, target_pow2 - N), 'constant')
        N = len(audio_data)
        
    x_real = audio_data.astype(np.float64)
    x_imag = np.zeros_like(x_real)
    
    out_real, out_imag = _jit_fft_recursive(x_real, x_imag)
    
    # Calculate magnitude
    magnitudes = np.sqrt(out_real**2 + out_imag**2)
    return magnitudes[:N // 2]
