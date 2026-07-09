import ctypes
import os
import sys
import numpy as np

# Load the compiled DLL
core_dll_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "core", "core.dll")
try:
    core = ctypes.CDLL(core_dll_path)
    
    # Setup argument types for compute_fft
    core.compute_fft.argtypes = [
        ctypes.POINTER(ctypes.c_double), # input_data
        ctypes.POINTER(ctypes.c_double), # output_magnitudes
        ctypes.c_int                     # N
    ]
    
    # Setup argument types for hash_neuron_id
    core.hash_neuron_id.argtypes = [
        ctypes.c_char_p, # concept_str
        ctypes.c_char_p, # domain_str
        ctypes.c_int     # offset
    ]
    core.hash_neuron_id.restype = ctypes.c_uint64
    
    # Setup argument types for compute_astrocyte_calcium
    core.compute_astrocyte_calcium.argtypes = [
        ctypes.c_double, # current_calcium
        ctypes.c_double, # pre_synaptic_spike
        ctypes.c_double, # post_synaptic_spike
        ctypes.c_double  # dopamine
    ]
    core.compute_astrocyte_calcium.restype = ctypes.c_double
    
    # Setup argument types for calculate_atp_drain
    core.calculate_atp_drain.argtypes = [
        ctypes.c_double, # current_atp
        ctypes.c_int,    # active_synapses
        ctypes.c_double  # cognitive_load
    ]
    core.calculate_atp_drain.restype = ctypes.c_double
    
except Exception as e:
    print(f"[Fatal] Failed to load native C-Core. Did you run build_core.bat? Error: {e}")
    core = None

def native_hash_neuron(concept: str, domain: str, offset: int) -> int:
    if not core: return abs(hash(concept + domain + str(offset))) % 100000000000000
    return core.hash_neuron_id(concept.encode('utf-8'), domain.encode('utf-8'), offset)

def native_fft(audio_data: np.ndarray) -> np.ndarray:
    if not core: return np.zeros(len(audio_data) // 2)
    
    N = len(audio_data)
    # Convert numpy array to ctypes array
    InputArrayType = ctypes.c_double * N
    OutputArrayType = ctypes.c_double * (N // 2)
    
    input_arr = InputArrayType(*audio_data.astype(np.float64))
    output_arr = OutputArrayType()
    
    core.compute_fft(input_arr, output_arr, N)
    
    # Convert back to numpy array
    return np.array(output_arr)

def native_calcium(current: float, pre: float, post: float, dopamine: float) -> float:
    if not core: return (current * 0.95) + ((pre * post) * 0.1 * dopamine)
    return core.compute_astrocyte_calcium(current, pre, post, dopamine)

def native_atp_drain(current_atp: float, synapses: int, load: float) -> float:
    if not core: return current_atp - (synapses * 0.0001 * (1.0 + load))
    return core.calculate_atp_drain(current_atp, synapses, load)
