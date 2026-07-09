import numpy as np
import wave
import struct
import os
import sys

# Ensure root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import rust_backend_shim

class AuditoryCortex:
    def __init__(self, output_dim=256):
        self.output_dim = output_dim
        print("[Auditory Cortex] Native Digital Cochlea online. Ready to process spectral frequencies via Rust PyO3.")

    def process_audio(self, audio_data, sample_rate=44100):
        """
        Converts raw audio waveform into biological frequency spikes.
        Now uses 100% native Python 'wave' and 'struct' (No Scipy).
        """
        if isinstance(audio_data, str):
            try:
                # Natively read the WAV file RIFF header and PCM data
                with wave.open(audio_data, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    n_channels = wav_file.getnchannels()
                    sampwidth = wav_file.getsampwidth()
                    n_frames = wav_file.getnframes()
                    
                    raw_frames = wav_file.readframes(n_frames)
                    
                    # Unpack bytes to integers natively based on sample width
                    if sampwidth == 2:
                        fmt = f"<{n_frames * n_channels}h"
                    elif sampwidth == 4:
                        fmt = f"<{n_frames * n_channels}i"
                    else:
                        fmt = f"<{n_frames * n_channels}b"
                        
                    data = np.array(struct.unpack(fmt, raw_frames))
                    
                    if n_channels > 1:
                        data = data.reshape(-1, n_channels).mean(axis=1) # Convert stereo to mono
                    audio_data = data
            except Exception as e:
                print(f"      [Auditory Error]: Native WAV parsing failed for {audio_data} - {e}")
                audio_data = np.random.randn(sample_rate)
                
        # Fast Fourier Transform (FFT) to extract frequencies natively using Rust
        fft_result = rust_backend_shim.compute_fft(audio_data.tolist())
        
        # Bin frequencies into biologically accurate cochlear spikes
        binned = np.zeros(self.output_dim)
        bin_size = len(fft_result) // self.output_dim
        
        if bin_size > 0:
            for i in range(self.output_dim):
                start = i * bin_size
                end = (i + 1) * bin_size
                binned[i] = np.mean(fft_result[start:end])
                
        threshold = np.mean(binned) + np.std(binned)
        spikes = binned > threshold
        return spikes
