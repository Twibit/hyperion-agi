#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265358979323846

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// ----------------------------------------------------------------------------
// 1. FAST FOURIER TRANSFORM (FFT) FOR DIGITAL COCHLEA
// ----------------------------------------------------------------------------
typedef struct {
    double real;
    double imag;
} Complex;

void _fft_recursive(Complex* x, int N) {
    if (N <= 1) return;
    
    Complex* even = (Complex*)malloc(N / 2 * sizeof(Complex));
    Complex* odd = (Complex*)malloc(N / 2 * sizeof(Complex));
    
    for (int i = 0; i < N / 2; i++) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }
    
    _fft_recursive(even, N / 2);
    _fft_recursive(odd, N / 2);
    
    for (int k = 0; k < N / 2; k++) {
        double t_real = cos(-2.0 * PI * k / N) * odd[k].real - sin(-2.0 * PI * k / N) * odd[k].imag;
        double t_imag = sin(-2.0 * PI * k / N) * odd[k].real + cos(-2.0 * PI * k / N) * odd[k].imag;
        
        x[k].real = even[k].real + t_real;
        x[k].imag = even[k].imag + t_imag;
        
        x[k + N / 2].real = even[k].real - t_real;
        x[k + N / 2].imag = even[k].imag - t_imag;
    }
    
    free(even);
    free(odd);
}

// Exported function for Python ctypes
EXPORT void compute_fft(double* input_data, double* output_magnitudes, int N) {
    Complex* x = (Complex*)malloc(N * sizeof(Complex));
    for (int i = 0; i < N; i++) {
        x[i].real = input_data[i];
        x[i].imag = 0.0;
    }
    
    _fft_recursive(x, N);
    
    for (int i = 0; i < N / 2; i++) {
        output_magnitudes[i] = sqrt(x[i].real * x[i].real + x[i].imag * x[i].imag);
    }
    
    free(x);
}

// ----------------------------------------------------------------------------
// 2. 100-TRILLION PARAMETER VIRTUAL SUBSTRATE (FNV-1a HASH)
// ----------------------------------------------------------------------------
EXPORT uint64_t hash_neuron_id(const char* concept_str, const char* domain_str, int offset) {
    // FNV-1a 64-bit hash
    uint64_t hash = 14695981039346656037ULL;
    
    // Hash concept
    for (int i = 0; concept_str[i] != '\0'; i++) {
        hash ^= (uint64_t)(unsigned char)concept_str[i];
        hash *= 1099511628211ULL;
    }
    
    // Hash domain
    for (int i = 0; domain_str[i] != '\0'; i++) {
        hash ^= (uint64_t)(unsigned char)domain_str[i];
        hash *= 1099511628211ULL;
    }
    
    // Hash offset (for sparsity generation)
    hash ^= (uint64_t)offset;
    hash *= 1099511628211ULL;
    
    // Modulo 100 Trillion (100,000,000,000,000)
    return hash % 100000000000000ULL;
}

// ----------------------------------------------------------------------------
// 3. TRIPARTITE SYNAPSE & ASTROCYTE CALCIUM (BIOLOGY)
// ----------------------------------------------------------------------------
EXPORT double compute_astrocyte_calcium(double current_calcium, double pre_synaptic_spike, double post_synaptic_spike, double dopamine) {
    // Astrocytes integrate spikes and decay over time
    double decay_rate = 0.95;
    double calcium_influx = (pre_synaptic_spike * post_synaptic_spike) * 0.1;
    
    // Dopamine modulates the calcium channel sensitivity
    double modulated_influx = calcium_influx * dopamine;
    
    double new_calcium = (current_calcium * decay_rate) + modulated_influx;
    if (new_calcium > 1.0) new_calcium = 1.0;
    
    return new_calcium;
}

// ----------------------------------------------------------------------------
// 4. THERMODYNAMIC ATP BURN
// ----------------------------------------------------------------------------
EXPORT double calculate_atp_drain(double current_atp, int active_synapses, double cognitive_load) {
    // Each active synapse burns exactly 0.0001 ATP molecules
    double metabolic_cost = (double)active_synapses * 0.0001;
    
    // Spiking heavily (high load) multiplies the burn
    metabolic_cost *= (1.0 + cognitive_load);
    
    return current_atp - metabolic_cost;
}
