// SPDX-License-Identifier: MIT
// Izhikevich (2003) integrator — CUDA, 32 neurons per warp.
//
// Two-variable dynamics:
//   dv/dt = 0.04 v² + 5v + 140 − u + I
//   du/dt = a (b v − u)
//   if v ≥ 30 mV: spike, v ← c, u ← u + d
//
// Uses semi-implicit Euler with dt = 0.5 ms internally (2 substeps per
// outer dt = 0.1 ms is optional; 4 substeps at 0.025 ms is safer near
// bifurcations).

#include <cuda_runtime.h>
#include <cstdint>
#include "agi/neuron.hpp"

namespace agi::cuda {

extern __constant__ float d_nm_da;
extern __constant__ float d_nm_ne;
extern __constant__ float d_nm_ach;
extern __constant__ float d_nm_ht5;

__device__ __forceinline__ bool izh_substep(IzhState& s, const IzhParams& p,
                                            float i_in, float h) {
    // Semi-implicit Euler on v: numerically stable near the parabolic nullcline.
    const float dv = 0.04f * s.v * s.v + 5.0f * s.v + 140.0f - s.u + i_in;
    s.v += h * dv;
    const float du = p.a * (p.b * s.v - s.u);
    s.u += h * du;
    const bool spiked = (s.v >= 30.0f);
    if (spiked) {
        s.v = p.c;
        s.u += p.d;
    }
    return spiked;
}

__global__ void izh_step_kernel(IzhState* __restrict__ state,
                                const IzhParams* __restrict__ params,
                                const float* __restrict__ syn_current,
                                const float* __restrict__ i_ext,
                                std::uint32_t* __restrict__ spike_bits,
                                float dt_ms,
                                std::uint32_t n_neurons) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;

    IzhState s = state[i];
    IzhParams p = params[i];

    // Cholinergic lowering of spike threshold: nudge b upward slightly.
    p.b += 0.02f * d_nm_ach;

    const float i_in = syn_current[i] + i_ext[i];

    // 2 substeps of 0.5·dt each (dt = 0.1 ms → h = 0.05 ms).
    bool spiked = false;
    const float h = dt_ms * 0.5f;
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        spiked |= izh_substep(s, p, i_in, h);
    }

    state[i] = s;

    const std::uint32_t mask = __ballot_sync(0xFFFFFFFFu, spiked);
    if ((i & 31u) == 0u) {
        spike_bits[i >> 5] = mask;
    }
}

extern "C" cudaError_t launch_izh_step(IzhState* d_state, const IzhParams* d_params,
                                       const float* d_syn_current, const float* d_i_ext,
                                       std::uint32_t* d_spike_bits, float dt_ms,
                                       std::uint32_t n_neurons, cudaStream_t stream) {
    constexpr std::uint32_t block = 256;
    const std::uint32_t grid = (n_neurons + block - 1) / block;
    izh_step_kernel<<<grid, block, 0, stream>>>(d_state, d_params, d_syn_current, d_i_ext,
                                                d_spike_bits, dt_ms, n_neurons);
    return cudaGetLastError();
}

}  // namespace agi::cuda
