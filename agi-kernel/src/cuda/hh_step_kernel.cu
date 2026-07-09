// SPDX-License-Identifier: MIT
// Hodgkin–Huxley (1952) integrator — CUDA, one warp per neuron block.
//
// Numerical method: classical Runge–Kutta 4, 4 sub-steps per outer dt
// (so effective substep = 0.025 ms at dt = 0.1 ms, well within stability).
// One thread per neuron, 32 neurons per warp, 8 warps per block.
//
// Ported from neurorust::HodgkinHuxleyNeuron.

#include <cuda_runtime.h>
#include <cstdint>
#include "agi/neuron.hpp"
#include "agi/core.hpp"

namespace agi::cuda {

// ---------- Rate constants (Hodgkin & Huxley 1952) --------------------------
__device__ __forceinline__ float alpha_m(float v) {
    // Handle removable singularity at v = -40 mV.
    float d = v + 40.0f;
    if (fabsf(d) < 1e-6f) return 1.0f;
    return 0.1f * d / (1.0f - expf(-d / 10.0f));
}
__device__ __forceinline__ float beta_m(float v)  { return 4.0f  * expf(-(v + 65.0f) / 18.0f); }
__device__ __forceinline__ float alpha_h(float v) { return 0.07f * expf(-(v + 65.0f) / 20.0f); }
__device__ __forceinline__ float beta_h(float v)  { return 1.0f  / (1.0f + expf(-(v + 35.0f) / 10.0f)); }
__device__ __forceinline__ float alpha_n(float v) {
    float d = v + 55.0f;
    if (fabsf(d) < 1e-6f) return 0.1f;
    return 0.01f * d / (1.0f - expf(-d / 10.0f));
}
__device__ __forceinline__ float beta_n(float v)  { return 0.125f * expf(-(v + 65.0f) / 80.0f); }

struct HHDerivs { float dv, dm, dh, dn; };

__device__ __forceinline__ HHDerivs hh_rhs(const HHState& s, const HHParams& p, float i_inj) {
    const float i_na = p.g_na * s.m * s.m * s.m * s.h * (s.v - p.e_na);
    const float i_k  = p.g_k  * s.n * s.n * s.n * s.n * (s.v - p.e_k);
    const float i_l  = p.g_l                          * (s.v - p.e_l);
    HHDerivs d;
    d.dv = (i_inj - i_na - i_k - i_l) / p.c_m;
    d.dm = alpha_m(s.v) * (1.0f - s.m) - beta_m(s.v) * s.m;
    d.dh = alpha_h(s.v) * (1.0f - s.h) - beta_h(s.v) * s.h;
    d.dn = alpha_n(s.v) * (1.0f - s.n) - beta_n(s.v) * s.n;
    return d;
}

__device__ __forceinline__ HHState hh_add(const HHState& s, const HHDerivs& d, float h) {
    HHState o = s;
    o.v = s.v + h * d.dv;
    o.m = s.m + h * d.dm;
    o.h = s.h + h * d.dh;
    o.n = s.n + h * d.dn;
    return o;
}

__device__ __forceinline__ HHState hh_rk4_substep(const HHState& s, const HHParams& p,
                                                  float i_inj, float h) {
    HHDerivs k1 = hh_rhs(s,               p, i_inj);
    HHDerivs k2 = hh_rhs(hh_add(s, k1, h * 0.5f), p, i_inj);
    HHDerivs k3 = hh_rhs(hh_add(s, k2, h * 0.5f), p, i_inj);
    HHDerivs k4 = hh_rhs(hh_add(s, k3, h),        p, i_inj);
    HHState o = s;
    const float c = h / 6.0f;
    o.v = s.v + c * (k1.dv + 2.0f * k2.dv + 2.0f * k3.dv + k4.dv);
    o.m = s.m + c * (k1.dm + 2.0f * k2.dm + 2.0f * k3.dm + k4.dm);
    o.h = s.h + c * (k1.dh + 2.0f * k2.dh + 2.0f * k3.dh + k4.dh);
    o.n = s.n + c * (k1.dn + 2.0f * k2.dn + 2.0f * k3.dn + k4.dn);
    // clamp gating vars to [0, 1] to keep RK4 numerically nice
    o.m = fminf(fmaxf(o.m, 0.0f), 1.0f);
    o.h = fminf(fmaxf(o.h, 0.0f), 1.0f);
    o.n = fminf(fmaxf(o.n, 0.0f), 1.0f);
    return o;
}

// ---------------------------------------------------------------------------
// Kernel: one thread per neuron.
//
//   state[i]        — current HHState (in/out)
//   syn_current[i]  — accumulated synaptic current (µA/cm²)
//   i_ext[i]        — external drive
//   spike_bits[]    — packed output bitmask (1 bit / neuron), 32 neurons / word
//
// Also reads the noradrenergic rail `ne` from __constant__ memory to scale
// the leak conductance (arousal → higher gain, Aston-Jones 2005).
// ---------------------------------------------------------------------------

extern __constant__ float d_nm_da;
extern __constant__ float d_nm_ne;
extern __constant__ float d_nm_ach;
extern __constant__ float d_nm_ht5;

__global__ void hh_step_kernel(HHState* __restrict__ state,
                               const HHParams params,
                               const float* __restrict__ syn_current,
                               const float* __restrict__ i_ext,
                               std::uint32_t* __restrict__ spike_bits,
                               float dt_ms,
                               std::uint32_t n_neurons) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;

    HHState s = state[i];
    const float prev_v = s.v;

    // NE modulates gain: raising ne 0..1 shrinks g_L by up to 30 %
    // (matches Aston-Jones 2005 arousal-gain effect).
    HHParams p = params;
    p.g_l *= (1.0f - 0.3f * d_nm_ne);
    // ACh lowers spike threshold implicitly via a small g_l bias.
    p.g_l -= 0.05f * d_nm_ach;

    const float i_in = syn_current[i] + i_ext[i];

    // 4 RK4 substeps per outer dt.
    const float h = dt_ms * 0.25f;
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        s = hh_rk4_substep(s, p, i_in, h);
    }

    // Spike detection = up-crossing of 0 mV.
    const bool spiked = (prev_v < 0.0f) && (s.v >= 0.0f);
    s.prev_v = prev_v;
    state[i] = s;

    // Reduce warp-wide into a 32-bit ballot; lane 0 stores.
    const std::uint32_t mask = __ballot_sync(0xFFFFFFFFu, spiked);
    if ((i & 31u) == 0u) {
        spike_bits[i >> 5] = mask;
    }
}

// Host-side launch wrapper (called by column.cpp).
extern "C" cudaError_t launch_hh_step(HHState* d_state, HHParams params,
                                      const float* d_syn_current, const float* d_i_ext,
                                      std::uint32_t* d_spike_bits, float dt_ms,
                                      std::uint32_t n_neurons, cudaStream_t stream) {
    constexpr std::uint32_t block = 256;
    const std::uint32_t grid = (n_neurons + block - 1) / block;
    hh_step_kernel<<<grid, block, 0, stream>>>(d_state, params, d_syn_current, d_i_ext,
                                               d_spike_bits, dt_ms, n_neurons);
    return cudaGetLastError();
}

}  // namespace agi::cuda
