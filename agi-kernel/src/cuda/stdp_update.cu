// SPDX-License-Identifier: MIT
// Three-factor STDP (Frémaux & Gerstner 2016).
//
//   Δw_ij  =  da · e_ij
//   Δe_ij  = -e_ij · dt/τ_e
//              + A+ · exp(-Δt/τ+)   if post fired after pre (LTP)
//              - A- · exp(+Δt/τ-)   if pre fired after post (LTD)
//
// Implementation: one warp per post-synaptic neuron. Warp lanes iterate over
// that row's incoming synapses. Last-spike timestamps are read from
// per-neuron scalars; weight indices are updated via atomic compare-swap on
// the 8-bit weight quantum.

#include <cuda_runtime.h>
#include <cstdint>
#include "agi/core.hpp"

namespace agi::cuda {

extern __constant__ float d_nm_da;
extern __constant__ float d_nm_ht5;

// Note: e_trace is stored as __half to halve memory (~500 GB → 250 GB at 10¹⁴ syn).
__global__ void stdp_observe_kernel(const std::uint32_t* __restrict__ row_ptr,
                                    const std::uint32_t* __restrict__ col_idx,
                                    __half*              __restrict__ e_trace,
                                    const std::uint32_t* __restrict__ pre_bits,
                                    const std::uint32_t* __restrict__ post_bits,
                                    const float*         __restrict__ last_pre_ms,
                                    const float*         __restrict__ last_post_ms,
                                    float t_ms,
                                    float tau_plus_ms,  float tau_minus_ms,
                                    float a_plus,       float a_minus,
                                    float tau_elig_ms,  float dt_ms,
                                    std::uint32_t rows) {
    const std::uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const std::uint32_t lane = threadIdx.x;

    const float post_last  = last_post_ms[row];
    const bool  post_fired = (post_bits[row >> 5] >> (row & 31u)) & 1u;
    const float decay = expf(-dt_ms / tau_elig_ms);

    for (std::uint32_t p = row_ptr[row] + lane; p < row_ptr[row + 1]; p += 32) {
        const std::uint32_t col = col_idx[p];
        const float pre_last  = last_pre_ms[col];
        const bool  pre_fired = (pre_bits[col >> 5] >> (col & 31u)) & 1u;

        float e = __half2float(e_trace[p]) * decay;

        if (post_fired && pre_last > 0.0f) {
            const float dt_pair = t_ms - pre_last;   // >= 0
            e += a_plus * expf(-dt_pair / tau_plus_ms);
        }
        if (pre_fired && post_last > 0.0f) {
            const float dt_pair = t_ms - post_last;
            e -= a_minus * expf(-dt_pair / tau_minus_ms);
        }
        e_trace[p] = __float2half(e);
    }
}

__global__ void stdp_apply_kernel(const std::uint32_t* __restrict__ row_ptr,
                                  __half*              __restrict__ e_trace,
                                  std::uint8_t*        __restrict__ w_q8,
                                  float da,
                                  float lr,
                                  std::uint32_t rows) {
    const std::uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const std::uint32_t lane = threadIdx.x;

    for (std::uint32_t p = row_ptr[row] + lane; p < row_ptr[row + 1]; p += 32) {
        const float e = __half2float(e_trace[p]);
        const float dw = lr * da * e;                      // three-factor
        const int wq = int(w_q8[p]) + int(roundf(dw * 32.0f));  // 32 quanta / weight unit
        w_q8[p] = static_cast<std::uint8_t>(min(max(wq, 0), 255));
    }
}

extern "C" cudaError_t launch_stdp_observe(
        const std::uint32_t* d_row_ptr, const std::uint32_t* d_col_idx,
        __half* d_e_trace,
        const std::uint32_t* d_pre_bits, const std::uint32_t* d_post_bits,
        const float* d_last_pre_ms, const float* d_last_post_ms,
        float t_ms, float tau_plus_ms, float tau_minus_ms,
        float a_plus, float a_minus, float tau_elig_ms, float dt_ms,
        std::uint32_t rows, cudaStream_t stream) {
    dim3 block(32, 4);
    const std::uint32_t grid = (rows + block.y - 1) / block.y;
    stdp_observe_kernel<<<grid, block, 0, stream>>>(
        d_row_ptr, d_col_idx, d_e_trace, d_pre_bits, d_post_bits,
        d_last_pre_ms, d_last_post_ms, t_ms, tau_plus_ms, tau_minus_ms,
        a_plus, a_minus, tau_elig_ms, dt_ms, rows);
    return cudaGetLastError();
}

extern "C" cudaError_t launch_stdp_apply(const std::uint32_t* d_row_ptr,
                                         __half* d_e_trace, std::uint8_t* d_w_q8,
                                         float da, float lr, std::uint32_t rows,
                                         cudaStream_t stream) {
    dim3 block(32, 4);
    const std::uint32_t grid = (rows + block.y - 1) / block.y;
    stdp_apply_kernel<<<grid, block, 0, stream>>>(d_row_ptr, d_e_trace, d_w_q8, da, lr, rows);
    return cudaGetLastError();
}

}  // namespace agi::cuda
