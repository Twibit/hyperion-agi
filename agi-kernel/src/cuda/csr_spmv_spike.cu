// SPDX-License-Identifier: MIT
// Spike-driven CSR SpMV.
//
// Semantics:   syn_current[i] += Σ_j  LUT[w_q8_ij] · spike_bit_j
//
// The pre-synaptic input is a packed bit-vector (one bit per source neuron),
// which is the natural product of the ballot reduction in hh_step_kernel /
// izh_step_kernel. Weights are 8-bit log-quantised (Song et al. 2005) and
// dequantised via a 256-entry constant-memory LUT.
//
// Strategy: one warp per row. Warp collectively iterates over that row's
// non-zeros. `__ldg` on the bit-vector minimizes global bandwidth (each word
// covers 32 source neurons).

#include <cuda_runtime.h>
#include <cstdint>

namespace agi::cuda {

__constant__ float d_weight_lut[256];

__global__ void csr_spmv_spike_kernel(const std::uint32_t* __restrict__ row_ptr,
                                      const std::uint32_t* __restrict__ col_idx,
                                      const std::uint8_t*  __restrict__ w_q8,
                                      const std::uint32_t* __restrict__ spike_bits,
                                      float*               __restrict__ syn_current,
                                      std::uint32_t rows) {
    const std::uint32_t row = blockIdx.x * (blockDim.y) + threadIdx.y;
    if (row >= rows) return;

    const std::uint32_t lane = threadIdx.x;                  // 0..31
    const std::uint32_t begin = row_ptr[row];
    const std::uint32_t end   = row_ptr[row + 1];

    float sum = 0.0f;
    for (std::uint32_t p = begin + lane; p < end; p += 32) {
        const std::uint32_t col = col_idx[p];
        const std::uint32_t word = spike_bits[col >> 5];
        const std::uint32_t bit  = (word >> (col & 31u)) & 1u;
        if (bit) {
            sum += d_weight_lut[w_q8[p]];
        }
    }
    // Warp-wide reduction.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFu, sum, off);
    }
    if (lane == 0) {
        // Accumulate rather than overwrite: the same row may receive from
        // both the local CSR and the long-range/hub CSR.
        atomicAdd(&syn_current[row], sum);
    }
}

extern "C" cudaError_t launch_csr_spmv_spike(const std::uint32_t* d_row_ptr,
                                             const std::uint32_t* d_col_idx,
                                             const std::uint8_t*  d_w_q8,
                                             const std::uint32_t* d_spike_bits,
                                             float* d_syn_current,
                                             std::uint32_t rows,
                                             cudaStream_t stream) {
    dim3 block(32, 8);  // one warp per row, 8 rows per block
    const std::uint32_t grid = (rows + block.y - 1) / block.y;
    csr_spmv_spike_kernel<<<grid, block, 0, stream>>>(d_row_ptr, d_col_idx, d_w_q8,
                                                      d_spike_bits, d_syn_current, rows);
    return cudaGetLastError();
}

// Host-side LUT initialiser. Log-normal spacing: w = exp(mu + sigma·Z)
// where Z ∈ [-3, 3] is mapped from the 8-bit index. Matches the fat-tailed
// weight distribution reported by Song et al. 2005.
extern "C" void set_weight_lut(float mu, float sigma) {
    float host_lut[256];
    for (int k = 0; k < 256; ++k) {
        const float z = -3.0f + 6.0f * (float(k) / 255.0f);
        host_lut[k] = expf(mu + sigma * z);
    }
    cudaMemcpyToSymbol(d_weight_lut, host_lut, sizeof(host_lut));
}

}  // namespace agi::cuda
