// SPDX-License-Identifier: MIT
// Basal-ganglia-style winner-take-all action selector.
//
// Redgrave et al. (1999): action channels compete via lateral inhibition
// through the striatum → globus pallidus loop; the highest-activity channel
// disinhibits its target motor circuit. CortexOS implements this as a
// softmax-with-temperature over per-channel firing counts.
//
//   T = 0.1 + 4.0 · (1 − ht5)   ← serotonin → patience → lower temperature
//   p_k ∝ exp(rate_k / T)
//   winner = sample(p)

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

namespace agi::cuda {

extern __constant__ float d_nm_ht5;

__global__ void wta_select_kernel(const float* __restrict__ rates,
                                  std::uint32_t n_channels,
                                  std::uint32_t* __restrict__ out_winner,
                                  std::uint64_t seed) {
    // Single-block kernel: block-reduce max, then softmax + sample.
    extern __shared__ float smem[];
    float* rate_shm = smem;                    // [n_channels]
    float* prob_shm = smem + n_channels;       // [n_channels]

    const std::uint32_t tid = threadIdx.x;
    const float T = 0.1f + 4.0f * (1.0f - d_nm_ht5);

    // Load & block-reduce max.
    float r = -1e30f;
    for (std::uint32_t k = tid; k < n_channels; k += blockDim.x) {
        rate_shm[k] = rates[k];
        r = fmaxf(r, rates[k]);
    }
    // Warp reduce max
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        r = fmaxf(r, __shfl_xor_sync(0xFFFFFFFFu, r, off));
    }
    __shared__ float max_r;
    if (tid == 0) max_r = r;
    __syncthreads();

    // Exponentiate & sum.
    float sum = 0.0f;
    for (std::uint32_t k = tid; k < n_channels; k += blockDim.x) {
        const float e = expf((rate_shm[k] - max_r) / T);
        prob_shm[k] = e;
        sum += e;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
    }
    __shared__ float total;
    if (tid == 0) total = sum;
    __syncthreads();

    if (tid == 0) {
        curandStatePhilox4_32_10_t rng;
        curand_init(seed, 0, 0, &rng);
        float u = curand_uniform(&rng) * total;
        float acc = 0.0f;
        std::uint32_t winner = 0;
        for (std::uint32_t k = 0; k < n_channels; ++k) {
            acc += prob_shm[k];
            if (acc >= u) { winner = k; break; }
        }
        *out_winner = winner;
    }
}

extern "C" cudaError_t launch_wta_select(const float* d_rates, std::uint32_t n_channels,
                                         std::uint32_t* d_out_winner, std::uint64_t seed,
                                         cudaStream_t stream) {
    const std::uint32_t block = 128;
    const std::size_t shmem = 2 * n_channels * sizeof(float);
    wta_select_kernel<<<1, block, shmem, stream>>>(d_rates, n_channels, d_out_winner, seed);
    return cudaGetLastError();
}

}  // namespace agi::cuda
