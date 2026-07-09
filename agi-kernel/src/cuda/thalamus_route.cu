// SPDX-License-Identifier: MIT
// Thalamo-cortical routing kernel.
//
// Every cortico-cortical spike passes through here before being enqueued in
// the AER ring. Operations per packet:
//   1. Look up per-source-column attentional gain g ∈ [0, 4].
//   2. Modulate the packet's weight-index by log2(g) (LUT is log-spaced).
//   3. In bursting mode (ne > 0.7), stochastically drop packets with p=0.5
//      to enforce phasic firing.
//   4. Optionally duplicate to a second area (pulvinar rule).

#include <cuda_runtime.h>
#include <cstdint>
#include <curand_kernel.h>
#include "agi/core.hpp"

namespace agi::cuda {

extern __constant__ float d_nm_ne;

// Pulvinar rules stored in constant memory (max 8 rules).
struct PulvinarRuleGpu {
    std::uint32_t from_area, gate_area, to_area;
    float gate_thresh_hz;
    bool  active;
};
__constant__ PulvinarRuleGpu d_pulvinar_rules[8];
__constant__ int d_n_pulvinar_rules;

__global__ void thalamus_route_kernel(SpikePacket* __restrict__ packets,
                                      std::uint32_t n_packets,
                                      const float* __restrict__ attention,
                                      std::uint32_t n_source_columns,
                                      std::uint64_t seed) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_packets) return;

    SpikePacket p = packets[i];
    const std::uint32_t src_column = p.neuron_id / 10000u;   // 10 000 neurons/column
    const float gain = (src_column < n_source_columns) ? attention[src_column] : 1.0f;

    // Modulate weight index: log2(gain) mapped to +/- 32 quanta.
    int wq = int(p.weight_meta & 0x0F);
    int adj = int(roundf(log2f(fmaxf(gain, 1e-3f)) * 4.0f));
    wq = min(max(wq + adj, 0), 15);
    p.weight_meta = static_cast<std::uint8_t>((p.weight_meta & 0xF0) | wq);

    // Bursting mode gating: high `ne` → stochastic drop for phasic firing.
    if (d_nm_ne > 0.7f) {
        curandStatePhilox4_32_10_t rng;
        curand_init(seed, i, 0, &rng);
        if (curand_uniform(&rng) < 0.5f) {
            p.neuron_id = 0xFFFFFFFFu;   // mark as dropped
        }
    }

    packets[i] = p;
}

extern "C" cudaError_t launch_thalamus_route(SpikePacket* d_packets,
                                             std::uint32_t n_packets,
                                             const float* d_attention,
                                             std::uint32_t n_source_columns,
                                             std::uint64_t seed,
                                             cudaStream_t stream) {
    if (n_packets == 0) return cudaSuccess;
    constexpr std::uint32_t block = 256;
    const std::uint32_t grid = (n_packets + block - 1) / block;
    thalamus_route_kernel<<<grid, block, 0, stream>>>(d_packets, n_packets, d_attention,
                                                      n_source_columns, seed);
    return cudaGetLastError();
}

}  // namespace agi::cuda
