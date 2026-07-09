// SPDX-License-Identifier: MIT
// Hippocampal-style replay engine.
//
// Wilson & McNaughton (1994): during slow-wave sleep the hippocampus replays
// recent place-cell sequences in reverse and at high speed, driving cortical
// plasticity. CortexOS emulates this with a ring buffer of AER packets:
//
//   * The buffer accumulates spikes during "wake" (last N seconds).
//   * During sleep(T), the scheduler replays this buffer in reverse at
//     acceleration factor A (e.g. 20×), feeding the packets straight into
//     the STDP kernel — no external drive.
//
// The result: consolidations of correlated wake activity into long-term
// weight changes, with `da > 0` (the reward tag stored in the packet's
// meta nibble) selectively strengthening rewarded sequences.

#include <cuda_runtime.h>
#include <cstdint>
#include "agi/core.hpp"

namespace agi::cuda {

// Ring buffer of AER packets sorted by emission time.
struct ReplayBuffer {
    SpikePacket* packets;
    std::uint64_t* t_ms;     // emission time per packet (µs granularity fine)
    std::uint32_t capacity;
    std::uint32_t head;      // write index
    std::uint32_t size;
};

__global__ void replay_emit_kernel(const ReplayBuffer buf,
                                   std::uint64_t t_start_us,
                                   std::uint64_t t_end_us,
                                   float speed_up,
                                   SpikePacket* __restrict__ out,
                                   std::uint32_t* __restrict__ out_count,
                                   std::uint32_t out_capacity) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buf.size) return;

    // Iterate in *reverse* order (SWS replay is often reverse-ordered).
    const std::uint32_t j = buf.size - 1 - i;
    const std::uint64_t t = buf.t_ms[j];
    if (t < t_start_us || t >= t_end_us) return;

    const std::uint32_t slot = atomicAdd(out_count, 1u);
    if (slot < out_capacity) {
        SpikePacket p = buf.packets[j];
        // Compress delay by speed_up: replay 20× faster ⇒ delay/=20.
        p.delay_slots = static_cast<std::uint8_t>(
            max(1, int(p.delay_slots) / max(1, int(speed_up))));
        // Mark as replay in meta nibble (high bit).
        p.weight_meta |= 0x80;
        out[slot] = p;
    }
}

extern "C" cudaError_t launch_replay_emit(const ReplayBuffer buf,
                                          std::uint64_t t_start_us,
                                          std::uint64_t t_end_us,
                                          float speed_up,
                                          SpikePacket* d_out,
                                          std::uint32_t* d_out_count,
                                          std::uint32_t out_capacity,
                                          cudaStream_t stream) {
    if (buf.size == 0) return cudaSuccess;
    cudaMemsetAsync(d_out_count, 0, sizeof(std::uint32_t), stream);
    constexpr std::uint32_t block = 256;
    const std::uint32_t grid = (buf.size + block - 1) / block;
    replay_emit_kernel<<<grid, block, 0, stream>>>(
        buf, t_start_us, t_end_us, speed_up, d_out, d_out_count, out_capacity);
    return cudaGetLastError();
}

}  // namespace agi::cuda
