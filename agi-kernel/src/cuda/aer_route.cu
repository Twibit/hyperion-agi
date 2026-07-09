// SPDX-License-Identifier: MIT
// Address-Event Representation router.
//
// A spike emitted at step T with delay D is delivered at step T+D. The AER
// ring is `R` slots deep (typical R = 256), with each slot a bucket holding
// all packets that must fire at that future step.
//
// Layout:
//   ring[bucket] : { count, capacity, packets[] }
//
// Enqueue is a single atomicAdd on the bucket's count, followed by a write
// to the packet slot. Delivery is a warp-wide scan of the current bucket.

#include <cuda_runtime.h>
#include <cstdint>
#include "agi/core.hpp"

namespace agi::cuda {

// Bucket header + packets are stored back-to-back:
//   header[0..R-1]         : uint32 count
//   packets[0..R-1][cap]   : SpikePacket
struct AerRing {
    std::uint32_t*  header;    // R uint32 (counts, atomically incremented)
    SpikePacket*    packets;   // R * capacity packets
    std::uint32_t   ring_size; // R
    std::uint32_t   capacity;  // per-bucket capacity
};

__global__ void aer_enqueue_kernel(AerRing ring,
                                   const SpikePacket* __restrict__ src,
                                   std::uint32_t n,
                                   std::uint32_t base_bucket) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    SpikePacket p = src[i];
    const std::uint32_t bucket = (base_bucket + p.delay_slots) % ring.ring_size;
    p.bucket = static_cast<std::uint16_t>(bucket);
    const std::uint32_t slot = atomicAdd(&ring.header[bucket], 1u);
    if (slot < ring.capacity) {
        ring.packets[bucket * ring.capacity + slot] = p;
    }
    // else: dropped. In production this signals AER overflow → Status::AerQueueOverflow.
}

// Deliver an entire bucket: it becomes the input `spike_bits` for the next
// step. This kernel converts a list of packets into a packed bit-vector
// (one bit per neuron_id in the bucket), which downstream csr_spmv_spike
// consumes.
__global__ void aer_deliver_kernel(AerRing ring,
                                   std::uint32_t bucket,
                                   std::uint32_t* __restrict__ out_bits,
                                   std::uint32_t n_bit_words) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = ring.header[bucket];
    if (i >= count) {
        if (i < n_bit_words) out_bits[i] = 0u;  // best-effort clear
        return;
    }
    const SpikePacket p = ring.packets[bucket * ring.capacity + i];
    const std::uint32_t word = p.neuron_id >> 5;
    const std::uint32_t bit  = 1u << (p.neuron_id & 31u);
    atomicOr(&out_bits[word], bit);
}

__global__ void aer_clear_bucket_kernel(AerRing ring, std::uint32_t bucket) {
    if (threadIdx.x == 0 && blockIdx.x == 0) ring.header[bucket] = 0;
}

extern "C" cudaError_t launch_aer_enqueue(AerRing ring, const SpikePacket* d_src,
                                          std::uint32_t n, std::uint32_t base_bucket,
                                          cudaStream_t stream) {
    if (n == 0) return cudaSuccess;
    constexpr std::uint32_t block = 256;
    const std::uint32_t grid = (n + block - 1) / block;
    aer_enqueue_kernel<<<grid, block, 0, stream>>>(ring, d_src, n, base_bucket);
    return cudaGetLastError();
}

extern "C" cudaError_t launch_aer_deliver(AerRing ring, std::uint32_t bucket,
                                          std::uint32_t* d_out_bits,
                                          std::uint32_t n_bit_words,
                                          cudaStream_t stream) {
    // Zero the destination bit-vector first.
    cudaMemsetAsync(d_out_bits, 0, n_bit_words * sizeof(std::uint32_t), stream);
    const std::uint32_t block = 256;
    // Grid must cover max(count, n_bit_words).
    const std::uint32_t grid = (ring.capacity + block - 1) / block;
    aer_deliver_kernel<<<grid, block, 0, stream>>>(ring, bucket, d_out_bits, n_bit_words);
    aer_clear_bucket_kernel<<<1, 1, 0, stream>>>(ring, bucket);
    return cudaGetLastError();
}

}  // namespace agi::cuda
