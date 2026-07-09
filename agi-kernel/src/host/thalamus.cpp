// SPDX-License-Identifier: MIT
// ThalamicNucleus + AttentionVector — host implementation.

#include "agi/thalamus.hpp"

#include <cuda_runtime.h>

extern "C" cudaError_t launch_thalamus_route(agi::SpikePacket*, std::uint32_t,
                                             const float*, std::uint32_t,
                                             std::uint64_t, cudaStream_t);

namespace agi {

AttentionVector::AttentionVector(std::size_t n) : n_(n) {
    cudaMalloc(&d_gains_, n * sizeof(float));
    // Default gain = 1.0 (transparent thalamus).
    std::vector<float> ones(n, 1.0f);
    cudaMemcpy(d_gains_, ones.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}
AttentionVector::~AttentionVector() { if (d_gains_) cudaFree(d_gains_); }

Status AttentionVector::set(column_id_t source, float gain) {
    if (source >= n_) return Status::InvalidNeuronId;
    const float g = std::clamp(gain, 0.0f, 4.0f);
    cudaMemcpy(d_gains_ + source, &g, sizeof(float), cudaMemcpyHostToDevice);
    return Status::Ok;
}

Status AttentionVector::set_all_device(const float* d_gains) {
    cudaMemcpy(d_gains_, d_gains, n_ * sizeof(float), cudaMemcpyDeviceToDevice);
    return Status::Ok;
}

// ---------------------------------------------------------------------------
ThalamicNucleus::ThalamicNucleus(std::string_view name, std::size_t n_source_columns)
    : name_(name), n_source_columns_(n_source_columns) {}

Status ThalamicNucleus::route(SpikePacket* d_packets, std::size_t n_packets,
                              const AttentionVector& attention,
                              float ne_scalar, cudaStream_t stream) {
    (void)ne_scalar;   // consumed via __constant__ memory, but kept for local logging
    const auto err = launch_thalamus_route(d_packets,
                                           static_cast<std::uint32_t>(n_packets),
                                           attention.device_ptr(),
                                           static_cast<std::uint32_t>(n_source_columns_),
                                           /*seed*/ 0xC0DEC0DEuLL,
                                           stream);
    return err == cudaSuccess ? Status::Ok : Status::CudaLaunchFailed;
}

Status ThalamicNucleus::configure_pulvinar(std::uint32_t from_area, std::uint32_t gate_area,
                                           std::uint32_t to_area, float gate_threshold_hz) {
    if (pulvinar_rules_.size() >= 8) return Status::ShapeMismatch;
    pulvinar_rules_.push_back({from_area, gate_area, to_area, gate_threshold_hz});
    return Status::Ok;
}

}  // namespace agi
