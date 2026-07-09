// SPDX-License-Identifier: MIT
// PlasticityEngine — host side of the STDP + eligibility pipeline.

#include "agi/plasticity.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "agi/connectivity.hpp"

extern "C" cudaError_t launch_stdp_observe(
    const std::uint32_t*, const std::uint32_t*, __half*,
    const std::uint32_t*, const std::uint32_t*,
    const float*, const float*,
    float, float, float, float, float, float, float,
    std::uint32_t, cudaStream_t);

extern "C" cudaError_t launch_stdp_apply(const std::uint32_t*, __half*, std::uint8_t*,
                                         float, float, std::uint32_t, cudaStream_t);

namespace agi {

PlasticityEngine::PlasticityEngine(SparseConnectivity& csr, const STDPConfig& cfg)
    : csr_(csr), cfg_(cfg) {
    if (csr_.nnz() > 0) {
        cudaMalloc(&d_e_trace_, csr_.nnz() * sizeof(__half));
        cudaMemset(d_e_trace_, 0, csr_.nnz() * sizeof(__half));
    }
    cudaMalloc(&d_last_pre_ms_,  csr_.cols() * sizeof(float));
    cudaMalloc(&d_last_post_ms_, csr_.rows() * sizeof(float));
    cudaMemset(d_last_pre_ms_,  0xFF, csr_.cols() * sizeof(float));   // -inf-ish
    cudaMemset(d_last_post_ms_, 0xFF, csr_.rows() * sizeof(float));
}

Status PlasticityEngine::observe(const std::uint32_t* d_pre_bits,
                                 const std::uint32_t* d_post_bits,
                                 time_ms_t dt_ms, cudaStream_t stream) {
    t_ms_ += static_cast<float>(dt_ms);
    if (!d_pre_bits || !d_post_bits || !d_e_trace_) return Status::Ok;
    const auto err = launch_stdp_observe(
        csr_.d_row_ptr(), csr_.d_col_idx(), static_cast<__half*>(d_e_trace_),
        d_pre_bits, d_post_bits, d_last_pre_ms_, d_last_post_ms_,
        t_ms_, cfg_.tau_plus_ms, cfg_.tau_minus_ms,
        cfg_.a_plus, cfg_.a_minus, cfg_.tau_elig_ms, static_cast<float>(dt_ms),
        csr_.rows(), stream);
    return err == cudaSuccess ? Status::Ok : Status::CudaLaunchFailed;
}

Status PlasticityEngine::apply(float da, time_ms_t /*dt_ms*/, cudaStream_t stream) {
    if (!d_e_trace_) return Status::Ok;
    const float lr = 1.0f;   // absorbed into a_plus/a_minus if desired
    // The CSR type owns d_w_q8_ as a const pointer; PlasticityEngine mutates
    // it via a friendship contract, or via a mutable accessor in production.
    auto* w_q8_mut = const_cast<std::uint8_t*>(csr_.d_w_q8());
    const auto err = launch_stdp_apply(csr_.d_row_ptr(),
                                       static_cast<__half*>(d_e_trace_),
                                       w_q8_mut, da, lr, csr_.rows(), stream);
    return err == cudaSuccess ? Status::Ok : Status::CudaLaunchFailed;
}

Status PlasticityEngine::structural_step(float /*prune_eps*/, std::size_t /*grow_budget*/) {
    // Off-line synaptic homeostasis (Tononi & Cirelli 2014).
    // Design-skeleton stub — full implementation walks the CSR, atomically
    // sets w_q8 = 0 for weights below prune_eps, and appends new random
    // edges between the top-K co-active pairs from the replay buffer.
    return Status::Ok;
}

}  // namespace agi
