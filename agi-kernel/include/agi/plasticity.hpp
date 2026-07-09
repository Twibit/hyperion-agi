// SPDX-License-Identifier: MIT
// Plasticity rules.
//
// Three-factor STDP (Frémaux & Gerstner 2016):
//   dw_ij/dt = da(t) · e_ij(t)
//   de_ij/dt = -e_ij/τ_e + [ A+ · exp(-Δt/τ+)  if post-then-pre
//                          - A- · exp(+Δt/τ-)  if pre-then-post ]
//
// The eligibility trace e_ij closes the temporal gap between activity and a
// distal reward (Izhikevich 2007) — τ_e ≈ 500 ms.
//
// Structural plasticity (offline): weak synapses (|w| < eps) are pruned and
// new random synapses are grown between highly co-active neurons during
// replay (Tononi & Cirelli 2014).

#pragma once

#include <cstdint>

#include "agi/core.hpp"

namespace agi {

class SparseConnectivity;

struct STDPConfig {
    // Exponential windows.
    float tau_plus_ms  = 20.0f;
    float tau_minus_ms = 20.0f;
    // Amplitudes (in log-weight units).
    float a_plus  = 0.01f;
    float a_minus = 0.012f;
    // Eligibility trace time constant.
    float tau_elig_ms = 500.0f;
    // Weight clip range (in weight LUT indices).
    std::uint8_t w_min_idx = 0;
    std::uint8_t w_max_idx = 255;
};

class PlasticityEngine {
 public:
    PlasticityEngine(SparseConnectivity& csr, const STDPConfig& cfg = {});

    // Push the pre- and post-synaptic spike bitmasks for the current step.
    // The engine internally maintains per-synapse eligibility traces.
    Status observe(const std::uint32_t* d_pre_spike_bits,
                   const std::uint32_t* d_post_spike_bits,
                   time_ms_t dt_ms,
                   cudaStream_t stream = nullptr);

    // Apply the accumulated eligibility × dopamine to update weights.
    // Called once per step, *after* neuromodulator commit().
    Status apply(float da_scalar, time_ms_t dt_ms,
                 cudaStream_t stream = nullptr);

    // Off-line structural update: prune |w| < eps synapses, sample new ones
    // between top-K co-active pairs. Called during sleep / replay phase.
    Status structural_step(float prune_eps, std::size_t grow_budget);

 private:
    SparseConnectivity& csr_;
    STDPConfig cfg_;
    // Per-synapse eligibility trace, sized to csr_.nnz(). Kept as f16 to
    // halve memory; overflow is impossible because updates are clamped.
    void* d_e_trace_ = nullptr;
    // Per-neuron last-spike timestamps for the pair-based window.
    float* d_last_pre_ms_  = nullptr;
    float* d_last_post_ms_ = nullptr;
    float t_ms_ = 0.0f;
};

}  // namespace agi
