// SPDX-License-Identifier: MIT
// Sensor encoders — convert continuous signals into spike trains.
//
// Three canonical schemes:
//   * Poisson rate coding      x -> Poisson(λ = f(x))         (Softky & Koch 1993)
//   * Latency (time-to-first)  x -> single spike at t = τ·(1 − x)  (Thorpe 1990)
//   * Population coding        x -> Gaussian-tiled receptive fields

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include "agi/core.hpp"

namespace agi::io {

// Emit a Poisson spike train over `n_steps` at `dt_ms`, with rate `hz`.
// Returned as absolute times (ms).
std::vector<time_ms_t> poisson_encode(float hz, time_ms_t dt_ms, std::size_t n_steps,
                                      std::uint64_t seed);

// Latency code: single spike at t = τ * (1 − x_norm), x_norm ∈ [0,1].
time_ms_t latency_encode(float x_norm, time_ms_t tau_ms);

// Population code: N Gaussian receptive fields tiled over [x_min, x_max].
// Each field emits one Poisson spike train whose rate peaks at its centre.
struct PopulationCodeParams {
    std::size_t n_fields = 20;
    float x_min = 0.0f, x_max = 1.0f;
    float sigma = 0.05f;
    float peak_hz = 100.0f;
};

std::vector<std::vector<time_ms_t>> population_encode(float x,
                                                      const PopulationCodeParams& p,
                                                      time_ms_t dt_ms,
                                                      std::size_t n_steps,
                                                      std::uint64_t seed);

}  // namespace agi::io
