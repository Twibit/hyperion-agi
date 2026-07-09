// SPDX-License-Identifier: MIT
#include "sensor_encoders.hpp"

#include <cmath>
#include <random>

namespace agi::io {

std::vector<time_ms_t> poisson_encode(float hz, time_ms_t dt_ms, std::size_t n_steps,
                                      std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const double p_step = double(hz) * double(dt_ms) * 1e-3;
    std::vector<time_ms_t> out;
    for (std::size_t s = 0; s < n_steps; ++s) {
        if (uni(rng) < p_step) out.push_back(double(s) * dt_ms);
    }
    return out;
}

time_ms_t latency_encode(float x_norm, time_ms_t tau_ms) {
    if (x_norm <= 0.0f) return tau_ms;
    if (x_norm >= 1.0f) return 0.0;
    return tau_ms * (1.0 - double(x_norm));
}

std::vector<std::vector<time_ms_t>> population_encode(float x, const PopulationCodeParams& p,
                                                      time_ms_t dt_ms, std::size_t n_steps,
                                                      std::uint64_t seed) {
    std::vector<std::vector<time_ms_t>> out(p.n_fields);
    for (std::size_t k = 0; k < p.n_fields; ++k) {
        const float centre = p.x_min + (p.x_max - p.x_min) * float(k) / float(p.n_fields - 1);
        const float d = (x - centre) / p.sigma;
        const float rate = p.peak_hz * std::exp(-0.5f * d * d);
        out[k] = poisson_encode(rate, dt_ms, n_steps, seed + k);
    }
    return out;
}

}  // namespace agi::io
