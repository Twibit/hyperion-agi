// SPDX-License-Identifier: MIT
// NeuromodController — updates the four broadcast rails.

#include "agi/neuromod.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

extern "C" cudaError_t launch_neuromod_broadcast(const agi::NeuromodSnapshot, cudaStream_t);

namespace agi {

NeuromodController::NeuromodController() = default;

void NeuromodController::set(Rail r, float v) {
    switch (r) {
        case Rail::DA:  staged_.da  = std::clamp(v, -5.0f, 5.0f); break;
        case Rail::NE:  staged_.ne  = std::clamp(v,  0.0f, 1.0f); break;
        case Rail::ACh: staged_.ach = std::clamp(v,  0.0f, 1.0f); break;
        case Rail::HT5: staged_.ht5 = std::clamp(v,  0.0f, 1.0f); break;
        default: break;
    }
}

float NeuromodController::get(Rail r) const {
    switch (r) {
        case Rail::DA:  return staged_.da;
        case Rail::NE:  return staged_.ne;
        case Rail::ACh: return staged_.ach;
        case Rail::HT5: return staged_.ht5;
        default: return 0.0f;
    }
}

void NeuromodController::update(float rpe, float novelty, time_ms_t dt_ms) {
    // Single-pole low-pass filter, tau = 200 ms per rail.
    const float tau_ms = 200.0f;
    const float alpha  = static_cast<float>(dt_ms) / (tau_ms + dt_ms);

    // Dopamine tracks the reward-prediction error directly (phasic burst / dip).
    staged_.da += alpha * (rpe - staged_.da);

    // Norepinephrine ~ absolute value of rpe (surprise ≡ arousal).
    staged_.ne += alpha * (std::fabs(rpe) - staged_.ne);

    // Acetylcholine tracks novelty in [0, 1].
    staged_.ach += alpha * (novelty - staged_.ach);

    // Serotonin drifts slowly toward the moving average of reward stability
    // (crude but captures Doya's patience dimension).
    const float stability = 1.0f - std::fabs(rpe);
    staged_.ht5 += 0.1f * alpha * (stability - staged_.ht5);

    // Final clamp.
    staged_.ne  = std::clamp(staged_.ne,  0.0f, 1.0f);
    staged_.ach = std::clamp(staged_.ach, 0.0f, 1.0f);
    staged_.ht5 = std::clamp(staged_.ht5, 0.0f, 1.0f);
}

Status NeuromodController::commit(cudaStream_t stream) {
    const cudaError_t err = launch_neuromod_broadcast(staged_, stream);
    return err == cudaSuccess ? Status::Ok : Status::CudaLaunchFailed;
}

}  // namespace agi
