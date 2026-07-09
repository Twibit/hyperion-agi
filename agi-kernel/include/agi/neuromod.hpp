// SPDX-License-Identifier: MIT
// Neuromodulation broadcast rails.
//
// Four global scalar signals modulate the entire runtime once per dt:
//
//   da   — dopamine       (reward-prediction error, Schultz 1998)
//   ne   — norepinephrine (arousal / uncertainty,   Aston-Jones 2005)
//   ach  — acetylcholine  (novelty / attention,     Sarter 2005)
//   ht5  — serotonin      (patience / discount,     Doya 2002)
//
// Rails are stored in device __constant__ memory and read by every kernel
// launched during the current step. Writers must call `commit()` at the end
// of a step; readers see a consistent snapshot.

#pragma once

#include <array>
#include <cstdint>

#include "agi/core.hpp"

namespace agi {

enum class Rail : std::uint8_t {
    DA  = 0,
    NE  = 1,
    ACh = 2,
    HT5 = 3,
    N   = 4,   // count
};

struct NeuromodSnapshot {
    float da  = 0.0f;   // reward-prediction error (may be negative)
    float ne  = 0.5f;   // arousal, in [0, 1]
    float ach = 0.1f;   // uncertainty, in [0, 1]
    float ht5 = 0.5f;   // patience, in [0, 1]

    // Discount factor used by RL heads and by structural plasticity's
    // eligibility trace decay.
    float discount() const noexcept { return 0.5f + 0.49f * ht5; }
};

// Host-side controller. Writers set staged values, then commit() copies to
// __constant__ memory in a single cudaMemcpyToSymbolAsync.
class NeuromodController {
 public:
    NeuromodController();

    void set(Rail r, float value);
    float get(Rail r) const;

    // Update from a reward-prediction error and a novelty score; a small
    // fixed dynamics keeps the four scalars smooth (single-pole low-pass
    // with tau = 200 ms).
    void update(float rpe, float novelty, time_ms_t dt_ms);

    // Push the current staged snapshot to device __constant__ memory.
    Status commit(cudaStream_t stream = nullptr);

    const NeuromodSnapshot& snapshot() const noexcept { return staged_; }

 private:
    NeuromodSnapshot staged_;
};

}  // namespace agi
