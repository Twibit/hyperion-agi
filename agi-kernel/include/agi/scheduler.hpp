// SPDX-License-Identifier: MIT
// Event-driven scheduler / step engine.
//
// The scheduler is the entry point for a simulation. Per outer step it:
//   1. Delivers AER packets whose (t_emit + delay) == t.
//   2. Advances each column (dispatched by NeuronKind).
//   3. Broadcasts the neuromod snapshot to __constant__ memory.
//   4. Applies gated STDP updates.
//   5. Feeds new spikes to the hippocampal ring buffer.
//   6. In "sleep" mode, drives the replay engine (which itself drives STDP).
//
// Steps 1–4 are pipelined across four CUDA streams to hide launch latency.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "agi/core.hpp"

namespace agi {

class Column;
class ThalamicNucleus;
class NeuromodController;
class PlasticityEngine;

class Scheduler {
 public:
    Scheduler(time_ms_t dt_ms = kDefaultDtMs, std::size_t aer_ring_size = 1 << 20);
    ~Scheduler();

    // ── Registration ───────────────────────────────────────────────────────
    void add_column(std::shared_ptr<Column> c);
    void add_thalamus(std::shared_ptr<ThalamicNucleus> t);
    void set_neuromod(std::shared_ptr<NeuromodController> nm);
    void add_plasticity(std::shared_ptr<PlasticityEngine> p);

    // ── Simulation ─────────────────────────────────────────────────────────
    // Advance the whole runtime by one dt.
    Status step(const float* i_ext_device = nullptr);
    // Run for T ms, returning cumulative spike count.
    std::uint64_t run(time_ms_t T_ms);

    // ── Sleep / replay ─────────────────────────────────────────────────────
    // Switch to off-line replay for `duration_ms`; drives structural
    // plasticity + weight consolidation (SWS-like phase).
    Status sleep(time_ms_t duration_ms);

    time_ms_t t_ms() const noexcept { return t_ms_; }
    step_t step_count() const noexcept { return step_count_; }

 private:
    time_ms_t dt_ms_;
    time_ms_t t_ms_ = 0.0;
    step_t step_count_ = 0;

    std::size_t aer_ring_size_;

    std::vector<std::shared_ptr<Column>> columns_;
    std::vector<std::shared_ptr<ThalamicNucleus>> thalami_;
    std::shared_ptr<NeuromodController> neuromod_;
    std::vector<std::shared_ptr<PlasticityEngine>> plasticity_;

    // CUDA streams for pipelining (opaque — implementation detail).
    struct Streams;
    std::unique_ptr<Streams> streams_;
};

}  // namespace agi
