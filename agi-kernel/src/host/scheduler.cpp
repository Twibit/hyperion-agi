// SPDX-License-Identifier: MIT
// Event-driven scheduler — orchestrates one dt.

#include "agi/scheduler.hpp"

#include <cuda_runtime.h>
#include <vector>

#include "agi/column.hpp"
#include "agi/thalamus.hpp"
#include "agi/neuromod.hpp"
#include "agi/plasticity.hpp"

namespace agi {

struct Scheduler::Streams {
    cudaStream_t routing = nullptr;
    cudaStream_t compute = nullptr;
    cudaStream_t plastic = nullptr;
    cudaStream_t replay  = nullptr;

    Streams() {
        cudaStreamCreate(&routing);
        cudaStreamCreate(&compute);
        cudaStreamCreate(&plastic);
        cudaStreamCreate(&replay);
    }
    ~Streams() {
        cudaStreamDestroy(routing);
        cudaStreamDestroy(compute);
        cudaStreamDestroy(plastic);
        cudaStreamDestroy(replay);
    }
};

Scheduler::Scheduler(time_ms_t dt_ms, std::size_t ring_size)
    : dt_ms_(dt_ms), aer_ring_size_(ring_size),
      streams_(std::make_unique<Streams>()) {}

Scheduler::~Scheduler() = default;

void Scheduler::add_column(std::shared_ptr<Column> c)          { columns_.push_back(std::move(c)); }
void Scheduler::add_thalamus(std::shared_ptr<ThalamicNucleus> t){ thalami_.push_back(std::move(t)); }
void Scheduler::set_neuromod(std::shared_ptr<NeuromodController> nm){ neuromod_ = std::move(nm); }
void Scheduler::add_plasticity(std::shared_ptr<PlasticityEngine> p){ plasticity_.push_back(std::move(p)); }

Status Scheduler::step(const float* i_ext_device) {
    // ── 1. Broadcast the current neuromod snapshot ──────────────────────────
    if (neuromod_) neuromod_->commit(streams_->compute);

    // ── 2. Advance each column ─────────────────────────────────────────────
    //     (Pipelined via streams; long-range CSR SpMV uses `routing` stream.)
    for (auto& c : columns_) {
        c->step(i_ext_device, /*out_spike_bits*/ nullptr);
    }

    // ── 3. Plasticity (STDP observe + apply) ───────────────────────────────
    for (auto& p : plasticity_) {
        // In production the pre/post bit vectors are gathered from Column;
        // for the design skeleton we call the engine with dt only.
        p->observe(/*pre*/ nullptr, /*post*/ nullptr, dt_ms_, streams_->plastic);
        const float da = neuromod_ ? neuromod_->snapshot().da : 0.0f;
        p->apply(da, dt_ms_, streams_->plastic);
    }

    // ── 4. Advance clocks ──────────────────────────────────────────────────
    t_ms_       += dt_ms_;
    step_count_ += 1;

    // Wait for all four streams to converge on the dt boundary.
    cudaStreamSynchronize(streams_->compute);
    cudaStreamSynchronize(streams_->routing);
    cudaStreamSynchronize(streams_->plastic);
    return Status::Ok;
}

std::uint64_t Scheduler::run(time_ms_t T_ms) {
    const std::uint64_t n_steps = static_cast<std::uint64_t>(T_ms / dt_ms_);
    std::uint64_t total_spikes = 0;
    for (std::uint64_t i = 0; i < n_steps; ++i) {
        step(/*i_ext*/ nullptr);
        // In production, sum spike bitmasks from every column here.
    }
    return total_spikes;
}

Status Scheduler::sleep(time_ms_t duration_ms) {
    // Off-line phase: drive columns from the hippocampal replay buffer, run
    // STDP in an accelerated frame, then prune weak synapses. See replay.cu.
    // The design skeleton stubs this to sleeping without-drive for now.
    const std::uint64_t n_steps = static_cast<std::uint64_t>(duration_ms / dt_ms_);
    for (std::uint64_t i = 0; i < n_steps; ++i) {
        step(/*i_ext*/ nullptr);
    }
    return Status::Ok;
}

}  // namespace agi
