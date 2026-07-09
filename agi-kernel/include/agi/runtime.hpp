// SPDX-License-Identifier: MIT
// Top-level session API — the thing user code holds.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "agi/core.hpp"
#include "agi/column.hpp"
#include "agi/connectivity.hpp"
#include "agi/thalamus.hpp"
#include "agi/neuromod.hpp"
#include "agi/plasticity.hpp"
#include "agi/scheduler.hpp"

namespace agi {

struct RuntimeConfig {
    time_ms_t dt_ms = kDefaultDtMs;
    std::vector<int> gpu_devices = {0};   // physical GPUs to bind
    bool enable_triton = true;
    bool enable_replay = true;
    std::size_t aer_ring_size = 1u << 20;  // 1 M packet-slots per delay bucket
};

class Runtime {
 public:
    explicit Runtime(RuntimeConfig cfg = {});
    ~Runtime();

    // ── Factory helpers ────────────────────────────────────────────────────
    std::shared_ptr<Column> make_column(std::uint32_t n_neurons, NeuronKind kind);
    std::shared_ptr<SparseConnectivity> make_csr(std::uint32_t rows, std::uint32_t cols,
                                                 std::size_t nnz_reserve = 0);
    std::shared_ptr<ThalamicNucleus> make_thalamus(std::string_view name,
                                                   std::size_t n_source_columns);
    std::shared_ptr<PlasticityEngine> make_plasticity(SparseConnectivity& csr,
                                                      STDPConfig cfg = {});

    // ── Accessors ──────────────────────────────────────────────────────────
    NeuromodController& neuromod() { return *neuromod_; }
    Scheduler& scheduler() { return *scheduler_; }
    const RuntimeConfig& config() const { return cfg_; }

    // ── Convenience wrappers ───────────────────────────────────────────────
    std::uint64_t run(time_ms_t T_ms) { return scheduler_->run(T_ms); }
    Status sleep(time_ms_t T_ms) { return scheduler_->sleep(T_ms); }

    // Save / restore session (columns + weights + memory arenas).
    Status snapshot(const std::string& path);
    Status restore(const std::string& path);

 private:
    RuntimeConfig cfg_;
    std::shared_ptr<NeuromodController> neuromod_;
    std::shared_ptr<Scheduler> scheduler_;
};

}  // namespace agi
