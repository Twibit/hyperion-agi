// SPDX-License-Identifier: MIT
// Runtime — top-level session.

#include "agi/runtime.hpp"

#include <cuda_runtime.h>
#include <fstream>

namespace agi {

Runtime::Runtime(RuntimeConfig cfg) : cfg_(std::move(cfg)) {
    // Bind primary GPU.
    if (!cfg_.gpu_devices.empty()) {
        cudaSetDevice(cfg_.gpu_devices.front());
    }
    neuromod_  = std::make_shared<NeuromodController>();
    scheduler_ = std::make_shared<Scheduler>(cfg_.dt_ms, cfg_.aer_ring_size);
    scheduler_->set_neuromod(neuromod_);
}

Runtime::~Runtime() = default;

std::shared_ptr<Column> Runtime::make_column(std::uint32_t n_neurons, NeuronKind kind) {
    auto c = std::make_shared<Column>(/*id*/ static_cast<column_id_t>(scheduler_->step_count()),
                                      n_neurons, kind, cfg_.dt_ms);
    scheduler_->add_column(c);
    return c;
}

std::shared_ptr<SparseConnectivity> Runtime::make_csr(std::uint32_t rows, std::uint32_t cols,
                                                     std::size_t nnz_reserve) {
    return std::make_shared<SparseConnectivity>(rows, cols, nnz_reserve);
}

std::shared_ptr<ThalamicNucleus> Runtime::make_thalamus(std::string_view name,
                                                        std::size_t n_source_columns) {
    auto t = std::make_shared<ThalamicNucleus>(name, n_source_columns);
    scheduler_->add_thalamus(t);
    return t;
}

std::shared_ptr<PlasticityEngine> Runtime::make_plasticity(SparseConnectivity& csr,
                                                           STDPConfig cfg) {
    auto p = std::make_shared<PlasticityEngine>(csr, cfg);
    scheduler_->add_plasticity(p);
    return p;
}

Status Runtime::snapshot(const std::string& path) {
    // Full implementation writes: config, per-column state (compressed),
    // CSR arrays, memory arenas. This stub writes only a marker file.
    std::ofstream out(path, std::ios::binary);
    if (!out) return Status::CheckpointCorrupt;
    const char magic[8] = {'C','O','R','T','E','X','O','S'};
    out.write(magic, 8);
    const std::uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    return Status::Ok;
}

Status Runtime::restore(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return Status::CheckpointCorrupt;
    char magic[8];
    in.read(magic, 8);
    if (std::string(magic, 8) != "CORTEXOS") return Status::CheckpointCorrupt;
    return Status::Ok;
}

}  // namespace agi
