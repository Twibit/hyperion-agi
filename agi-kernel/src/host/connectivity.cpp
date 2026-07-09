// SPDX-License-Identifier: MIT
// SparseConnectivity + HubOverlay implementation.

#include "agi/connectivity.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

// CUDA launcher (implemented in src/cuda/csr_spmv_spike.cu).
extern "C" cudaError_t launch_csr_spmv_spike(const std::uint32_t*, const std::uint32_t*,
                                             const std::uint8_t*, const std::uint32_t*,
                                             float*, std::uint32_t, cudaStream_t);
extern "C" void set_weight_lut(float mu, float sigma);

namespace agi {

// Convert a linear weight into its 8-bit log-quantised index.
// Inverse of the LUT built by set_weight_lut(mu, sigma).
static std::uint8_t quantise_weight(float w, float mu, float sigma) {
    // log-normal: z = (log(w) - mu) / sigma;  index = clip((z + 3)/6 · 255, 0, 255)
    const float z = (std::log(std::max(w, 1e-6f)) - mu) / std::max(sigma, 1e-3f);
    const float t = (z + 3.0f) / 6.0f;
    const int idx = static_cast<int>(std::round(t * 255.0f));
    return static_cast<std::uint8_t>(std::clamp(idx, 0, 255));
}

SparseConnectivity::SparseConnectivity(std::uint32_t rows, std::uint32_t cols, std::size_t)
    : rows_(rows), cols_(cols),
      row_edges_pre_(rows), row_edges_w_(rows),
      row_edges_delay_(rows), row_edges_kind_(rows) {}

SparseConnectivity::~SparseConnectivity() {
    if (d_row_ptr_) cudaFree(d_row_ptr_);
    if (d_col_idx_) cudaFree(d_col_idx_);
    if (d_w_q8_)    cudaFree(d_w_q8_);
    if (d_delay_)   cudaFree(d_delay_);
    if (d_kind_)    cudaFree(d_kind_);
}

Status SparseConnectivity::push_edge(std::uint32_t post, std::uint32_t pre,
                                     float w, float delay_ms, SynapseKind k) {
    if (finalized_) return Status::ShapeMismatch;
    if (post >= rows_ || pre >= cols_) return Status::InvalidNeuronId;
    row_edges_pre_[post].push_back(pre);
    row_edges_w_[post].push_back(w);
    row_edges_delay_[post].push_back(delay_ms);
    row_edges_kind_[post].push_back(k);
    return Status::Ok;
}

Status SparseConnectivity::finalize() {
    if (finalized_) return Status::Ok;

    // Sort each row by pre index for coalesced access, count total nnz.
    nnz_ = 0;
    for (std::uint32_t r = 0; r < rows_; ++r) {
        // Sort tuple by pre.
        std::vector<std::size_t> perm(row_edges_pre_[r].size());
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(),
                  [&](std::size_t a, std::size_t b) {
                      return row_edges_pre_[r][a] < row_edges_pre_[r][b];
                  });
        auto reorder = [&](auto& v) {
            using T = typename std::decay_t<decltype(v)>::value_type;
            std::vector<T> out;
            out.reserve(v.size());
            for (auto i : perm) out.push_back(v[i]);
            v = std::move(out);
        };
        reorder(row_edges_pre_[r]);
        reorder(row_edges_w_[r]);
        reorder(row_edges_delay_[r]);
        reorder(row_edges_kind_[r]);
        nnz_ += row_edges_pre_[r].size();
    }

    // Log-normal LUT parameters (Song et al. 2005 fit).
    const float mu = -1.0f;
    const float sigma = 1.0f;
    set_weight_lut(mu, sigma);

    // Flatten CSR arrays.
    std::vector<std::uint32_t> row_ptr(rows_ + 1, 0);
    std::vector<std::uint32_t> col_idx;
    std::vector<std::uint8_t>  w_q8, delay_u8, kind_u8;
    col_idx.reserve(nnz_);
    w_q8.reserve(nnz_);
    delay_u8.reserve(nnz_);
    kind_u8.reserve(nnz_);

    for (std::uint32_t r = 0; r < rows_; ++r) {
        row_ptr[r + 1] = row_ptr[r] + static_cast<std::uint32_t>(row_edges_pre_[r].size());
        for (std::size_t i = 0; i < row_edges_pre_[r].size(); ++i) {
            col_idx.push_back(row_edges_pre_[r][i]);
            w_q8.push_back(quantise_weight(row_edges_w_[r][i], mu, sigma));
            // Delay in 0.1 ms slots, clamped to 8-bit.
            const int slots = static_cast<int>(std::round(row_edges_delay_[r][i] * 10.0f));
            delay_u8.push_back(static_cast<std::uint8_t>(std::clamp(slots, 0, 255)));
            kind_u8.push_back(static_cast<std::uint8_t>(row_edges_kind_[r][i]));
        }
    }

    // Move to device.
    cudaMalloc(&d_row_ptr_, (rows_ + 1) * sizeof(std::uint32_t));
    cudaMalloc(&d_col_idx_, nnz_ * sizeof(std::uint32_t));
    cudaMalloc(&d_w_q8_,    nnz_ * sizeof(std::uint8_t));
    cudaMalloc(&d_delay_,   nnz_ * sizeof(std::uint8_t));
    cudaMalloc(&d_kind_,    nnz_ * sizeof(std::uint8_t));
    cudaMemcpy(d_row_ptr_, row_ptr.data(), row_ptr.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx_, col_idx.data(), col_idx.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_q8_,    w_q8.data(),    w_q8.size(),                            cudaMemcpyHostToDevice);
    cudaMemcpy(d_delay_,   delay_u8.data(), delay_u8.size(),                       cudaMemcpyHostToDevice);
    cudaMemcpy(d_kind_,    kind_u8.data(), kind_u8.size(),                         cudaMemcpyHostToDevice);

    // Free build-time scratch.
    row_edges_pre_.clear(); row_edges_pre_.shrink_to_fit();
    row_edges_w_.clear();   row_edges_w_.shrink_to_fit();
    row_edges_delay_.clear(); row_edges_delay_.shrink_to_fit();
    row_edges_kind_.clear();  row_edges_kind_.shrink_to_fit();

    finalized_ = true;
    return Status::Ok;
}

Status SparseConnectivity::spmv_spike(const std::uint32_t* d_spike_bits,
                                      float* d_syn_current,
                                      cudaStream_t stream) const {
    if (!finalized_) return Status::ShapeMismatch;
    const cudaError_t err = launch_csr_spmv_spike(d_row_ptr_, d_col_idx_, d_w_q8_,
                                                  d_spike_bits, d_syn_current,
                                                  rows_, stream);
    return err == cudaSuccess ? Status::Ok : Status::CudaLaunchFailed;
}

// ---------------------------------------------------------------------------
// HubOverlay
// ---------------------------------------------------------------------------
HubOverlay::HubOverlay(std::size_t n_hubs)
    : csr_(static_cast<std::uint32_t>(n_hubs), static_cast<std::uint32_t>(n_hubs), n_hubs * n_hubs) {}

Status HubOverlay::add_hub(neuron_id_t id) { hub_ids_.push_back(id); return Status::Ok; }

Status HubOverlay::connect_hubs(neuron_id_t a, neuron_id_t b,
                                float w, float delay_ms, SynapseKind k) {
    // Map global neuron ids to hub-local indices.
    auto it_a = std::find(hub_ids_.begin(), hub_ids_.end(), a);
    auto it_b = std::find(hub_ids_.begin(), hub_ids_.end(), b);
    if (it_a == hub_ids_.end() || it_b == hub_ids_.end()) return Status::InvalidNeuronId;
    const std::uint32_t ai = static_cast<std::uint32_t>(std::distance(hub_ids_.begin(), it_a));
    const std::uint32_t bi = static_cast<std::uint32_t>(std::distance(hub_ids_.begin(), it_b));
    return csr_.push_edge(bi, ai, w, delay_ms, k);
}

Status HubOverlay::finalize() { return csr_.finalize(); }

}  // namespace agi
