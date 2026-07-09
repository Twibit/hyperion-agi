// SPDX-License-Identifier: MIT
// Sparse connectivity — CSR layout with 8-bit log-quantised weights.
//
// Storage model (per shard):
//   row_ptr [rows+1]   uint32_t   — CSR row pointers
//   col_idx [nnz]      uint32_t   — column (pre-synaptic neuron) index
//   w_q8    [nnz]      uint8_t    — log-quantised weight (see WEIGHT_LUT)
//   delay   [nnz]      uint8_t    — axonal delay in 0.1 ms slots
//   kind    [nnz]      uint8_t    — SynapseKind
//
// The layout matches neurorust::SparseConnectivity, with the addition of
// per-synapse delay and kind. Weight quantisation table WEIGHT_LUT[256]
// is a log-normal spacing (Song et al. 2005) built at init.

#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

#include "agi/core.hpp"

namespace agi {

class SparseConnectivity {
 public:
    // Build an empty CSR with `rows`×`cols` shape and `nnz_reserve` capacity.
    SparseConnectivity(std::uint32_t rows, std::uint32_t cols,
                       std::size_t nnz_reserve = 0);
    ~SparseConnectivity();

    // ── Construction helpers ───────────────────────────────────────────────
    //
    // Push a single edge j -> i (post ← pre) with weight `w`, delay `d` (ms)
    // and synapse kind `k`. Only legal during the "building" phase; call
    // `finalize()` to move to device memory.
    Status push_edge(std::uint32_t post, std::uint32_t pre,
                     float w, float delay_ms, SynapseKind k);

    Status finalize();  // sort per row, quantise weights, move to GPU

    // ── Access ─────────────────────────────────────────────────────────────
    std::uint32_t rows() const noexcept { return rows_; }
    std::uint32_t cols() const noexcept { return cols_; }
    std::size_t nnz() const noexcept { return nnz_; }

    const std::uint32_t* d_row_ptr() const noexcept { return d_row_ptr_; }
    const std::uint32_t* d_col_idx() const noexcept { return d_col_idx_; }
    const std::uint8_t*  d_w_q8()    const noexcept { return d_w_q8_; }
    const std::uint8_t*  d_delay()   const noexcept { return d_delay_; }
    const std::uint8_t*  d_kind()    const noexcept { return d_kind_; }

    // The dequantisation LUT (256 floats) — kept in constant memory on device.
    static const float* weight_lut_device();

    // Spike-driven SpMV:
    //   syn_current[i] += Σ_j  LUT[w_q8_ij] · spike_bit_j
    //
    // The input is a packed bit-vector of pre-synaptic spikes; the output is
    // a dense per-post-neuron synaptic current. Lowers to csr_spmv_spike.cu.
    Status spmv_spike(const std::uint32_t* d_spike_bits,
                      float* d_syn_current,
                      cudaStream_t stream = nullptr) const;

 private:
    std::uint32_t rows_, cols_;
    std::size_t nnz_ = 0;
    bool finalized_ = false;

    // Host-side scratch during building.
    std::vector<std::vector<std::uint32_t>> row_edges_pre_;
    std::vector<std::vector<float>>         row_edges_w_;
    std::vector<std::vector<float>>         row_edges_delay_;
    std::vector<std::vector<SynapseKind>>   row_edges_kind_;

    // Device buffers, populated by finalize().
    std::uint32_t* d_row_ptr_ = nullptr;
    std::uint32_t* d_col_idx_ = nullptr;
    std::uint8_t*  d_w_q8_    = nullptr;
    std::uint8_t*  d_delay_   = nullptr;
    std::uint8_t*  d_kind_    = nullptr;
};

// ---------------------------------------------------------------------------
// Hub / rich-club overlay
// ---------------------------------------------------------------------------
//
// 90 % of edges are local (same or neighbouring column) and stored in each
// Column's local CSR. The remaining ~10 %, following Bullmore & Sporns 2009,
// form a *rich club* — a densely interconnected small subset of high-degree
// "hub" neurons. `HubOverlay` stores those globally, replicated on every GPU.
class HubOverlay {
 public:
    explicit HubOverlay(std::size_t n_hubs);

    Status add_hub(neuron_id_t id);
    Status connect_hubs(neuron_id_t a, neuron_id_t b,
                        float w, float delay_ms, SynapseKind k);
    Status finalize();

    const SparseConnectivity& csr() const { return csr_; }

 private:
    SparseConnectivity csr_;
    std::vector<neuron_id_t> hub_ids_;
};

}  // namespace agi
