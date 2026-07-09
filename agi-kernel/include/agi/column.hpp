// SPDX-License-Identifier: MIT
// Cortical column — the first-class kernel object in CortexOS.
//
// A `Column` bundles:
//   * A homogeneous slab of ~10 000 neurons (SoA layout)
//   * Local intra-column CSR connectivity
//   * A handle to the global inter-area (long-range) CSR
//   * Layer tags (L1..L6) so a canonical microcircuit can be scheduled
//   * A NeuronMemory arena for hybrid-codec observation storage
//
// Column::step() is the top-level kernel launch — it may lower to either
// (a) a fused Triton kernel (column_step.triton.py) or
// (b) a CUDA sequence: csr_spmv → neuron_step → spike_emit.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "agi/core.hpp"
#include "agi/neuron.hpp"
#include "agi/connectivity.hpp"
#include "agi/memory.hpp"

namespace agi {

// The canonical 6-layer cortical column, following Douglas & Martin 2004.
//   L1  — sparse; only apical tuft targets from other columns
//   L2/3 — cortico-cortical projections (pyramidal cells)
//   L4  — thalamo-cortical input layer (spiny stellates)
//   L5  — cortico-subcortical output (large pyramidal cells)
//   L6  — cortico-thalamic feedback
//
// A Column with fewer than 6 layers is legal; unused layers have offset==count==0.
struct LayerSpan {
    std::uint32_t offset;   // first neuron index of this layer in the column's SoA
    std::uint32_t count;    // number of neurons in this layer
};

class Column {
 public:
    Column(column_id_t id, std::uint32_t n_neurons, NeuronKind kind,
           time_ms_t dt_ms = kDefaultDtMs);
    ~Column();

    // ── Structural setters ─────────────────────────────────────────────────
    void set_layer(std::uint8_t layer_idx, LayerSpan span);
    void set_local_csr(std::shared_ptr<SparseConnectivity> csr);
    void set_long_range_csr(std::shared_ptr<SparseConnectivity> csr);

    // ── Access ─────────────────────────────────────────────────────────────
    column_id_t id() const noexcept { return id_; }
    std::uint32_t n_neurons() const noexcept { return n_neurons_; }
    NeuronKind kind() const noexcept { return kind_; }
    time_ms_t dt_ms() const noexcept { return dt_ms_; }
    const LayerSpan& layer(std::uint8_t i) const { return layers_[i]; }

    NeuronMemory& memory() { return memory_; }
    const NeuronMemory& memory() const { return memory_; }

    // ── Step primitive ─────────────────────────────────────────────────────
    //
    // Advances the whole column by one dt. External drive `i_ext` is a
    // per-neuron injected current (µA/cm²) — typically thalamic drive to L4.
    // Returns a bitmask of spikers (n_neurons bits, packed into u32 words).
    //
    // Implementation dispatches at construction time:
    //   * kind == HodgkinHuxley    → hh_step_kernel
    //   * kind == Izhikevich       → izh_step_kernel
    //   * kind == MultiCompartment → column_step.triton (fused)
    Status step(const float* i_ext_device, std::uint32_t* out_spike_bits_device);

    // Predictive-coding split: emit (prediction, error) tensors from this
    // column's L5 output. Implements the Rao & Ballard (1999) motif.
    Status predictive_split(float* prediction_out, float* error_out);

 private:
    column_id_t id_;
    std::uint32_t n_neurons_;
    NeuronKind kind_;
    time_ms_t dt_ms_;
    LayerSpan layers_[6]{};
    std::shared_ptr<SparseConnectivity> local_csr_;
    std::shared_ptr<SparseConnectivity> long_range_csr_;
    NeuronMemory memory_;

    // Device-resident SoA buffers, tagged by kind (only one is active).
    void* d_state_soa_ = nullptr;         // HHState* / IzhState* / CompartmentPacked*
    float* d_syn_current_ = nullptr;      // per-neuron accumulated synaptic current
    std::uint32_t* d_spike_bits_ = nullptr; // packed spike bitmask
};

}  // namespace agi
