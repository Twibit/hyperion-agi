// SPDX-License-Identifier: MIT
// CortexOS IR — MLIR-style dialect (façade header).
//
// Full spec: docs/IR_SPEC.md. This header exposes the C++ builder API,
// leaving the parser / printer / passes in src/ir/.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "agi/core.hpp"

namespace agi::ir {

class Module;
class Op;
class Value;
class Type;

// ── Types ───────────────────────────────────────────────────────────────────
Type spike_type();
Type neuron_type(NeuronKind kind);
Type column_type(std::uint32_t n_neurons, std::uint8_t n_layers);
Type csr_type();
Type rail_type();
Type trace_type(std::uint32_t length, time_ms_t dt);

// ── Builder ─────────────────────────────────────────────────────────────────
class Builder {
 public:
    Builder();
    Module module();

    Value neuron_create(NeuronKind kind);
    Value column_create(std::uint32_t n_neurons, std::uint8_t n_layers);
    Value column_step(Value column, Value inputs, time_ms_t dt);
    Value thalamus_route(Value packets, Value attention);
    Value pcoding_split(Value column);         // returns (pred, err)
    Op    stdp_update(Value csr, Value pre_bits, Value post_bits, Value da);
    Op    rail_broadcast(Rail rail, float value);
    Op    spike_emit(Value src_id, std::uint8_t delay, SynapseKind k);
};

// ── Passes ──────────────────────────────────────────────────────────────────
void run_fuse_column_step(Module& m);
void run_quantise_weights(Module& m);
void run_schedule_events(Module& m);
void run_place_columns(Module& m, const std::vector<int>& gpu_devices);
void run_insert_replay_checkpoints(Module& m);

// One-shot lowering to CUDA + Triton object modules.
struct LoweredModule {
    std::vector<std::string> cuda_kernels;    // filenames of emitted .cu / .cubin
    std::vector<std::string> triton_kernels;  // filenames of emitted .py / .so
};
LoweredModule lower(Module& m);

}  // namespace agi::ir
