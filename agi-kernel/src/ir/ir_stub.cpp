// SPDX-License-Identifier: MIT
// CortexOS IR — minimal design-skeleton implementation.
//
// The full IR uses MLIR C++ APIs; this stub gives the design's shape
// without pulling in the MLIR build. Ops are represented as tagged
// std::variants; passes walk the module and rewrite in place.

#include <agi/ir.hpp>
#include <agi/neuromod.hpp>
#include <iostream>
#include <variant>
#include <vector>

namespace agi::ir {

// ── Types (opaque handles) ──────────────────────────────────────────────────
class Type {
 public:
    std::string tag;
    std::string params;
    Type(std::string t = "", std::string p = "") : tag(std::move(t)), params(std::move(p)) {}
};

Type spike_type()  { return Type("spike"); }
Type neuron_type(NeuronKind k) { return Type("neuron", std::string(neuron_kind_name(k))); }
Type column_type(std::uint32_t n, std::uint8_t l) {
    return Type("column", std::to_string(n) + "x" + std::to_string(int(l)));
}
Type csr_type()    { return Type("csr"); }
Type rail_type()   { return Type("rail"); }
Type trace_type(std::uint32_t len, time_ms_t dt) {
    return Type("trace", std::to_string(len) + "@dt=" + std::to_string(dt));
}

// ── Values / Ops ────────────────────────────────────────────────────────────
class Value {
 public:
    std::uint32_t id = 0;
    Type type;
};

class Op {
 public:
    std::string mnemonic;
    std::vector<Value> operands;
    std::vector<Value> results;
    std::string attrs;
};

class Module {
 public:
    std::vector<Op> ops;
};

Builder::Builder() = default;
Module Builder::module() { return {}; }

Value Builder::neuron_create(NeuronKind kind) {
    (void)kind;
    return {};
}
Value Builder::column_create(std::uint32_t n, std::uint8_t l) { (void)n; (void)l; return {}; }
Value Builder::column_step(Value c, Value i, time_ms_t dt)    { (void)c; (void)i; (void)dt; return {}; }
Value Builder::thalamus_route(Value p, Value a)               { (void)p; (void)a; return {}; }
Value Builder::pcoding_split(Value c)                          { (void)c; return {}; }
Op    Builder::stdp_update(Value, Value, Value, Value)         { return {}; }
Op    Builder::rail_broadcast(Rail, float)                     { return {}; }
Op    Builder::spike_emit(Value, std::uint8_t, SynapseKind)    { return {}; }

// ── Passes (no-ops; documented in docs/IR_SPEC.md) ──────────────────────────
void run_fuse_column_step(Module&)             {}
void run_quantise_weights(Module&)             {}
void run_schedule_events(Module&)              {}
void run_place_columns(Module&, const std::vector<int>&) {}
void run_insert_replay_checkpoints(Module&)    {}

LoweredModule lower(Module&) {
    return LoweredModule{
        /*cuda_kernels =*/ {"hh_step_kernel", "izh_step_kernel", "csr_spmv_spike",
                            "aer_route", "stdp_update", "thalamus_route",
                            "wta_select", "replay", "neuromod_broadcast"},
        /*triton_kernels =*/ {"column_step", "cable_step", "stdp_fused"},
    };
}

}  // namespace agi::ir
