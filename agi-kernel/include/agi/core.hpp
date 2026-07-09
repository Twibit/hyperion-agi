// SPDX-License-Identifier: MIT
// CortexOS core types.
//
// This header defines the low-level, POD types shared by every subsystem:
//   * AER spike packet
//   * NeuronKind enum (mirrors neurorust::NeuronKind)
//   * Simulation-time and identifier types
//   * Error codes
//
// Nothing in this header depends on CUDA or Triton; it is safe to include
// from host code and from device code alike.

#pragma once

#include <cstdint>
#include <cstddef>
#include <string_view>

namespace agi {

// ---------------------------------------------------------------------------
// Fundamental scalar types
// ---------------------------------------------------------------------------

using neuron_id_t   = std::uint32_t;   // global neuron address (per shard)
using column_id_t   = std::uint32_t;   // global column address
using synapse_id_t  = std::uint64_t;   // sparse synapse address
using step_t        = std::uint64_t;   // outer simulation step count
using time_ms_t     = double;          // biophysical time in ms

// dt of the outer scheduler loop. 0.1 ms is a good compromise between
// Hodgkin-Huxley stability (needs <= 0.02 ms per RK4 substep, so 4 substeps
// per outer step) and AER traffic (10 kHz outer rate is fine for NVLink 5).
inline constexpr time_ms_t kDefaultDtMs = 0.1;

// ---------------------------------------------------------------------------
// Neuron families — mirrors neurorust::NeuronKind exactly.
// ---------------------------------------------------------------------------

enum class NeuronKind : std::uint8_t {
    HodgkinHuxley    = 0,  // 4-state ODE, RK4, gold-standard biophysics
    Izhikevich       = 1,  // 2-state phenomenological, 20+ firing regimes
    MultiCompartment = 2,  // soma + dendrite compartments, cable eq.
};

constexpr std::string_view neuron_kind_name(NeuronKind k) noexcept {
    switch (k) {
        case NeuronKind::HodgkinHuxley:    return "HodgkinHuxley";
        case NeuronKind::Izhikevich:       return "Izhikevich";
        case NeuronKind::MultiCompartment: return "MultiCompartment";
    }
    return "unknown";
}

// Firing pattern labels used by classifiers and the WTA selector.
// (Names from Izhikevich 2003, Table 1.)
enum class FiringPattern : std::uint8_t {
    Quiescent            = 0,
    RegularSpiking       = 1,
    IntrinsicallyBursting = 2,
    Chattering           = 3,
    FastSpiking          = 4,
    LowThresholdSpiking  = 5,
};

// ---------------------------------------------------------------------------
// AER — Address-Event Representation packet
// ---------------------------------------------------------------------------
//
// 48-bit spike packet, packed into 8 bytes (top 16 bits reserved for the
// scheduler's delay-slot bucket header when queued):
//
//   bits [0..31]  neuron_id           — global source neuron
//   bits [32..39] delay_slots         — axonal delay, 0.1 ms per slot (25.5 ms max)
//   bits [40..43] weight_idx          — 1 of 16 synapse types (AMPA, NMDA, GABA-A, ...)
//   bits [44..47] meta                — dopamine tag, replay marker, ...
//
// This layout matches Boahen (2000) silicon-retina AER, extended with synapse
// type and metadata for neuromodulation.
struct alignas(8) SpikePacket {
    std::uint32_t neuron_id;      // source neuron id
    std::uint8_t  delay_slots;    // 0-255, in units of dt (0.1 ms)
    std::uint8_t  weight_meta;    // low nibble = weight_idx, high nibble = meta
    std::uint16_t bucket;         // set by scheduler on enqueue (delivery step % ringsize)

    constexpr std::uint8_t weight_idx() const noexcept { return weight_meta & 0x0F; }
    constexpr std::uint8_t meta()       const noexcept { return (weight_meta >> 4) & 0x0F; }
};
static_assert(sizeof(SpikePacket) == 8, "SpikePacket must pack to 8 bytes");

// Synapse type table — must match `weight_idx` above. Extend but do not
// reorder without bumping the on-disk snapshot version.
enum class SynapseKind : std::uint8_t {
    AMPA   = 0,   // fast glutamatergic excitation, tau ≈ 2 ms
    NMDA   = 1,   // slow, Mg-blocked, tau ≈ 100 ms
    GABAA  = 2,   // fast Cl- inhibition, tau ≈ 5 ms
    GABAB  = 3,   // slow K+ inhibition, tau ≈ 150 ms
    ACH    = 4,   // cholinergic (nicotinic + muscarinic mix, simplified)
    DA_D1  = 5,   // dopaminergic D1 (excitatory, striatum)
    DA_D2  = 6,   // dopaminergic D2 (inhibitory, striatum)
    NE     = 7,   // noradrenergic
    HT5_2A = 8,   // serotonergic 5-HT2A
    Electrical = 9, // gap junction (instantaneous, symmetric)
    // Reserved: 10..15 for future synapse kinds.
};

// ---------------------------------------------------------------------------
// Error handling — tiny enum, converted to std::error_code in host code.
// ---------------------------------------------------------------------------

enum class Status : std::int32_t {
    Ok                    = 0,
    OutOfMemory           = 1,
    ShapeMismatch         = 2,
    UnknownNeuronKind     = 3,
    CudaLaunchFailed      = 4,
    TritonCompileFailed   = 5,
    AerQueueOverflow      = 6,
    InvalidNeuronId       = 7,
    CheckpointCorrupt     = 8,
};

constexpr bool ok(Status s) noexcept { return s == Status::Ok; }

}  // namespace agi
