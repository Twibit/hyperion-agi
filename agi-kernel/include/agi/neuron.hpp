// SPDX-License-Identifier: MIT
// Neuron state types — one struct per NeuronKind.
//
// Design note: these are plain-old-data types laid out for coalesced access
// on the GPU. Structure-of-arrays is preferred in the storage layer
// (see include/agi/column.hpp) but each type below is the AoS-style view
// used inside a single kernel invocation.

#pragma once

#include <cstdint>
#include "agi/core.hpp"

namespace agi {

// ---------------------------------------------------------------------------
// Hodgkin–Huxley — 4-state (V, m, h, n)
// ---------------------------------------------------------------------------
//
// Values ported from neurorust::HHParams (squid-giant-axon canonical set,
// Hodgkin & Huxley 1952). Units:
//   c_m  µF/cm²
//   g_*  mS/cm²
//   e_*  mV
struct HHParams {
    float c_m  = 1.0f;
    float g_na = 120.0f;
    float g_k  = 36.0f;
    float g_l  = 0.3f;
    float e_na = 50.0f;
    float e_k  = -77.0f;
    float e_l  = -54.4f;
};

// Per-neuron state. 24 bytes → 3 float4 loads.
struct HHState {
    float v;       // membrane potential (mV)
    float m;       // Na+ activation
    float h;       // Na+ inactivation
    float n;       // K+  activation
    float prev_v;  // for spike detection (upcrossing of 0 mV)
    float pad;     // pad to 24 bytes for alignment
};
static_assert(sizeof(HHState) == 24, "HHState must be 24 bytes");

// ---------------------------------------------------------------------------
// Izhikevich — 2-state (V, u); 4 parameters
// ---------------------------------------------------------------------------
//
// Presets from Izhikevich (2003) Table 1. Encoded in a 16-byte struct so
// that we can pack (state, params) into a float8 for the CUDA kernel.
struct IzhParams {
    float a;   // recovery time-scale
    float b;   // recovery sensitivity
    float c;   // reset potential (mV)
    float d;   // recovery after-spike jump

    // ── Canonical presets ─────────────────────────────────────────────────
    static constexpr IzhParams regular_spiking()      { return { 0.02f, 0.20f, -65.0f, 8.0f  }; }
    static constexpr IzhParams intrinsically_bursting(){ return { 0.02f, 0.20f, -55.0f, 4.0f  }; }
    static constexpr IzhParams chattering()           { return { 0.02f, 0.20f, -50.0f, 2.0f  }; }
    static constexpr IzhParams fast_spiking()         { return { 0.10f, 0.20f, -65.0f, 2.0f  }; }
    static constexpr IzhParams low_threshold_spiking(){ return { 0.02f, 0.25f, -65.0f, 2.0f  }; }
};

struct IzhState {
    float v;      // membrane potential (mV)
    float u;      // recovery variable
    float pad0;
    float pad1;
};
static_assert(sizeof(IzhState) == 16, "IzhState must be 16 bytes");

// ---------------------------------------------------------------------------
// Multi-compartment — soma + K dendritic compartments
// ---------------------------------------------------------------------------
//
// Cable equation on a rooted tree of compartments:
//    C dV_i/dt = -g_L (V_i - E_L) + Σ_{j∈nbr(i)} g_ij (V_j - V_i) + I_inj_i
// Compartment 0 is the soma and carries the active HH machinery (m,h,n);
// dendrites are passive unless flagged `nmda_capable`, in which case an
// NMDA-plateau nonlinearity is enabled (Larkum 1999).
struct Compartment {
    float v;              // membrane potential (mV)
    float g_axial;        // axial coupling to parent (mS/cm²)
    float g_leak;         // membrane leak (mS/cm²)
    float e_leak;         // leak reversal (mV)
    float c_m;            // capacitance (µF/cm²)
    float area_um2;       // surface area (µm²), for scaling
    std::uint16_t parent; // parent compartment index (0xFFFF = root/soma)
    std::uint8_t  flags;  // bit 0 = nmda_capable, bit 1 = active_Na, bit 2 = active_K
    std::uint8_t  layer;  // cortical layer (0..5)
};
static_assert(sizeof(Compartment) == 28, "Compartment size");

// Padded compartment for coalesced loads (32 bytes = float8).
struct alignas(32) CompartmentPacked { Compartment c; float pad; };
static_assert(sizeof(CompartmentPacked) == 32, "CompartmentPacked = float8");

// ---------------------------------------------------------------------------
// Uniform neuron descriptor stored in host tables. The `union` is
// discriminated by NeuronKind. For multi-compartment neurons the union
// carries only an offset+count into a global compartment buffer.
// ---------------------------------------------------------------------------
struct NeuronDescriptor {
    NeuronKind kind;
    std::uint8_t pad0;
    std::uint16_t layer;      // cortical layer this soma sits in (0..5)
    std::uint32_t area_id;    // which cortical area this neuron belongs to
    union {
        HHState  hh;
        IzhState izh;
        struct { std::uint32_t comp_off; std::uint32_t comp_count; } mc;
    } state;
};

}  // namespace agi
