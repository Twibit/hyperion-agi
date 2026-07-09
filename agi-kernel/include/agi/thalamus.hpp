// SPDX-License-Identifier: MIT
// Thalamo-cortical router.
//
// The mammalian thalamus is not a passive relay — it implements attention
// and cross-area routing (Sherman & Guillery 2006; Halassa & Kastner 2017).
// CortexOS's `Thalamus` intercepts every cortico-cortical AER packet before
// delivery and:
//   1. Multiplies the packet's effective weight by a per-source attentional
//      gain scalar (top-down control, written by a PFC-like column region).
//   2. Switches between *tonic* relay (transparent) and *bursting* relay
//      (rate-limited) based on the noradrenergic rail `ne`.
//   3. Optionally re-routes packets from area A to area C conditioned on
//      the activity of area B (pulvinar-style cognitive control,
//      Saalmann et al. 2012).

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "agi/core.hpp"

namespace agi {

// Attention vector — one gain scalar per source column. Lives on device.
class AttentionVector {
 public:
    explicit AttentionVector(std::size_t n_source_columns);
    ~AttentionVector();

    // Set the gain for a single source column. Values are clamped to [0, 4].
    Status set(column_id_t source, float gain);
    Status set_all_device(const float* d_gains);  // bulk update from PFC

    const float* device_ptr() const noexcept { return d_gains_; }
    std::size_t size() const noexcept { return n_; }

 private:
    std::size_t n_;
    float* d_gains_ = nullptr;
};

// A single "thalamic nucleus" — a router responsible for a slice of the
// AER traffic. Multiple nuclei may run in parallel (LGN for vision,
// MGB for audition, pulvinar for cross-area, etc.).
class ThalamicNucleus {
 public:
    ThalamicNucleus(std::string_view name, std::size_t n_source_columns);

    // Route a batch of AER packets in place, applying attentional gain,
    // bursting/tonic mode selection, and pulvinar re-routing if configured.
    Status route(SpikePacket* d_packets, std::size_t n_packets,
                 const AttentionVector& attention,
                 float ne_scalar,   // norepinephrine level (0..1)
                 cudaStream_t stream = nullptr);

    // Configure pulvinar-style re-routing: when area B (`gate_area`) is
    // active above `gate_threshold` Hz, packets from area A (`from_area`)
    // are duplicated to area C (`to_area`).
    Status configure_pulvinar(std::uint32_t from_area,
                              std::uint32_t gate_area,
                              std::uint32_t to_area,
                              float gate_threshold_hz);

 private:
    std::string name_;
    std::size_t n_source_columns_;
    struct PulvinarRule {
        std::uint32_t from_area, gate_area, to_area;
        float gate_threshold_hz;
    };
    std::vector<PulvinarRule> pulvinar_rules_;
};

}  // namespace agi
