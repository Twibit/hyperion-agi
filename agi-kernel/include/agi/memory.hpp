// SPDX-License-Identifier: MIT
// Hybrid per-datum memory arena — scaled port of neurorust::NeuronMemory.
//
// Four codecs, one arena per column:
//
//   spike_train      SpikeTrainCodec     lossless   delta-ISI + zstd
//   membrane_trace   MembraneTraceCodec  lossy 5uV  affine i16 quantisation
//   connectivity     handled by SparseConnectivity (CSR)
//   population_raster BitRasterCodec     lossless   bit-packed 8x reduction
//
// The lossy V-trace codec is only lossy *on paper*: over a 160 mV window,
// i16 quantisation gives 2.4 µV resolution, several orders of magnitude
// below the ~1 mV intrinsic membrane noise of cortical neurons.

#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

#include "agi/core.hpp"

namespace agi {

// ---------------------------------------------------------------------------
// 1. Spike train codec (lossless)
// ---------------------------------------------------------------------------
class SpikeTrainCodec {
 public:
    explicit SpikeTrainCodec(time_ms_t dt_ms);
    Status push_spike(time_ms_t t_ms);
    std::vector<time_ms_t> decode() const;   // for tests / analysis
    std::size_t compressed_bytes() const noexcept { return payload_.size(); }
    std::size_t raw_bytes() const noexcept { return raw_bytes_; }
    std::size_t n_spikes() const noexcept { return n_spikes_; }

 private:
    time_ms_t dt_ms_;
    std::uint64_t last_step_ = 0;
    std::vector<std::uint8_t> payload_;   // zstd(varint(ISI))
    std::size_t n_spikes_ = 0;
    std::size_t raw_bytes_ = 0;
};

// ---------------------------------------------------------------------------
// 2. Membrane trace codec (lossy, ≤ 5 µV error)
// ---------------------------------------------------------------------------
class MembraneTraceCodec {
 public:
    MembraneTraceCodec(float v_min_mv, float v_max_mv);
    Status push(float v_mv);
    std::vector<float> decode() const;
    std::size_t compressed_bytes() const noexcept { return q16_.size() * 2; }
    std::size_t raw_bytes() const noexcept { return q16_.size() * 4; }
    // Nominal per-sample quantisation error, µV.
    float step_uv() const noexcept { return (v_max_ - v_min_) / 65535.0f * 1000.0f; }

 private:
    float v_min_, v_max_;
    std::vector<std::int16_t> q16_;
};

// ---------------------------------------------------------------------------
// 3. Bit-raster codec (lossless, exact 8× shrink)
// ---------------------------------------------------------------------------
class BitRasterCodec {
 public:
    BitRasterCodec(std::size_t n_neurons, std::size_t n_bins);
    Status set(std::size_t neuron, std::size_t bin, bool spiked);
    bool   get(std::size_t neuron, std::size_t bin) const;
    std::size_t compressed_bytes() const noexcept { return bits_.size() / 8; }
    std::size_t raw_bytes() const noexcept { return n_neurons_ * n_bins_; }

 private:
    std::size_t n_neurons_, n_bins_;
    std::vector<std::uint8_t> bits_;
};

// ---------------------------------------------------------------------------
// Arena — one per Column
// ---------------------------------------------------------------------------
class NeuronMemory {
 public:
    NeuronMemory(time_ms_t dt_ms, float v_min_mv, float v_max_mv,
                 std::size_t n_neurons, std::size_t n_bins);

    SpikeTrainCodec&    spikes()   { return spikes_; }
    MembraneTraceCodec& membrane() { return membrane_; }
    BitRasterCodec&     raster()   { return raster_; }

    // Total compressed footprint of this arena, in bytes.
    std::size_t compressed_bytes() const noexcept {
        return spikes_.compressed_bytes()
             + membrane_.compressed_bytes()
             + raster_.compressed_bytes();
    }
    std::size_t raw_bytes() const noexcept {
        return spikes_.raw_bytes()
             + membrane_.raw_bytes()
             + raster_.raw_bytes();
    }
    double compression_ratio() const noexcept {
        return raw_bytes() > 0 ? double(raw_bytes()) / double(compressed_bytes()) : 1.0;
    }

 private:
    SpikeTrainCodec spikes_;
    MembraneTraceCodec membrane_;
    BitRasterCodec raster_;
};

}  // namespace agi
