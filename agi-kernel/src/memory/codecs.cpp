// SPDX-License-Identifier: MIT
// Codec implementations — direct port of neurorust::compression to C++.

#include "agi/memory.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace agi {

// ---------------------------------------------------------------------------
// SpikeTrainCodec — delta-ISI encoding. (zstd wrapping omitted here to keep
// the design skeleton dependency-free; drop-in `zstd_compress()` call goes
// where the comment says.)
// ---------------------------------------------------------------------------
SpikeTrainCodec::SpikeTrainCodec(time_ms_t dt_ms) : dt_ms_(dt_ms) {}

Status SpikeTrainCodec::push_spike(time_ms_t t_ms) {
    const std::uint64_t step = static_cast<std::uint64_t>(t_ms / dt_ms_);
    const std::uint64_t isi = step - last_step_;
    last_step_ = step;

    // Varint encode ISI (LEB128).
    std::uint64_t v = isi;
    while (v >= 0x80) {
        payload_.push_back(static_cast<std::uint8_t>((v & 0x7F) | 0x80));
        v >>= 7;
    }
    payload_.push_back(static_cast<std::uint8_t>(v));

    // NOTE: for production, run `payload_ = zstd_compress(payload_)` at each
    // 4 KiB boundary; the current implementation is the "before-zstd" stage.

    ++n_spikes_;
    raw_bytes_ += sizeof(double);
    return Status::Ok;
}

std::vector<time_ms_t> SpikeTrainCodec::decode() const {
    std::vector<time_ms_t> out;
    out.reserve(n_spikes_);
    std::uint64_t step = 0;
    std::size_t i = 0;
    while (i < payload_.size()) {
        std::uint64_t v = 0;
        int shift = 0;
        for (;;) {
            const std::uint8_t b = payload_[i++];
            v |= (std::uint64_t(b & 0x7F) << shift);
            if (!(b & 0x80)) break;
            shift += 7;
        }
        step += v;
        out.push_back(step * dt_ms_);
    }
    return out;
}

// ---------------------------------------------------------------------------
// MembraneTraceCodec — affine i16 quantisation, ≤ 5 µV over a 160 mV window.
// ---------------------------------------------------------------------------
MembraneTraceCodec::MembraneTraceCodec(float v_min_mv, float v_max_mv)
    : v_min_(v_min_mv), v_max_(v_max_mv) {}

Status MembraneTraceCodec::push(float v_mv) {
    const float t = (v_mv - v_min_) / (v_max_ - v_min_);
    const int q = static_cast<int>(std::round(t * 65535.0f)) - 32768;
    q16_.push_back(static_cast<std::int16_t>(std::clamp(q, -32768, 32767)));
    return Status::Ok;
}

std::vector<float> MembraneTraceCodec::decode() const {
    std::vector<float> out;
    out.reserve(q16_.size());
    for (auto q : q16_) {
        const float t = (float(q) + 32768.0f) / 65535.0f;
        out.push_back(v_min_ + t * (v_max_ - v_min_));
    }
    return out;
}

// ---------------------------------------------------------------------------
// BitRasterCodec — bit-packed population raster (exact 8× shrink).
// ---------------------------------------------------------------------------
BitRasterCodec::BitRasterCodec(std::size_t n_neurons, std::size_t n_bins)
    : n_neurons_(n_neurons), n_bins_(n_bins) {
    bits_.assign((n_neurons * n_bins + 7) / 8, 0);
}

Status BitRasterCodec::set(std::size_t neuron, std::size_t bin, bool spiked) {
    if (neuron >= n_neurons_ || bin >= n_bins_) return Status::ShapeMismatch;
    const std::size_t idx = neuron * n_bins_ + bin;
    if (spiked) bits_[idx >> 3] |=  (1u << (idx & 7));
    else        bits_[idx >> 3] &= ~(1u << (idx & 7));
    return Status::Ok;
}

bool BitRasterCodec::get(std::size_t neuron, std::size_t bin) const {
    const std::size_t idx = neuron * n_bins_ + bin;
    return (bits_[idx >> 3] >> (idx & 7)) & 1u;
}

// ---------------------------------------------------------------------------
// NeuronMemory — arena bundling the three codecs.
// ---------------------------------------------------------------------------
NeuronMemory::NeuronMemory(time_ms_t dt_ms, float v_min_mv, float v_max_mv,
                           std::size_t n_neurons, std::size_t n_bins)
    : spikes_(dt_ms),
      membrane_(v_min_mv, v_max_mv),
      raster_(n_neurons, n_bins) {}

}  // namespace agi
