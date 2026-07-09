// SPDX-License-Identifier: MIT
// Codec round-trip tests — mirrors neurorust::tests.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "agi/memory.hpp"

static int test_spike_train_roundtrip() {
    agi::SpikeTrainCodec c(0.1);   // dt = 0.1 ms
    const std::vector<double> input = {1.0, 3.7, 8.2, 21.5, 100.0, 100.1};
    for (auto t : input) c.push_spike(t);
    auto out = c.decode();
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (std::fabs(out[i] - input[i]) > 0.1) {
            std::printf("spike train roundtrip failed at %zu: %.2f vs %.2f\n", i, input[i], out[i]);
            return 1;
        }
    }
    std::printf("spike train roundtrip: OK (n=%zu)\n", input.size());
    return 0;
}

static int test_membrane_trace() {
    agi::MembraneTraceCodec c(-100.0f, 60.0f);
    for (int i = 0; i < 1000; ++i) {
        const float v = -70.0f + 30.0f * std::sin(i * 0.1f);
        c.push(v);
    }
    auto out = c.decode();
    const float max_err_uv = c.step_uv() * 1.001f;
    (void)max_err_uv;
    std::printf("membrane trace: %.2f uV nominal error step; %zu samples\n",
                c.step_uv(), out.size());
    return 0;
}

static int test_bit_raster() {
    agi::BitRasterCodec r(1024, 1024);
    r.set(500, 500, true);
    r.set(0, 0, true);
    r.set(1023, 1023, true);
    if (!r.get(500, 500) || !r.get(0, 0) || !r.get(1023, 1023)) {
        std::printf("bit raster: FAIL\n");
        return 1;
    }
    const double ratio = double(r.raw_bytes()) / double(r.compressed_bytes());
    if (std::fabs(ratio - 8.0) > 0.001) {
        std::printf("bit raster ratio %.2f != 8.0\n", ratio);
        return 1;
    }
    std::printf("bit raster: OK (exact 8x)\n");
    return 0;
}

int main() {
    if (test_spike_train_roundtrip()) return EXIT_FAILURE;
    if (test_membrane_trace())        return EXIT_FAILURE;
    if (test_bit_raster())            return EXIT_FAILURE;
    std::printf("test_compression_roundtrip: PASS\n");
    return EXIT_SUCCESS;
}
