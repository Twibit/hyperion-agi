// SPDX-License-Identifier: MIT
// Minimum runnable example: build one cortical column of 1024 Izhikevich
// regular-spiking neurons, wire them with 5 % sparse recurrent connectivity,
// drive with Poisson input, and run for 1 s of simulated time.

#include <cstdio>
#include <memory>

#include "agi/runtime.hpp"

int main() {
    agi::RuntimeConfig cfg;
    cfg.dt_ms = 0.1;
    cfg.gpu_devices = {0};

    agi::Runtime rt(cfg);
    auto col = rt.make_column(1024, agi::NeuronKind::Izhikevich);

    // Local 5% sparse CSR (rec).
    auto csr = rt.make_csr(1024, 1024, /*nnz_reserve*/ 1024 * 50);
    for (std::uint32_t i = 0; i < 1024; ++i) {
        for (std::uint32_t j = 0; j < 1024; ++j) {
            if (i == j) continue;
            // 5 % sparse; weight = 0.5 mV drop; delay 1 ms; AMPA.
            if ((i * 733u + j * 991u) % 20u == 0u) {
                csr->push_edge(i, j, 0.5f, 1.0f, agi::SynapseKind::AMPA);
            }
        }
    }
    csr->finalize();
    col->set_local_csr(csr);

    // Neuromod baseline.
    rt.neuromod().set(agi::Rail::NE,  0.4f);
    rt.neuromod().set(agi::Rail::ACh, 0.2f);
    rt.neuromod().set(agi::Rail::HT5, 0.6f);
    rt.neuromod().set(agi::Rail::DA,  0.0f);

    // Run for 1000 ms of simulated biological time.
    const auto total = rt.run(1000.0);
    std::printf("Ran 1000 ms — total spikes: %llu\n",
                static_cast<unsigned long long>(total));
    std::printf("Column compressed footprint: %zu B (raw %zu B) — ratio %.2fx\n",
                col->memory().compressed_bytes(),
                col->memory().raw_bytes(),
                col->memory().compression_ratio());
    return 0;
}
