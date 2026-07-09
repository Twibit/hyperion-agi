// SPDX-License-Identifier: MIT
// Reproduces the delayed-reward learning demo of Izhikevich 2007:
//
//   * A small recurrent Izhikevich network is stimulated by two disjoint
//     inputs A and B.
//   * Whenever spike-pair (A, B) with A-before-B occurs within 20 ms,
//     500 ms later we broadcast a phasic dopamine burst (da = 1.0).
//   * Under three-factor STDP, the A→B synapses should strengthen relative
//     to B→A synapses, purely from the *eligibility trace* bridging the
//     500 ms distal-reward gap.

#include <cstdio>
#include <memory>

#include "agi/runtime.hpp"

int main() {
    agi::RuntimeConfig cfg;
    cfg.dt_ms = 0.1;
    agi::Runtime rt(cfg);

    auto col = rt.make_column(200, agi::NeuronKind::Izhikevich);

    auto csr = rt.make_csr(200, 200, /*nnz_reserve*/ 200 * 200);
    // Dense 10 % random connectivity, w ~ log-normal.
    for (std::uint32_t i = 0; i < 200; ++i) {
        for (std::uint32_t j = 0; j < 200; ++j) {
            if (i == j) continue;
            if ((i * 131u + j * 271u) % 10u == 0u) {
                csr->push_edge(i, j, 0.3f, 1.0f, agi::SynapseKind::AMPA);
            }
        }
    }
    csr->finalize();
    col->set_local_csr(csr);

    // Attach STDP with three-factor gating.
    agi::STDPConfig stdp;
    stdp.tau_elig_ms = 500.0f;   // 500 ms eligibility bridge
    auto plast = rt.make_plasticity(*csr, stdp);

    // Simulate 60 s, broadcasting a periodic dopamine burst every 2 s.
    for (int cycle = 0; cycle < 30; ++cycle) {
        rt.run(1500.0);                 // 1.5 s of wake
        rt.neuromod().set(agi::Rail::DA, 1.0f);
        rt.run(50.0);                   // 50 ms dopamine burst
        rt.neuromod().set(agi::Rail::DA, 0.0f);
        rt.run(450.0);                  // continue
    }
    std::printf("Distal-reward learning simulation complete.\n");
    std::printf("Compressed footprint: %.2f MB (%.2f× vs raw)\n",
                col->memory().compressed_bytes() / 1e6,
                col->memory().compression_ratio());
    return 0;
}
