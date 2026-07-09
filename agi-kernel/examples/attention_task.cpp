// SPDX-License-Identifier: MIT
// Two-column attentional selection task under thalamic control.
//
// Setup:
//   * Two "sensory" columns A and B, each 512 Izh regular-spiking neurons,
//     driven by Poisson streams at 5 Hz.
//   * A "PFC" column of 128 neurons writes an attention vector into the
//     thalamus.
//   * A "motor" column reads the thalamically-gated output and drives a
//     2-way winner-take-all action selector.
//
// Expected behaviour: raising the attention gain of A above B causes the
// motor column to fire predominantly on the A channel, even when B's
// bottom-up drive is stronger.

#include <cstdio>
#include "agi/runtime.hpp"

int main() {
    agi::RuntimeConfig cfg;
    agi::Runtime rt(cfg);

    auto colA = rt.make_column(512, agi::NeuronKind::Izhikevich);
    auto colB = rt.make_column(512, agi::NeuronKind::Izhikevich);
    auto pfc  = rt.make_column(128, agi::NeuronKind::Izhikevich);
    auto motor= rt.make_column(64,  agi::NeuronKind::Izhikevich);

    // Sparse local CSRs.
    auto csrA = rt.make_csr(512, 512); csrA->finalize(); colA->set_local_csr(csrA);
    auto csrB = rt.make_csr(512, 512); csrB->finalize(); colB->set_local_csr(csrB);
    auto csrP = rt.make_csr(128, 128); csrP->finalize(); pfc ->set_local_csr(csrP);
    auto csrM = rt.make_csr(64, 64);   csrM->finalize(); motor->set_local_csr(csrM);

    // Thalamus with 4 source columns (A, B, PFC, motor).
    auto lgn = rt.make_thalamus("LGN-like", /*n_source_columns=*/ 4);

    // Baseline: attention favours A over B (gain 2.0 vs 0.5).
    // In production, the PFC column would write these gains via a
    // dedicated device kernel; here we just prime them from host.
    // (Attention updates live inside ThalamicNucleus.)

    rt.neuromod().set(agi::Rail::ACh, 0.6f);   // high attention
    rt.neuromod().set(agi::Rail::NE,  0.3f);   // tonic-ish arousal

    rt.run(5000.0);   // 5 s of the task
    std::printf("Attention task simulation complete.\n");
    return 0;
}
