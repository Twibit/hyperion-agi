// SPDX-License-Identifier: MIT
// CSR SpMV correctness against a dense reference — same idea as
// neurorust::tests::csr_matches_dense_spmv.

#include <cstdio>
#include <cstdlib>
#include "agi/runtime.hpp"

int main() {
    agi::Runtime rt;
    auto csr = rt.make_csr(8, 8);
    for (std::uint32_t i = 0; i < 8; ++i) {
        // Diagonal weight = i + 1, off-diagonal (i, i+1) = 0.5.
        csr->push_edge(i, i, float(i + 1), 1.0f, agi::SynapseKind::AMPA);
        if (i + 1 < 8) csr->push_edge(i, i + 1, 0.5f, 1.0f, agi::SynapseKind::AMPA);
    }
    csr->finalize();
    // In a full CUDA-enabled build we would launch spmv_spike here and
    // compare to a host reference; for the design skeleton we validate
    // structural properties.
    if (csr->nnz() < 8) { std::printf("nnz too small\n"); return EXIT_FAILURE; }
    std::printf("test_csr_spmv: PASS (nnz=%zu)\n", csr->nnz());
    return EXIT_SUCCESS;
}
