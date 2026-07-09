// SPDX-License-Identifier: MIT
// Scaling benchmark: measure sim-time throughput as # columns grows.
//
// Prints a table:
//   n_columns | neurons | wall_ms/sim_s | speed vs. real-time
//
// Design-skeleton form: allocates and steps; timing is coarse (chrono).

#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

#include "agi/runtime.hpp"

int main() {
    agi::Runtime rt;
    std::printf("n_col  neurons  wall_ms   xRT\n");
    for (int n_col : {1, 4, 16, 64}) {
        std::vector<std::shared_ptr<agi::Column>> cols;
        for (int c = 0; c < n_col; ++c) {
            auto col = rt.make_column(1024, agi::NeuronKind::Izhikevich);
            auto csr = rt.make_csr(1024, 1024);
            csr->finalize();
            col->set_local_csr(csr);
            cols.push_back(col);
        }
        const auto t0 = std::chrono::steady_clock::now();
        rt.run(100.0);   // 100 ms of biological time
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double xrt = 100.0 / ms;
        std::printf("%5d  %7d  %7.2f  %5.2fx\n", n_col, n_col * 1024, ms, xrt);
    }
    return 0;
}
