// SPDX-License-Identifier: MIT
// Neuromodulator broadcast: writes the current NeuromodSnapshot into
// __constant__ memory, where every subsequent kernel launched in this step
// will read from it via `d_nm_*`.

#include <cuda_runtime.h>
#include "agi/neuromod.hpp"

namespace agi::cuda {

// The four rail scalars, in __constant__ memory (broadcast by every launch).
__constant__ float d_nm_da  = 0.0f;
__constant__ float d_nm_ne  = 0.5f;
__constant__ float d_nm_ach = 0.1f;
__constant__ float d_nm_ht5 = 0.5f;

extern "C" cudaError_t launch_neuromod_broadcast(const agi::NeuromodSnapshot snap,
                                                 cudaStream_t stream) {
    cudaError_t err;
    err = cudaMemcpyToSymbolAsync(d_nm_da,  &snap.da,  sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbolAsync(d_nm_ne,  &snap.ne,  sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbolAsync(d_nm_ach, &snap.ach, sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbolAsync(d_nm_ht5, &snap.ht5, sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    return err;
}

}  // namespace agi::cuda
