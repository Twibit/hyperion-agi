// SPDX-License-Identifier: MIT
// Column implementation — dispatches to the right kernel per NeuronKind.

#include "agi/column.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

#include "agi/neuron.hpp"

// Forward decls of CUDA launchers (implemented in src/cuda/*.cu).
extern "C" cudaError_t launch_hh_step(agi::HHState*, agi::HHParams,
                                      const float*, const float*,
                                      std::uint32_t*, float, std::uint32_t, cudaStream_t);
extern "C" cudaError_t launch_izh_step(agi::IzhState*, const agi::IzhParams*,
                                       const float*, const float*,
                                       std::uint32_t*, float, std::uint32_t, cudaStream_t);
// Triton launcher is called via a small Python-side stub during design;
// in production the Triton kernel is AOT-compiled and this becomes a
// direct cudaLaunchKernel call.
extern "C" cudaError_t launch_column_step_triton(std::uint32_t n_neurons,
                                                 std::uint32_t max_nnz_per_row,
                                                 int neuron_kind,
                                                 float dt_ms, cudaStream_t stream,
                                                 void* args_pack);

namespace agi {

Column::Column(column_id_t id, std::uint32_t n_neurons, NeuronKind kind, time_ms_t dt_ms)
    : id_(id), n_neurons_(n_neurons), kind_(kind), dt_ms_(dt_ms),
      memory_(dt_ms, /*v_min*/ -100.0f, /*v_max*/ 60.0f,
              /*n_neurons*/ n_neurons, /*n_bins*/ 4096) {
    // Allocate device state SoA sized to the neuron kind.
    std::size_t state_bytes = 0;
    switch (kind) {
        case NeuronKind::HodgkinHuxley:
            state_bytes = n_neurons * sizeof(HHState);
            break;
        case NeuronKind::Izhikevich:
            state_bytes = n_neurons * sizeof(IzhState);
            break;
        case NeuronKind::MultiCompartment:
            // Real allocation happens after set_layer() defines compartment counts.
            state_bytes = 0;
            break;
    }
    if (state_bytes > 0) {
        cudaMalloc(&d_state_soa_, state_bytes);
    }
    cudaMalloc(&d_syn_current_, n_neurons * sizeof(float));
    cudaMalloc(&d_spike_bits_,  ((n_neurons + 31) / 32) * sizeof(std::uint32_t));
    cudaMemset(d_syn_current_, 0, n_neurons * sizeof(float));
    cudaMemset(d_spike_bits_,  0, ((n_neurons + 31) / 32) * sizeof(std::uint32_t));
}

Column::~Column() {
    if (d_state_soa_)    cudaFree(d_state_soa_);
    if (d_syn_current_)  cudaFree(d_syn_current_);
    if (d_spike_bits_)   cudaFree(d_spike_bits_);
}

void Column::set_layer(std::uint8_t i, LayerSpan s) {
    if (i < 6) layers_[i] = s;
}
void Column::set_local_csr(std::shared_ptr<SparseConnectivity> csr) {
    local_csr_ = std::move(csr);
}
void Column::set_long_range_csr(std::shared_ptr<SparseConnectivity> csr) {
    long_range_csr_ = std::move(csr);
}

Status Column::step(const float* i_ext_device, std::uint32_t* out_spike_bits_device) {
    if (!local_csr_) return Status::ShapeMismatch;

    // 1. Compute synaptic current from local + long-range CSR (both accumulate
    //    into d_syn_current_ via atomicAdd, cleared first).
    cudaMemsetAsync(d_syn_current_, 0, n_neurons_ * sizeof(float), /*stream*/ 0);
    // The spike bit-vector consumed here is d_spike_bits_ from the *previous*
    // step — kept as a class member so it survives across step() calls.
    if (!ok(local_csr_->spmv_spike(d_spike_bits_, d_syn_current_))) {
        return Status::CudaLaunchFailed;
    }
    if (long_range_csr_) {
        long_range_csr_->spmv_spike(d_spike_bits_, d_syn_current_);
    }

    // 2. Advance neuron dynamics.
    cudaError_t err = cudaSuccess;
    switch (kind_) {
        case NeuronKind::HodgkinHuxley: {
            HHParams params;   // default
            err = launch_hh_step(static_cast<HHState*>(d_state_soa_), params,
                                 d_syn_current_, i_ext_device,
                                 d_spike_bits_, static_cast<float>(dt_ms_),
                                 n_neurons_, /*stream*/ 0);
            break;
        }
        case NeuronKind::Izhikevich: {
            // In production, per-neuron IzhParams are stored on device;
            // for the design skeleton we assume a homogeneous RS preset.
            err = launch_izh_step(static_cast<IzhState*>(d_state_soa_),
                                  /*d_params*/ nullptr,   // → RS default in-kernel
                                  d_syn_current_, i_ext_device,
                                  d_spike_bits_, static_cast<float>(dt_ms_),
                                  n_neurons_, /*stream*/ 0);
            break;
        }
        case NeuronKind::MultiCompartment: {
            // Fused Triton column_step covers this path; the stub below is
            // resolved at load time to the AOT-compiled Triton object.
            err = launch_column_step_triton(n_neurons_, /*max_nnz*/ 128,
                                            /*kind*/ 2, static_cast<float>(dt_ms_),
                                            /*stream*/ 0, /*args*/ nullptr);
            break;
        }
    }
    if (err != cudaSuccess) return Status::CudaLaunchFailed;

    // 3. Publish the produced spike bit-vector.
    if (out_spike_bits_device) {
        cudaMemcpyAsync(out_spike_bits_device, d_spike_bits_,
                        ((n_neurons_ + 31) / 32) * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToDevice, 0);
    }
    return Status::Ok;
}

Status Column::predictive_split(float* prediction_out, float* error_out) {
    // Rao & Ballard (1999) motif: L5 pyramidal cells encode top-down
    // predictions; L2/3 cells encode residual errors. We approximate this
    // by splitting the column's rate over layer spans.
    //
    // For the design skeleton we defer the implementation to the runtime —
    // it involves reducing the column's spike bits over each layer and
    // writing per-neuron rate estimates. See src/host/column_pcoding.cpp.
    (void)prediction_out; (void)error_out;
    return Status::Ok;
}

}  // namespace agi
