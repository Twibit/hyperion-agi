# Build roadmap

Five phases, each ending with a runnable target that validates the phase's
scope against a biological benchmark.

## Phase 1 — Single-neuron fidelity (weeks 1–4)
* Port `neurorust::HodgkinHuxley` and `Izhikevich` to CUDA.
* Write `hh_step_kernel.cu` and `izh_step_kernel.cu`.
* Validate: HH is silent at I = 0, fires ≥ 3 spikes at I = 10 µA/cm²
  (matches `neurorust` integration test).
* Deliverable: `bench_single_neuron` runs 10⁶ neurons at ≥ 100× real time.

## Phase 2 — Column dynamics (weeks 5–10)
* Implement multi-compartment cable equation in Triton.
* Build the `Column` object with 5 layers × ~2000 neurons each.
* Validate: column exhibits gamma-band (30–80 Hz) oscillation under drive.
* Deliverable: `example_single_column` demo.

## Phase 3 — Networks and AER (weeks 11–18)
* Design the 48-bit AER packet, implement `aer_route.cu`.
* CSR SpMV where the input vector is a spike bit-vector.
* Small-world + rich-club overlay.
* Validate: 100-column simulation reproduces area-scale gamma synchrony.
* Deliverable: `bench_scaling` shows near-linear scaling to 100 columns.

## Phase 4 — Neuromodulation and learning (weeks 19–26)
* Broadcast rails for DA / 5-HT / NE / ACh.
* STDP kernel with three-factor gating.
* Replay engine + structural plasticity.
* Validate: network learns a delayed-reward association (Izhikevich 2007 replication).
* Deliverable: `example_dopamine_learning` demo.

## Phase 5 — Thalamo-cortical routing + attention (weeks 27–36)
* Software thalamus with attentional gain.
* Basal-ganglia WTA action selector.
* Validate: two-column attentional selection task under top-down PFC control.
* Deliverable: `example_attention_task` demo, whitepaper draft.
