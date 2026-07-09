# CortexOS IR — dialect specification

An MLIR-style dialect that expresses spiking cortical programs at a level
above CUDA and Triton, and lowers to them.

## 1. Types

| Type | Meaning |
|------|---------|
| `!agi.spike<t>` | An event with timestamp, source neuron, weight index |
| `!agi.neuron<kind>` | A neuron with a `NeuronKind` attribute (HH / Izh / MC) |
| `!agi.column<N,L>` | A column of N neurons across L layers |
| `!agi.csr<f8>` | Log-quantised CSR connectivity |
| `!agi.rail` | A neuromodulation scalar (0-D f32) |
| `!agi.trace<T,dt>` | A compressed membrane trace of length T at step dt |

## 2. Core ops

```mlir
// Instantiate an HH neuron
%n = agi.neuron.create {kind = #agi.hh, params = #agi.hh_default} : !agi.neuron<hh>

// Advance the neuron by one time step
%v_new, %spiked = agi.neuron.step %n, %i_in, %dt : !agi.neuron<hh>, f64, f64

// Build a column of 10 000 neurons, 5 layers deep
%c = agi.column.create {n = 10000, layers = 5} : !agi.column<10000, 5>

// Advance the whole column (lowers to column_step.triton.py)
%s = agi.column.step %c, %inputs, %dt : !agi.column<10000, 5>, tensor<10000xf32>, f64

// Emit a spike packet
%p = agi.spike.emit %src_id, %delay, %w_idx : !agi.spike<t>

// Route packets through the thalamus
%routed = agi.thalamus.route %packets, %attention : tensor<?x!agi.spike<t>>, tensor<Nxf32>

// STDP update on a connectivity block
agi.stdp.update %csr, %spikes_pre, %spikes_post, %da : !agi.csr<f8>, ...

// Neuromodulator broadcast
agi.rail.broadcast @da, %new_value : !agi.rail, f32

// Predictive coding pair
%pred, %err = agi.pcoding.split %c : !agi.column<10000, 5> -> (tensor<Nxf32>, tensor<Nxf32>)
```

## 3. Lowering

| IR op | Lowered to |
|-------|------------|
| `agi.neuron.step {kind = #hh}` | `hh_step_kernel.cu` |
| `agi.neuron.step {kind = #izh}` | `izh_step_kernel.cu` |
| `agi.column.step` | `column_step.triton.py` (fused) |
| `agi.thalamus.route` | `thalamus_route.cu` |
| `agi.stdp.update` | `stdp_update.cu` |
| `agi.rail.broadcast` | `neuromod_broadcast.cu` |
| `agi.spike.emit` in a hot loop | Fused into the calling `agi.column.step` |

## 4. Passes

1. **`--agi-fuse-column-step`** — Merges neuron.step + spike.emit + spmv into a single Triton column op.
2. **`--agi-quantise-weights`** — Rewrites f32 weights to 8-bit log-quantised.
3. **`--agi-schedule-events`** — Reorders spike deliveries into delay-slot buckets.
4. **`--agi-place-columns`** — Assigns columns to GPUs based on connectivity locality.
5. **`--agi-insert-replay-checkpoints`** — Inserts checkpoint ops for the replay engine.
