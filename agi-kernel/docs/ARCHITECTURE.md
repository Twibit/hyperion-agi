# CortexOS — Architecture Specification

Version 0.1 · Genspark AGI Hardware/Kernel Design Concept · 2026

---

## 0. Executive summary

CortexOS is a hardware/kernel stack whose primitives are chosen to reproduce the
computational motifs of biological neocortex at scale. It is designed around the
observation that AGI-relevant computation in mammals is:

* **Sparse in time** (mean cortical firing rate ~1–5 Hz, Attwell & Laughlin 2001),
* **Sparse in space** (~1–10 % local connectivity, Braitenberg & Schüz 1998),
* **Massively parallel** (~2 × 10¹⁰ cortical neurons in humans, Herculano-Houzel 2009),
* **Modulated globally by scalar chemical signals** (DA, 5-HT, NE, ACh),
* **Organised into repeating canonical circuits** (cortical columns).

Dense-matmul silicon (today's GPUs used purely as tensor cores) is a poor fit
for these workloads. CortexOS reuses the same GPUs but reframes them as
**spike-event processors** with dendritic co-processing, using CUDA for
integrate-and-fire dynamics and Triton for fused cable-equation blocks.

---

## 1. Biological substrate → engineering primitives

The mapping is one-to-one; each row is expanded below.

| Layer | Biology | CortexOS primitive | Kernel |
|-------|---------|--------------------|--------|
| L0 | Ion channels (Na⁺, K⁺, leak) | `HHState` struct (V, m, h, n) | `hh_step_kernel.cu` |
| L1 | Single neuron (soma + dendrites) | `Neuron` = tag ∈ {HH, Izh, MC} + `Compartments[]` | `izh_step_kernel.cu`, `cable_step.triton.py` |
| L2 | Cortical minicolumn (~100 neurons, 6 layers) | `Minicolumn` = SM-warp-sized tile | `minicolumn_step.cu` |
| L3 | Cortical column (~10⁴ neurons) | `Column` = SM tile | `column_step.triton.py` |
| L4 | Cortical area (~10⁷ neurons) | `Area` = multi-SM / multi-GPU region | scheduler |
| L5 | Thalamo-cortical loop | `Thalamus` router with attentional gating | `thalamus_route.cu` |
| L6 | Basal ganglia (action selection) | `ActionSelector` — winner-take-all over channels | `wta_select.cu` |
| L7 | Neuromodulatory nuclei | 4 broadcast scalars: `da`, `ht5`, `ne`, `ach` | broadcast rail |
| L8 | Hippocampus (episodic + replay) | Ring buffer of AER packets, replay engine | `replay.cu` + host |
| L9 | Whole brain | `Runtime` session | `scheduler.cpp` |

### 1.1 Level of biophysical fidelity is a knob

Every neuron in CortexOS is stamped with a `NeuronKind` enum (from
`neurorust`):

```cpp
enum class NeuronKind : uint8_t { HodgkinHuxley = 0, Izhikevich = 1, MultiCompartment = 2 };
```

The scheduler dispatches a **kernel per kind** on the GPU, so a heterogeneous
network (HH pyramidal cells wired to Izhikevich interneurons wired to
multi-compartment L5 pyramids) is the default rather than a special case.
This is essential because different cortical cell types genuinely need
different fidelities: L5 pyramidal apical tufts require multi-compartment
dendrites to reproduce NMDA plateaus (Larkum 2013), while cortical PV+
interneurons are well approximated by 2-state Izhikevich fast-spiking.

---

## 2. Data plane

### 2.1 Spike Address-Event Representation (AER)

The wire format is a **48-bit spike packet**, following Silicon-Retina AER
(Boahen 2000):

```
┌──────────────┬───────────┬──────────────┬───────────┐
│ neuron_id 32 │ delay 8   │ weight_idx 4 │ meta 4    │
└──────────────┴───────────┴──────────────┴───────────┘
```

* 32-bit `neuron_id` addresses up to 4 × 10⁹ neurons per shard.
* 8-bit `delay` encodes axonal delay in 0.1 ms steps (25.5 ms max — matches
  cortical axonal delay distribution).
* 4-bit `weight_idx` picks one of 16 synapse types (AMPA, NMDA, GABA-A, GABA-B, …).
* 4-bit `meta` reserved (dopamine tag, replay marker, …).

Density-based routing: at physiological firing rates (~2 Hz) a 10⁹-neuron shard
emits ~2 × 10⁹ spikes/s = 12 GB/s of spike traffic — fits comfortably in
NVLink 5 (1.8 TB/s) and PCIe-Gen6.

### 2.2 Connectivity

CSR (Compressed Sparse Row) as in `neurorust::SparseConnectivity`, extended
with a **hub/rich-club overlay** matching Bullmore & Sporns (2009):

* 90 % of edges: local (same column or neighbouring column), stored per-column.
* 10 % of edges: long-range hub edges, stored in a global CSR replicated on
  every GPU.

Weight quantisation: 8-bit log-normal, decoded via LUT on the SM — a real
cortical synapse's weight distribution is heavy-tailed log-normal
(Song et al. 2005), so 256 discrete levels is not a constraint.

### 2.3 Memory tiers

Following `neurorust`'s hybrid-codec principle, but scaled across a physical
memory hierarchy:

| Tier | Media | Content | Codec |
|------|-------|---------|-------|
| T0 | Registers / SMEM | Current-step V, gating vars | dense f16 |
| T1 | HBM3e | Column state, local CSR, recent 100 ms of spikes | packed |
| T2 | DDR5 | Long-range CSR, 10-s replay buffer, membrane traces | i16-quantised |
| T3 | NVMe | Episodic memory, off-line replay corpus | delta-ISI + zstd |
| T4 | Cold storage | Consolidated weights, model snapshots | zstd + bincode |

Same four codecs as `neurorust`:
* **Spike train** → delta-ISI + zstd (T3, T4)
* **Membrane trace** → affine i16 quantisation (T2)
* **Connectivity** → CSR (T1, T2)
* **Population raster** → bit-packed (T1)

---

## 3. Compute plane

### 3.1 Kernel taxonomy

| Kernel | Language | Purpose | Complexity |
|--------|----------|---------|------------|
| `hh_step_kernel.cu` | CUDA | Hodgkin-Huxley RK4 step, warp-per-neuron | O(N) |
| `izh_step_kernel.cu` | CUDA | Izhikevich 2-ODE step, 32 neurons/warp | O(N) |
| `cable_step.triton.py` | Triton | Multi-compartment cable eq., Thomas algorithm block | O(N·K) |
| `aer_route.cu` | CUDA | Bitonic spike router with delay slots | O(S log S) |
| `csr_spmv_spike.cu` | CUDA | SpMV where x is a bit-vector (spikes) | O(nnz) |
| `stdp_update.cu` | CUDA | Pair-based STDP with neuromod gating | O(active_syn) |
| `thalamus_route.cu` | CUDA | Attentional gain + spike relay | O(N) |
| `replay.cu` | CUDA | Reverse-order replay of ring buffer | O(N·T) |
| `column_step.triton.py` | Triton | Fused: cable + synapse + spike gen | O(N·K) |
| `neuromod_broadcast.cu` | CUDA | Update 4 global scalars from reward | O(1) |
| `wta_select.cu` | CUDA | Basal-ganglia-style winner-take-all | O(actions) |

### 3.2 Why Triton for the fused column step

The multi-compartment cable equation is a **block-tridiagonal solve** — the
Thomas algorithm ports beautifully to Triton because:

1. The block size (typically 5–50 compartments per neuron) fits in SRAM.
2. `tl.dot` can fuse the synaptic input `Wx` with the tridiagonal factorisation.
3. Autotuning over `BLOCK_M` (# neurons in a column) is essential — the sweet
   spot moves with cortical layer (L2/3 vs. L5).

### 3.3 Why CUDA for the point-neuron kernels

Warp-level primitives (`__shfl_sync`, `__ballot_sync`) are perfect for:

* Warp-wide reduction of spike bits into a 32-bit mask,
* Fast atomic AER packet enqueue via `atomicAdd`,
* Cooperative-groups-based STDP accumulation.

Triton's abstraction over these primitives is currently thinner than raw CUDA.

---

## 4. Control plane

### 4.1 Event-driven scheduler

Time is discretised at `dt = 0.1 ms` (biophysically justified — action
potentials are ~1 ms wide, and Hodgkin-Huxley is stable at dt ≤ 0.02 ms
per RK4 substep; four substeps per outer step).

Each outer step:

```
for t in [0, T_sim] step dt:
    1. Deliver AER packets whose (t_emit + delay) == t             (aer_route)
    2. For each column tile:
         a. Sum synaptic current from delivered packets             (csr_spmv_spike)
         b. Advance neuron dynamics                                 (hh_step / izh_step / cable_step)
         c. Emit new AER packets for any spikers
    3. Update neuromodulatory broadcast rails                       (neuromod_broadcast)
    4. Gated STDP update on any recently-active synapses            (stdp_update)
    5. Push new spikes into hippocampal ring buffer                 (host)
    6. If sleep phase, run replay engine                            (replay)
```

The 6 steps are pipelined across streams to hide launch latency:
`stream[0]` = routing, `stream[1]` = neuron step, `stream[2]` = plasticity,
`stream[3]` = replay/checkpoint.

### 4.2 Thalamo-cortical routing

The thalamus in mammals is not just a relay — it implements attention by
gating cortico-cortical traffic (Halassa & Kastner 2017). CortexOS mirrors
this: cortico-cortical AER packets pass through a **software thalamus**
before reaching their destination column. The thalamus applies:

1. **Attentional gain** — per-source-column scalar, controlled by prefrontal
   area (a designated column region).
2. **Bursting relay mode** — pass-through when `ne` (norepinephrine) is high,
   suppressed otherwise (matches locus-coeruleus arousal).
3. **Pulvinar-style cross-area routing** — combines two source areas into a
   third's input, enabling flexible cognitive control.

### 4.3 Neuromodulation

Four global scalars, updated every ms:

| Rail | Biology | Effect in CortexOS |
|------|---------|--------------------|
| `da` (dopamine) | Reward-prediction error (Schultz 1998) | Gates STDP sign: positive `da` → LTP, negative → LTD |
| `ne` (norepinephrine) | Arousal / uncertainty (Aston-Jones 2005) | Scales neuronal gain (`g_leak` inversely) |
| `ach` (acetylcholine) | Novelty / attention (Sarter 2005) | Increases signal-to-noise by lowering firing threshold |
| `ht5` (serotonin) | Patience / discounting (Doya 2002) | Modulates temporal discount γ in RL heads |

They are broadcast via constant-memory scalars — updated once per step, read
by every kernel launched in that step.

---

## 5. Learning plane

### 5.1 Local plasticity

**STDP** with a pair-based rule and neuromodulatory gate (Frémaux & Gerstner 2016):

```
Δw_ij = da · [ A₊ · exp(−Δt/τ₊)  if post-before-pre  (LTP)
             − A₋ · exp(+Δt/τ₋)  if pre-before-post  (LTD) ]
```

The eligibility trace `e_ij` decays with τ_e ≈ 500 ms, letting sparse rewards
consolidate correlated activity from up to 1 s in the past — this is the
"three-factor" learning rule of Izhikevich (2007), and it is what turns
Hebbian correlation into value-driven credit assignment.

### 5.2 Structural plasticity (offline)

During "sleep" (replay) phases, weak synapses (`|w| < ε`) are pruned and new
random synapses are grown between highly co-active neurons — this is the
computational analogue of synaptic pruning during slow-wave sleep
(Tononi & Cirelli 2014, synaptic homeostasis hypothesis).

### 5.3 Global plasticity — neuromodulator tuning

The four broadcast scalars are themselves learnable — a small "meta"
network (a few thousand neurons in a designated PFC-like region) receives
task reward and outputs the neuromod rail values. This closes the loop
between task performance and system-wide arousal / patience / novelty.

---

## 6. I/O plane

### 6.1 Sensors → spikes

Any continuous input `x(t)` is converted to spikes via one of:

* **Rate coding** — Poisson process with `λ = f(x)`.
* **Latency coding** — first-spike time `= τ · (x_max − x) / x_max`.
* **Population coding** — Gaussian receptive fields tile the range.

Reference implementation: `src/io/sensor_encoders.hpp` + a CUDA kernel per code.

### 6.2 Spikes → actions

The motor output is a set of "action channels", each a small pool of neurons.
Winner-take-all over their firing rates selects the emitted action; ties are
broken stochastically to preserve exploration (linked to `ht5` for a
temperature knob).

---

## 7. Scaling model

For a target of **10¹⁰ neurons, 10¹⁴ synapses** (~1 % human cortex):

| Resource | Requirement | Basis |
|----------|-------------|-------|
| Neuron state (dense) | ~600 GB | 60 bytes/neuron × 10¹⁰ |
| Local CSR (T1) | ~200 TB | 10 bytes/synapse × 90 % of 10¹⁴ |
| Long-range CSR (T2) | ~20 TB | replicated per GPU |
| Spike traffic (steady state) | ~15 TB/s aggregate | 2 Hz × 10¹⁰ × 6 B |
| Compute | ~100 PF/s equivalent | 100 flop/neuron/step × 10⁴ step/s |
| GPU count (H100-class) | ~1000 | limited by HBM capacity |

This is **within one order of magnitude** of a 2026-era exascale system. The
memory tier codecs reduce the 200 TB local CSR to ~30 TB effective, which
brings the GPU count back under 500.

---

## 8. Failure modes and mitigations

| Failure | Mitigation |
|---------|------------|
| Runaway excitation (epileptic-like) | Global inhibition column, gain-limited via `ne` |
| Silent network (nothing fires) | Cholinergic boost via `ach`, and Poisson noise floor |
| Catastrophic forgetting | Replay engine + weight consolidation à la elastic weight consolidation |
| Weight overflow | 8-bit log-quant clipping + slow decay in absence of use |
| GPU drop-out | Checkpoint every 10 s of sim-time to NVMe (T3) |

---

## 9. What is deliberately *not* here

CortexOS is a substrate, not a cognitive architecture. It does **not**
prescribe:

* How language is grounded — that is the job of an application layer.
* Which learning objective is used at the top — RL, self-supervised, or a
  free-energy formulation all compose over the same substrate.
* Any commitment to symbolic vs. sub-symbolic representations — both
  emerge naturally from column dynamics.

This is intentional. The claim is that if AGI is possible on silicon, its
substrate must at minimum look like this; what runs on top of it is an
open research question.

---

## 10. References

* Hodgkin, A. L., & Huxley, A. F. (1952). *A quantitative description of membrane current and its application to conduction and excitation in nerve.* J. Physiol.
* Izhikevich, E. M. (2003). *Simple model of spiking neurons.* IEEE Trans. Neural Netw.
* Izhikevich, E. M. (2007). *Solving the distal reward problem through linkage of STDP and dopamine signaling.* Cerebral Cortex.
* Mountcastle, V. B. (1957). *Modality and topographic properties of single neurons of cat's somatic sensory cortex.* J. Neurophysiol.
* Douglas, R. J., & Martin, K. A. C. (2004). *Neuronal circuits of the neocortex.* Annu. Rev. Neurosci.
* London, M., & Häusser, M. (2005). *Dendritic computation.* Annu. Rev. Neurosci.
* Rao, R. P. N., & Ballard, D. H. (1999). *Predictive coding in the visual cortex.* Nat. Neurosci.
* Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nat. Rev. Neurosci.
* Bi, G., & Poo, M. (1998). *Synaptic modifications in cultured hippocampal neurons.* J. Neurosci.
* Frémaux, N., & Gerstner, W. (2016). *Neuromodulated STDP and theory of three-factor learning rules.* Front. Neural Circuits.
* Bullmore, E., & Sporns, O. (2009). *Complex brain networks.* Nat. Rev. Neurosci.
* Herculano-Houzel, S. (2009). *The human brain in numbers.* Front. Hum. Neurosci.
* Halassa, M. M., & Kastner, S. (2017). *Thalamic functions in distributed cognitive control.* Nat. Neurosci.
* Tononi, G., & Cirelli, C. (2014). *Sleep and the price of plasticity.* Neuron.
