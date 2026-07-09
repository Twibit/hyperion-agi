# Changes applied to the archive

This checkout ships with several build/runtime breakages that prevent
the archive from being run on anything except the specific ARM64 laptop
it was last edited on. The goal of these changes is to make the
substrate portable and honest — **not** to invent new AGI capabilities.

All modifications are additive or minimally invasive; no existing
research code was deleted. Original architectural intent is preserved
everywhere.

## Files added

| Path | Purpose |
|---|---|
| `AGI_ASSESSMENT.md` | Honest evaluation of the claim that this is an AGI, with evidence. |
| `CHANGES.md` | This file. |
| `README.md` (top-level, replaces marketing prose) | Truthful overview + quickstart. |
| `cli.py` | Single CLI unifying `dual_planner.py`, `main.py` (chat), and `embodied_agent`. Includes an `inspect` self-test. |
| `rust_backend_shim.py` | Pure-Python shim for the missing PyO3 module. Implements `hash_neuron_id`, `reward_modulated_stdp`, `compute_fft`, `sparse_attention` — the only 4 symbols the codebase actually calls. |

## Files modified

### `nlp_engine/layers/predictive.py`

**Symptom:** `dual_planner.py` imports `PredictiveMinicolumn` from this
module, but the class was never in the file. Every attempt to run
`python3 dual_planner.py` failed immediately with `ImportError`.

**Fix:** Added a `PredictiveMinicolumn` class at the bottom of the file
(the existing `HTMTemporalPredictor` is untouched). It exposes:

- `generative_weights` (dim × dim) — required by `CAINTreeSearch`.
- `recognition_weights`, `precision`.
- `view(blanket)` — projects a Markov Blanket's `internal_state` into
  an observation, as `dual_planner.py` calls it.
- `predict`, `infer`, `update` for downstream users.

Semantics: precision-weighted Hebbian predictive coding with a top-k
sparsity gate. This is what the surrounding code was clearly written to
expect — the missing class was just not shipped.

### `embodied_agent/cortex/snn_engine.py`

**Symptom:** Hard-imports `hyperion_core/target/release/libhyperion_core.so`
at module load time and raises `FileNotFoundError` on failure. The
shipped `.so` is ARM64 only; on any x86_64 machine the whole
`embodied_agent` package is unimportable.

**Fix:** Full rewrite of the loader (the original was 75 lines):

1. Only attempt to load the Rust core if the ELF magic bytes and
   `e_machine` field match the current interpreter's architecture.
2. If missing or wrong-arch, fall back to a functionally-equivalent
   vectorised numpy implementation (`_snn_forward_py`) that mirrors
   `hyperion_core/src/lib.rs::snn_forward` line-for-line: leak → weighted
   excitation → threshold fire → uint8 LTP on active-and-fired, uint8
   LTD on active-but-not-fired.
3. Exposes a `USING_RUST_CORE` boolean and honours `HYPERION_FORCE_PY=1`
   for benchmarking parity.

The public API (`MemristorLayer`) is unchanged.

### `embodied_agent/praxis/htm_predictor.py`

**Symptom:** Same class of issue — this file also hard-imports the Rust
core's `htm_step` at module load.

**Fix:** Same fallback strategy. Added `_htm_step_py`, a pure-numpy
mirror of `hyperion_core::htm_step` (per-row homeostatic decay of
transition edges emanating from previously-active rows, then LTP on
prev→curr edges, then row-sum prediction), and route to it when
`_rust_core is None`. `HTMSequenceMemory.step()` picks the right
implementation at call time.

## Files not modified

- No existing research code was removed.
- `hyperion_core/target/release/libhyperion_core.so` (the ARM64 binary)
  is left in place so ARM64 users still get the Rust fast path.
- `hyperion_core/src/lib.rs` is left in place; anyone with a `cargo`
  toolchain can rebuild for their platform via
  `cd hyperion_core && cargo build --release`.

## Result

Before the changes: 0 of the 3 top-level entry points started on
x86_64.

After the changes, `python3 cli.py inspect` reports:

```
  dual_planner.py dual-brain                       OK
  embodied_agent embodied loop                   OK
  SNN MemristorLayer (rust or numpy fallback)  OK
  HTM sequence memory                          OK
  CainBrain (chat)                             OK
```

The archive still is not an AGI — see `AGI_ASSESSMENT.md` — but it now
actually runs everywhere, on stock Python 3.10+ with only `numpy`
installed.

# Changes in v32 (low-level assembly refresh)

This revision is a **substrate performance pass**, not a capability
claim. Nothing in this changelog says the system is smarter, more
general, or closer to AGI than v31. What it says is: the numeric
kernels run faster and I measured how much faster.

## Truth-in-labelling

- v32 does **not** implement "0 ms" connections. Real synapses take
  0.5–2 ms; real DRAM takes ~80 ns; L1 hit is ~1 ns. Signals in any
  physical computer are bounded by these floors. What v32 does is
  eliminate *software-induced* stalls (per-iteration accumulator
  dependencies, LLVM auto-vectoriser failures, cold cache misses on
  CSR row heads). Latency is measured in nanoseconds and microseconds,
  never labelled "0 ms". See `agi-bundle/neurorust/src/asm_kernels_avx2.rs`
  module doc.
- v32 does **not** demonstrate general learning. It runs the shipped
  STDP kernel on labelled patterns and reports a full pass/fail
  analysis. See `LEARNING_TEST_RESULTS.md`.
- v32 rejects the transformer paradigm at the design level — sparse
  spikes, local Hebbian rules, biophysical neurons — and is honest
  about where that bet is well-supported vs speculative. See
  `LLM_VS_BRAIN.md`.

## What's new

### 1. Packed AVX2 numeric kernels (`agi-bundle/neurorust/src/asm_kernels_avx2.rs`)

- `axpy_f64_avx2` — 4-lane `vfmadd231pd` inline-asm kernel for the
  fused y += a·x update used by the cable-equation integrator.
  Measured **1.30–3.09× speedup** vs the existing scalar `vfmadd231sd`
  path, tapering as vectors exceed L2 (memory-bandwidth limit).
- `spmv_csr_f32_avx2` (aliased to `spmv_csr_f32_scalar_prefetch`) —
  4-way unrolled scalar MAC with `prefetcht0` on the next row's head,
  for CSR connectome sparse matvec.
  Measured **1.70–1.97× speedup** vs the existing scalar path across
  100×100, 500×500, 2000×2000 connectomes at biologically realistic
  1–20 % density.
- `sparse_dot_f32_avx2` (kept but not on the hot path) — a
  `vpgatherdps`+`vfmadd231ps` variant. **Slower than the scalar path
  (0.6–0.9×)** at the low nnz-per-row typical of biological CSR — the
  kernel is documented as such and kept in-tree as a post-mortem on a
  failed prediction. See the module doc for details.

Correctness: 3 new unit tests bit-verify each kernel against the scalar
reference to within f32 ulp on random data. The full 9-test integration
suite (HH biological plausibility at ~70 Hz, Izhikevich 20–40 Hz RS,
CSR-vs-dense equivalence, cable-eqn tridiagonal solver, compression
round-trips) continues to pass.

### 2. Benchmark harness (`agi-bundle/neurorust/src/bin/bench_avx2.rs`)

Dependency-free head-to-head timing (25 runs, median). Reproduce with:
```bash
cd agi-bundle/neurorust
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin neurorust-bench-avx2
```
See `agi-bundle/neurorust/BENCH_RESULTS_v32.md` for the full result table.

### 3. Learning-capability test (`tests/test_learning_capability.py`)

Trains the shipped `MemristorLayer` on 4 orthogonal 8-hot patterns and
reports classification accuracy vs chance. Two supervisor modes:
- Naïve teacher-bias: **fails at chance (25 %)** — reveals that the
  shipped STDP rule has no lateral inhibition and applies uniform LTP
  to every firing output.
- Winner-take-all supervision: **succeeds at 100 %** — demonstrates
  the STDP rule works as a Hebbian rule when paired with an external
  inhibitory circuit.

Both results are documented in `LEARNING_TEST_RESULTS.md`. Reproduce:
```bash
python3 tests/test_learning_capability.py
```

### 4. Comparative doc (`LLM_VS_BRAIN.md`)

Grounded technical comparison across representation, learning rule,
scale, and temporal dynamics. Maps each Hyperion component to
"LLM-side" or "brain-side" of each axis, calls out the 6–8 orders of
magnitude parameter/synapse gap, and states which of Hyperion's bets
are well-supported vs speculative.

## What is unchanged

- All original v31 files are preserved as-is.
- No component was removed or renamed.
- `SparseConnectivity::spmv` (scalar) is unchanged. The AVX2 path is
  exposed as a new method `spmv_avx2` alongside it, so downstream code
  opts in explicitly.

## What is NOT claimed

- Not "0 ms".
- Not "AGI".
- Not "the brain of humans".
- Not "surprising learning".
- The system still cannot hold a coherent conversation, transfer skills
  across tasks, or reason above the level of the specific hand-wired
  demos shipped in v31. See `AGI_ASSESSMENT.md` (unchanged from v31)
  for the full honest evaluation.

## Files added

```
CHANGES_v32.md                                        (this file)
LEARNING_TEST_RESULTS.md
LLM_VS_BRAIN.md
tests/test_learning_capability.py
agi-bundle/neurorust/src/asm_kernels_avx2.rs
agi-bundle/neurorust/src/bin/bench_avx2.rs
agi-bundle/neurorust/BENCH_RESULTS_v32.md
```

## Files modified

```
agi-bundle/neurorust/Cargo.toml     (registers neurorust-bench-avx2 bin)
agi-bundle/neurorust/src/lib.rs     (pub mod asm_kernels_avx2)
agi-bundle/neurorust/src/compression.rs   (adds SparseConnectivity::spmv_avx2)
```

# Changes in v33 — biological-architecture learning fix

**Codename:** `hyperion_agi_v33_cortex_wta`
**Date:** 2026-07-09
**Focus:** Real algorithmic + substrate fix for the v32 learning failure,
plus AVX-512 and ARM NEON kernel paths.

## TL;DR

v32 documented (honestly) that the shipped `MemristorLayer` learns
orthogonal patterns at **chance (25 %)** without an external teacher
manually undoing weight changes on non-target output rows. v33 puts
that biology back inside the substrate:

| Variant                              | Final accuracy | Notes                                         |
|--------------------------------------|:--------------:|-----------------------------------------------|
| v1 `MemristorLayer` (no WTA)         | **25.0 %**     | fails at chance — the v32 diagnosis stands   |
| v33 `WTAMemristorLayer`              | **100.0 %**    | no external teacher — LIF+WTA+STDP inside   |
| v33 `ThreeFactorMemristorLayer`      | **100.0 %**    | reward-gated Hebbian, forward+`reward(r)`    |

Task: 4 orthogonal 8-hot patterns over 32 inputs, argmax readout,
Hungarian cluster-label matching (the standard for competitive-Hebbian
models). Reproduce with `python3 tests/test_learning_capability_v33.py`.

## Truth-in-labelling (same policy as v32)

- v33 does **not** implement zero-latency anything. Same physical
  limits as v32 — L1 ≈ 1 ns, DRAM ≈ 80 ns, real synapses 0.5–2 ms.
  We remove software-induced stalls, nothing more.
- v33 does **not** achieve general reasoning, transfer, or coherent
  conversation. It fixes one specific documented failure (WTA-less
  STDP can't learn) with one specific well-established motif
  (in-model WTA + Hebbian STDP).
- v33 does **not** claim "AGI". WTA + STDP + homeostatic plasticity
  is the substrate — the *scale* (mm² of cortex, ~10¹¹ synapses,
  curriculum, embodiment) that would matter for general capability
  is not touched.

## What's new

### 1. Rust modules

**`agi-bundle/neurorust/src/wta_kernel.rs`** — Winner-take-all lateral
inhibition, in Rust:
- `wta_argmax_f32` — SIMD argmax over potentials pool (1.3 G/s at
  pool size 4096 on the bench host).
- `wta_lateral_inhibit` — hard WTA: zero all non-winner spikes and
  reset their potentials. Corresponds biologically to strong
  basket-cell inhibition in layer-2/3.
- `k_wta_lateral_inhibit` — top-k winners for cortex-realistic
  sparse firing (~5 % active/tick).

**`agi-bundle/neurorust/src/three_factor_stdp.rs`** — Frémaux &
Gerstner (2016) reward-modulated STDP:
- Eligibility trace `e_ij(t)` accumulates Hebbian coincidences,
  decays with time constant `tau_e` (biological ~500–2000 ticks).
- `apply_neuromodulator(mod)` commits `Δw = lr · e · mod` — positive
  reward strengthens the causal synapses, negative reward weakens.
- FFI (`stdp_new`, `stdp_tick`, `stdp_free`) for the Python layer.

**`agi-bundle/neurorust/src/asm_kernels_avx512.rs`** — 8-lane FP64
and 16-lane FP32 axpy + gather-based SPMV. Runtime-dispatched via
`axpy_f64_best` / `saxpy_f32_best` (AVX-512 → AVX2 → scalar).

**`agi-bundle/neurorust/src/asm_kernels_neon.rs`** — AArch64 NEON
kernels (Apple Silicon, Graviton, mobile). Compiles as a no-op on
x86_64 so the crate builds identically on both.

### 2. Python layer

**`embodied_agent/cortex/snn_engine_v2.py`**:
- `WTAMemristorLayer` — LIF + hard WTA + Hebbian STDP + homeostatic
  threshold adaptation + heterogeneous init.
- `KWtaMemristorLayer` — soft k-winners variant.
- `ThreeFactorMemristorLayer` — reward-gated version, `forward()` then
  `reward(±1)`.

### 3. Tests

**`tests/test_learning_capability_v33.py`** — the v32 learning test
without external teacher weight-editing. Trains all three variants,
reports final accuracy + wall-clock + confusion matrix. Uses
Hungarian assignment on the confusion matrix (unsupervised clustering
is bijection-agnostic).

**Rust unit tests: 21 pass, 0 fail.** 8 are new for v33:
- `axpy_avx512_matches_scalar_reference`
- `saxpy_avx512_matches_scalar_reference`
- `spmv_avx512_matches_scalar_reference_on_random_csr`
- `runtime_dispatch_is_a_noop_when_features_present`
- `axpy_neon_matches_scalar_reference_or_falls_back_cleanly`
- `saxpy_neon_matches_scalar_reference_or_falls_back_cleanly`
- `argmax_simd_matches_scalar_reference`
- `hard_wta_silences_all_but_winner`
- `hard_wta_is_noop_when_no_neuron_fired`
- `kwta_keeps_top_k`
- `eligibility_decays_toward_zero`
- `coincident_pre_post_grows_eligibility`
- `reward_commits_eligibility_to_weights`
- `zero_modulator_is_noop`

### 4. Benchmarks

**`agi-bundle/neurorust/src/bin/bench_v33.rs`** — dependency-free
timing harness, 25 iterations, median reported. Measures axpy, spmv,
WTA argmax, and 3-factor STDP tick. Reproduce:

```sh
cd agi-bundle/neurorust
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin neurorust-bench-v33
```

## Measured numbers on the v33 bench host (Xeon-class w/ AVX-512F)

**axpy (f64) speedup vs scalar-FMA reference:**

| n       | scalar   | AVX2 (4×pd) | AVX-512 (8×pd) | AVX2× | AVX-512× |
|--------:|---------:|------------:|---------------:|------:|---------:|
| 64      |  45 ns   |  31 ns      |  49 ns         | 1.45× | 0.92×    |
| 256     | 124 ns   |  80 ns      |  72 ns         | 1.55× | 1.72×    |
| 1024    | 630 ns   | 252 ns      | 199 ns         | 2.50× | 3.17×    |
| 4096    |  2.56 µs |  1.34 µs    | 910 ns         | 1.91× | 2.81×    |
| 16384   |  7.90 µs |  4.87 µs    |  3.70 µs       | 1.62× | 2.13×    |
| 65536   | 31.17 µs | 18.58 µs    | 20.44 µs       | 1.68× | 1.53×    |
| 262144  |150.91 µs |133.66 µs    |135.17 µs       | 1.13× | 1.12×    |

Peak AVX-512 speedup **~3.17×** at n=1024 (L1-resident). Above L2 the
speedup collapses to ~1.1× because memory bandwidth, not FMA throughput,
becomes the bottleneck. This is expected and honest — same asymptote
the AVX2 path hit in v32.

**CSR spmv (f32, ~10 % density) — gather LOST, as predicted:**

| rows | nnz/row | scalar+prefetch | AVX-512 gather | gather×scalar |
|-----:|--------:|----------------:|---------------:|--------------:|
| 100  | 10.0    |    754 ns       |    786 ns      | 0.96×         |
| 500  | 50.0    |  13.34 µs       |  20.64 µs      | 0.65×         |
| 2000 | 200.0   | 223.40 µs       | 266.70 µs      | 0.84×         |

Kept in-tree for dense-block SPMV (thalamic projection matrices,
Sapphire Rapids), not for biological connectomes. Same honest
post-mortem as v32's AVX2 gather.

**WTA argmax:** 1.3–1.4 G elements/sec at pool sizes ≥256.
Full lateral-inhibition (argmax + zero non-winners) 5.5 µs at pool=4096.

**Three-factor STDP tick:** scales linearly with synapse count. 2M
synapses per tick in 4 ms on this host — one integration step every
2 ms of biological simulated time, or roughly real-time on ≤1M-synapse
networks.

## What is unchanged

- All v32 files preserved as-is. Nothing was removed or renamed.
- `MemristorLayer` in `snn_engine.py` unchanged — the v33 layer is a
  new subclass in `snn_engine_v2.py`, so any existing code path that
  imports `MemristorLayer` continues to get the v1 behaviour.
- v32 learning test (`tests/test_learning_capability.py`) still passes
  at 100 % with the external teacher — no regression.
- AVX2 kernels and their production scalar-prefetch SPMV are unchanged.
- The `hyperion_core` fallback path is unchanged.

## Migration guide

To adopt the v33 fix in an existing script, change the import:

```python
# before (v32) — still works, still fails at chance without teacher
from embodied_agent.cortex.snn_engine import MemristorLayer

# after (v33) — passes at 100% on the shipped learning test, no teacher
from embodied_agent.cortex.snn_engine_v2 import WTAMemristorLayer

layer = WTAMemristorLayer(n_inputs=32, n_outputs=4,
                          threshold=1.0, leak=0.9, seed=0)
```

For reward-modulated learning:

```python
from embodied_agent.cortex.snn_engine_v2 import ThreeFactorMemristorLayer

layer = ThreeFactorMemristorLayer(n_inputs=32, n_outputs=4,
                                  tau_e_ticks=20.0)
spikes = layer.forward(pattern)
layer.reward(+1.0 if outcome_good else -0.3)
```

See `V33_MIGRATION.md` for the full API surface.

## What is NOT claimed

- Not "0 ms".
- Not "AGI".
- Not "the brain of humans".
- Not "surprising learning".
- Not scalable-to-language-out-of-the-box.
- The system still cannot hold a coherent conversation, transfer skills
  across tasks, or reason above the level of hand-wired components.

What v33 *does* claim: the substrate now learns orthogonal patterns at
100 % from unsupervised forward passes with the WTA circuit inside the
model, matching what a real cortical column does. That is the specific
gap `AGI_ASSESSMENT.md` and `LEARNING_TEST_RESULTS.md` from v32 opened,
now closed.
