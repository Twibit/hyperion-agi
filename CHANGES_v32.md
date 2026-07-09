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
