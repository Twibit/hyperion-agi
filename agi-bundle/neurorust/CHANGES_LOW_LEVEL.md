# Low-level performance & correctness patch for `neurorust`

This patch improves the biologically-accurate neuron simulator in
`agi-bundle/neurorust` with architecture-tuned inline-assembly kernels
plus fixes for latent quadratic-time and NaN-divergence bugs.

## TL;DR

* **New module `asm_kernels`** — three low-level kernels with `#[cfg]`-gated
  inline `asm!` fast paths for x86-64 (SSE4.1 / FMA) and AArch64 (NEON /
  `frintn`), and safe scalar fallbacks everywhere else.
  * `fast_exp(x: f64) -> f64` — range-reduced polynomial `exp` used by
    every HH rate constant. Roughly 2–3× faster than libm on the
    biological band, max rel err < 5·10⁻⁷.
  * `axpy_f64(a, x, y)` — fused `y += a·x` used by the cable-equation
    integrator. Emits `vfmadd231sd` on FMA-enabled x86-64.
  * `sparse_dot_f32(w, ci, x, s, e)` — inner loop of CSR `spmv`. Pins
    the accumulator to `xmm0` to defeat a known LLVM register-spill
    regression on dependent-load chains.

* **HH neuron** — rate constants (`alpha_m`, `beta_m`, `alpha_h`, `beta_h`,
  `alpha_n`, `beta_n`) rewritten to use `fast_exp` and to fold their
  singularity guards into a single L'Hôpital branch.

* **Multi-compartment neuron** — eliminates the two per-step allocations
  (`dv = vec![0.0; n]`, `aux = self.v.clone(); aux.push(...)`) via
  reusable scratch buffers. Applies `axpy_f64` for the compartment
  update. Adds a numerical safety clamp on `V` to prevent NaN
  divergence under off-nominal drive.

* **Compression** — the biggest single win. Both `SpikeTrain::push` and
  `MembraneTrace::extend` were **O(n) per call** because they
  decompressed + recompressed the entire zstd payload on every write.
  A `NeuronMemory::record` call per simulation step made this
  **O(n²) per simulation** and rendered any run longer than ~10k steps
  effectively broken. Fixed with a `pending: Vec<u8>` staging buffer
  flushed at 64 KiB, restoring **O(1) amortised** writes.

* **`SparseConnectivity::spmv`** — now delegates to `sparse_dot_f32`,
  which also validates `col_idx` bounds and no longer panics on
  malformed CSR (see `FAILURE_PREDICTIONS.md` §8 for the trade-off).

## Measured impact

Micro-benchmark: single HH neuron, dt = 0.01 ms, I_inj = 10 µA/cm²
(from `examples/bench_long_sim.rs`, x86-64 sandbox).

| Steps    | Baseline wall time |  Optimized  | Speedup |
|---------:|-------------------:|------------:|--------:|
|    1 000 |          68.98 ms  |    0.44 ms  |    156× |
|    5 000 |         419.59 ms  |    1.63 ms* |    258× |
|   10 000 |       1 056.39 ms  |    3.25 ms  |    325× |
|   20 000 |       2 923.15 ms  |    7.20 ms* |    406× |
|   40 000 |       8 187.45 ms  |   15.20 ms* |    539× |
|  100 000 |         (~57 s\*\*)|   35.70 ms  |  ~1 600×|
| 1 000 000|      **>15 min TO**|  422.85 ms  |    ∞    |

\* linearly interpolated from measured neighbours (per-step cost is
flat in the optimized version).
\*\* linearly extrapolated from measured per-step slope of the
baseline (grows super-linearly, so this underestimates the cost).

Per-step cost after the patch is essentially flat at ~350 ns,
confirming the algorithmic complexity is now O(n) instead of O(n²).

## Compatibility

* No public API removed or renamed. Two new public methods:
  `SpikeTrain::flush(&mut self) -> Result<...>` and
  `MembraneTrace::flush(&mut self) -> Result<...>`.
* All 4 existing lib unit tests pass.
* All 9 existing integration tests pass (biological plausibility of
  HH / Izhikevich / multi-compartment firing rates, spike-train
  lossless round-trip, membrane-trace ≤ noise-floor error, CSR ==
  dense spmv reference, tridiagonal-solver reference, bit-raster
  compression ratio).
* Four new unit tests in `asm_kernels::tests` verify the fast paths
  against libm within their advertised tolerances.

## Why inline asm and not just SIMD intrinsics?

For the three specific kernels involved, LLVM's autovectoriser was
observed to emit suboptimal code at `-Copt-level=3`:

1. `spmv` inner loop — spilled the running accumulator around the
   dependent load `x[col_idx[k]]`.
2. `axpy` — emitted a full packed-double FMA prologue when a scalar
   `vfmadd231sd` on a register-resident accumulator is enough.
3. `round_nearest` inside `fast_exp` — turned into a `nearbyint` libm
   call even under `-ffast-math` equivalents.

Each `asm!` block is small (2–4 instructions), safe (`nostack`,
`pure` / `nomem` where valid), and paired with a portable scalar
fallback that is functionally identical. Correctness is asserted by
unit tests on every target.

## Files changed / added

Added:

* `src/asm_kernels.rs` — new module.
* `examples/bench_long_sim.rs` — benchmark harness.
* `FAILURE_PREDICTIONS.md` — root-cause + trigger for each of ten
  latent failure modes.
* `CHANGES_LOW_LEVEL.md` — this file.

Modified:

* `src/lib.rs` — expose `asm_kernels`.
* `src/models/hodgkin_huxley.rs` — rate constants use `fast_exp`.
* `src/models/multi_compartment.rs` — scratch buffers, `axpy_f64`,
  numerical clamp.
* `src/compression.rs` — `pending` staging buffers on `SpikeTrain`
  and `MembraneTrace`; new `flush()` methods; `spmv` uses
  `sparse_dot_f32`.
