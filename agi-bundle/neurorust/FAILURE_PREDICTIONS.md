# Predicted future code failures in `neurorust`

This document catalogues failure modes that were latent in
`hyperion_agi_v30_final.tar.gz` and either (a) have now been fixed by this
patch, or (b) remain as risks the maintainer should track. Each entry
follows the same schema:

* **Symptom** — what the user or CI will observe when it fires.
* **Root cause** — the underlying code pathology.
* **Trigger** — the conditions under which it becomes visible.
* **Status** — Fixed / Mitigated / Documented risk.

---

## §1. Quadratic-time simulation from per-step recompression **[FIXED]**

* **Symptom.** Wall time explodes super-linearly with simulation length.
  Empirical measurement on this machine (see `examples/bench_long_sim.rs`):

  | Steps  | Baseline wall time | Per-step cost |
  |-------:|-------------------:|--------------:|
  |  1 000 |          0.069 s   |  68 984 ns    |
  |  5 000 |          0.420 s   |  83 918 ns    |
  | 10 000 |          1.056 s   | 105 639 ns    |
  | 20 000 |          2.923 s   | 146 157 ns    |
  | 40 000 |          8.187 s   | 204 686 ns    |
  |  1e6   |    **timed out at 900 s** | — |

  Per-step cost is growing linearly ⇒ overall cost is O(n²). A 1 s
  biological simulation at dt = 0.01 ms (10⁵ steps) already needs
  ~35 s per neuron; a 10 s simulation would need ~1 h; scaling to
  networks is impossible.

* **Root cause.** `NeuronMemory::record` was called once per step and
  routed to `MembraneTrace::extend(std::iter::once(v))`, which
  decompressed the ENTIRE zstd payload, appended two bytes, and
  recompressed. Same pattern in `SpikeTrain::push`. Because `zstd`
  level-3 is roughly O(n log n) in payload size, and the payload grows
  monotonically, the total work is `Σ_{k=1..n} O(k log k) = O(n² log n)`.

* **Trigger.** Any simulation longer than a few thousand steps.

* **Fix.** Added an unflushed `pending: Vec<u8>` staging buffer to both
  `SpikeTrain` and `MembraneTrace`. Writes go straight to `pending`;
  the zstd payload is only rebuilt when `pending` crosses
  `PENDING_FLUSH_BYTES = 64 KiB`, or when a reader (`to_samples`,
  `to_times_ms`) is called, or when the user calls the new public
  `flush()` method. Amortised cost is now O(1) per write and O(n log n)
  total. Empirical: **156–539× speedup** across the measured range,
  and — crucially — flat per-step cost regardless of length.

* **Post-fix measurement.**

  | Steps    | Optimized wall time | Per-step cost |
  |---------:|--------------------:|--------------:|
  |    1 000 |           0.44 ms   |    442 ns     |
  |   10 000 |           3.25 ms   |    325 ns     |
  |  100 000 |          35.7  ms   |    357 ns     |
  | 1 000 000 |         423   ms   |    423 ns     |

  Per-step cost is now **flat**.

---

## §2. NaN divergence in `MultiCompartmentNeuron` under off-nominal drive **[FIXED]**

* **Symptom.** After some number of steps, `v_mem()` becomes `NaN`; all
  downstream statistics (firing rate, compression ratio) then read as
  `NaN`, breaking JSON serialisation and any Python plotter that
  ingests them.

* **Root cause.** The somatic dynamics include the Izhikevich quadratic
  `0.04 · v²`. The original spike-reset check (`self.v[0] >= 30.0`)
  runs only *after* the forward-Euler update. If a caller injects a
  large current in the middle of a run — say, a network with feedback
  gain > 1 that briefly pushes `v` past +100 mV — then `0.04·v²` grows
  faster than the linear damping term and integer overflow in `f64`
  eventually collapses to `±inf → NaN`. Once `NaN` enters one
  compartment, axial coupling propagates it to neighbours next step.

* **Trigger.** Any injected-current schedule capable of pushing any
  compartment above ~+80 mV before spike reset — e.g. mis-scaled
  synaptic weights, or a network that accidentally learns positive
  feedback.

* **Fix.** After every integration step, each compartment voltage is
  clamped to `[V_CLAMP_LO, V_CLAMP_HI] = [-200 mV, +100 mV]` (the
  physiological rail plus a wide safety margin). The clamp is a no-op
  under physiological drive; under pathological drive it prevents NaN
  propagation and, in debug builds, fires a `debug_assert` so the
  user's tests catch the misconfiguration.

---

## §3. Per-step heap churn in `MultiCompartmentNeuron::step` **[FIXED]**

* **Symptom.** Perf profile shows 10–20 % of samples in
  `alloc::vec::Vec::with_capacity` and `__rust_alloc`. Under Miri or
  ASan, runs 100× slower than release.

* **Root cause.** `step()` allocated a fresh `dv = vec![0.0; n]` and
  produced a fresh `aux = self.v.clone(); aux.push(u)` on every call.
  With `dt = 0.01 ms` (100 000 steps per second of biology) that's
  200 000 allocations/s per neuron.

* **Trigger.** Any long simulation; the effect is most visible in the
  `MultiCompartment` model but the pattern also afflicts network
  code that stores the returned `aux` vector.

* **Fix.** Added `scratch_dv: Vec<f64>` and `scratch_aux: Vec<f64>`
  fields to `MultiCompartmentNeuron`. The `step` body reuses them
  in-place. The returned `NeuronState.aux` still owns its data (one
  `Vec::clone` per step is unavoidable at the API boundary), but the
  internal churn is gone.

---

## §4. LLVM regression on the CSR `spmv` inner loop **[MITIGATED]**

* **Symptom.** Network-level throughput scales sub-linearly with core
  count on x86-64 hosts. Perf annotates the innermost loop with
  extra load/store pairs around the accumulator.

* **Root cause.** The `spmv` inner loop mixes a strided load
  (`x[col_idx[k]]`) with a running accumulator. On some rustc versions
  the accumulator gets spilled to the stack across the dependent load,
  producing a store+load per iteration.

* **Trigger.** Present on x86-64 with rustc ≥ 1.72 at `-Copt-level=3`.

* **Mitigation.** New `asm_kernels::sparse_dot_f32` pins the
  accumulator to `xmm0` with two-instruction `mulss / addss` inline
  asm. On non-x86-64 hosts the code falls back to the scalar Rust
  form, which LLVM optimises well because there's no dependent-load
  chain to confuse it.

---

## §5. `.exp()` call rate in HH rate constants **[MITIGATED]**

* **Symptom.** Perf profile shows ~60 % of HH-neuron time in
  `libm_exp` / `__ieee754_exp_data`. This is unavoidable at the
  algorithmic level (RK4 needs 4 rate-constant evaluations per step,
  each of which uses up to 3 `exp` calls) but the constant factor is
  addressable.

* **Root cause.** libm's `exp` is a fully accurate, correctly-rounded
  implementation. HH does not need `< 1 ULP` accuracy — the RK4
  truncation error at dt = 0.01 ms dominates any error under 10⁻⁶.

* **Mitigation.** New `asm_kernels::fast_exp` does range-reduced
  Cody–Waite decomposition + degree-6 Horner polynomial + IEEE-754
  exponent-field construction. Verified in `mod tests` to match
  `f64::exp` to < 5 × 10⁻⁷ relative error on `[-40, 40]` — the entire
  biological voltage band. Out-of-band arguments fall back to libm so
  callers never see catastrophic error on off-nominal input.

  Additionally, on x86-64 with SSE4.1 the range-reduction rounding
  uses a single `roundsd` instruction via inline asm, avoiding a
  libm call for `nearbyint`. On AArch64 with NEON we use `frintn`.

---

## §6. `alpha_m` / `alpha_n` L'Hôpital branches were correct but wasteful **[MITIGATED]**

* **Symptom.** No observable failure, but the guard-and-fallback
  pattern executed `.exp()` twice on every non-degenerate call.

* **Fix.** Rewrote to a single `fast_exp` call inside the divisor
  and a single L'Hôpital-limit branch for the singularity. Same
  correctness, one fewer FLOP.

---

## §7. `MultiCompartmentNeuron` "distal tip injection" is a semantic
     surprise **[DOCUMENTED RISK — not changed]**

* **Symptom.** Users who inject current into the soma find that 5 %
  of that current also appears at the distal dendrite tip.

* **Root cause.** The reference implementation includes
  `dv[n-1] += 0.05 * i_inj` to *demonstrate* dendritic propagation.
  This is opinionated behaviour that most users would not expect from
  a function called `step(dt, i_inj)`.

* **Recommendation (not applied to preserve API and existing tests).**
  Split into two entry points: `step_soma(dt, i_inj)` (default) and
  `step_split(dt, i_soma, i_dendrite_tip)`. Deprecate the current
  behaviour over one release.

---

## §8. `SparseConnectivity::spmv` silently returns wrong result on
     malformed CSR **[MITIGATED]**

* **Symptom.** If `col_idx[k] >= x.len()` (corrupted or externally
  provided CSR), the original code index-panicked. `sparse_dot_f32`
  now silently skips such entries. This is a **trade-off**: safer
  against panics in production, but hides bugs. The invariant is
  documented at the call site.

* **Recommendation.** Add a `SparseConnectivity::validate() ->
  Result<(), Error>` method that callers invoke once after
  construction from external data.

---

## §9. `bincode`'s `Box<ErrorKind>` in `CompressionError` is a v1-only
     spelling **[DOCUMENTED RISK — not changed]**

* **Symptom.** If the crate is upgraded to `bincode = "2"` the
  `Bincode(#[from] Box<bincode::ErrorKind>)` variant stops compiling
  because bincode 2 has a completely different error type.

* **Recommendation.** Pin `bincode = "1.3"` explicitly (already done)
  and add a `# Version` comment; when bincode 2 is adopted, migrate
  in a single commit.

---

## §10. `neurorust` `pending` fields are serialised on `Serialize`
      derives **[DOCUMENTED]**

* **Symptom.** A user who serialises a `MembraneTrace` with unflushed
  pending bytes will get a JSON that de-serialises correctly (because
  of `#[serde(default)]`) but that a non-Rust reader (e.g. a Python
  tool that only reads `payload`) will see as inconsistent — the
  Python side will report N samples according to `n_samples`, but only
  see `payload.len() / 2` samples in `payload`.

* **Recommendation.** Either (a) call `.flush()?` before serialising
  from user code, or (b) implement a custom `Serialize` impl that
  flushes first. Not applied here to preserve `#[derive(Serialize)]`
  ergonomics and avoid an implicit `&mut self` in serialise paths.
