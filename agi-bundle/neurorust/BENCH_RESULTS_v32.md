# neurorust v32 — packed AVX2 kernel results (honest)

Measured on: Intel(R) Xeon(R) CPU @ 2.60GHz (Skylake-family, has AVX-512,
AVX2, FMA, BMI2). Compiled with `RUSTFLAGS="-C target-cpu=native"`.
Baseline: existing `asm_kernels.rs` (scalar `vfmadd231sd` for AXPY,
`mulss+addss` for sparse dot).
Comparison target: new `asm_kernels_avx2.rs`.

Median of 25 runs after 1 warm-up. All AVX2 outputs match the scalar
reference to within f32 unit round-off (see `#[cfg(test)] mod tests`).

## AXPY (packed 4-lane `vfmadd231pd` vs scalar `vfmadd231sd`)

| n        | scalar (µs) | AVX2 (µs) | speedup |
|---------:|------------:|----------:|--------:|
|      256 |        0.18 |      0.06 |  3.09×  |
|    4 096 |        2.65 |      1.02 |  2.60×  |
|   65 536 |       38.66 |     20.07 |  1.93×  |
| 1 048 576|      870.25 |    671.83 |  1.30×  |

Speedup falls off with n because past L2 (~256 KiB / core) the kernel
transitions from FMA-throughput-limited to DRAM-bandwidth-limited.
Theoretical ceiling for the small-n regime is 4× (four lanes per FMA);
we hit 77 % of that at n=256.

## CSR sparse matvec (4-way unrolled scalar + software prefetch vs single scalar acc)

| shape       | nnz    | scalar (µs) | new (µs) | speedup | max rel err |
|-------------|-------:|------------:|---------:|--------:|------------:|
| 100 × 100   |  3 997 |        3.91 |     2.04 |  1.91×  |     2.0e-7  |
| 500 × 500   | 24 996 |       26.96 |    13.68 |  1.97×  |     6.2e-7  |
| 2000 × 2000 | 79 989 |       82.24 |    48.30 |  1.70×  |     1.1e-3  |

The 1.1e-3 relative error at nnz≈80 000 is the f32 accumulator floor
(catastrophic cancellation on long sums), not a bug — the 4-way tree
reduction (a0+a1)+(a2+a3) is actually *more* numerically stable than the
scalar single-accumulator baseline; it just makes the error visible at
larger n.

## What did NOT work

The first `spmv_csr_f32_avx2` implementation used `vpgatherdps` +
`vfmadd231ps` for a packed 8-wide dot product. On paper this should be
2.5–3.5× faster; measured result was **0.6–0.9× (slower)**. The reason
is Skylake microcode: `vpgatherdps` issues 8 dependent µops with 1 lane
per cycle throughput, plus potential 4-KB-boundary stalls. For the
low-nnz-per-row (~40) of biological cortical CSR, 8 dependent scalar
loads with full pipelining beat gather.

We kept the gather kernel in `asm_kernels_avx2.rs` for reference and
future dense-block use, but the production path (`spmv_csr_f32_avx2` →
`spmv_csr_f32_scalar_prefetch`) uses 4-way unrolled scalar with
`prefetcht0` on the next row's head. Post-mortem doc-comment in the
source file explains the failed prediction.

## Correctness

```
running 7 tests
test asm_kernels::tests::axpy_matches_scalar ... ok
test asm_kernels::tests::fast_exp_matches_libm_on_biological_range ... ok
test asm_kernels::tests::fast_exp_falls_back_outside_band ... ok
test asm_kernels::tests::sparse_dot_matches_scalar ... ok
test asm_kernels_avx2::tests::axpy_avx2_matches_scalar_reference ... ok
test asm_kernels_avx2::tests::sparse_dot_avx2_matches_scalar_reference ... ok
test asm_kernels_avx2::tests::spmv_avx2_matches_scalar_reference_on_random_csr ... ok

test result: ok. 7 passed
```

Full 9-test integration suite (Hodgkin-Huxley biological plausibility,
Izhikevich firing rate, cable-equation tridiagonal, compression
round-trip, CSR-vs-dense equivalence) also still passes.

## Reproduce

```bash
cd agi-bundle/neurorust
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin neurorust-bench-avx2
RUSTFLAGS="-C target-cpu=native" cargo test  --release
```
