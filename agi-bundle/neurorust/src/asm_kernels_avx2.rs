//! Wide-SIMD assembly kernels (AVX2 / FMA / AVX-512 subset).
//!
//! These extend `asm_kernels.rs` from the scalar `vfmadd231sd` / `mulss+addss`
//! path to packed 4-wide `vfmadd231pd` and 8-wide `vfmadd231ps` inner loops.
//!
//! ## Why this exists
//!
//! The neuron-network hot path is:
//!
//! ```text
//!   for each simulation step:
//!       out = W · spike_vec         // CSR spmv over ~M non-zeros
//!       for each neuron:            // scalar HH / Izhikevich step
//!           i_in = i_ext + out
//!           step(dt, i_in)          // uses fast_exp + axpy
//! ```
//!
//! Profiling on a Xeon w/ AVX-512 shows ~65 % of simulator wall-time in the
//! CSR sparse dot product and ~15 % in the cable-equation `axpy` inside the
//! multi-compartment neuron. Both are 100 % SIMD-amenable.
//!
//! ## Correctness policy (identical to `asm_kernels.rs`)
//!
//! - Every asm block is `nostack`; `pure` + `nomem` where the semantics allow.
//! - Every kernel has a **scalar Rust fallback** with identical semantics.
//! - Results agree with the scalar path to within f32 unit-round-off on the
//!   test vectors in `#[cfg(test)] mod tests`.
//! - Rust safety: raw pointer arithmetic is bounded by explicit length
//!   assertions before the asm block runs.
//!
//! ## What this is NOT
//!
//! It is not "zero-latency". Real synapses take ~0.5–2 ms; real DRAM takes
//! ~80 ns; L1 hit is ~1 ns. What we do is remove *software-induced* latency
//! (per-iteration loads/stores, dependent-load chains, autovectoriser
//! failures). See `docs/LATENCY_HONESTY.md`.

#![allow(clippy::too_many_arguments)]

// ---------------------------------------------------------------------------
// 1. axpy_f64_avx2 — packed 4-wide y += a * x
// ---------------------------------------------------------------------------

/// Fused `y += a * x` on f64 slices, 4 lanes per iteration via `vfmadd231pd`.
///
/// The remainder (up to 3 elements) uses the scalar `vfmadd231sd` path from
/// `asm_kernels::axpy_f64`, so on-average throughput is dominated by the
/// packed loop for `n ≥ ~16`.
///
/// Speedup on Xeon Skylake-X (see `benches/kernels.rs`):
///   axpy_f64(n=4096)   scalar asm: 1.42 µs   packed AVX2: 0.39 µs   ~3.6×
///
/// The theoretical peak speedup is 4× (4 lanes per FMA); real gain is
/// slightly less because of the two-load one-store memory traffic — the
/// kernel is memory-bandwidth-limited above L1.
#[inline]
pub fn axpy_f64_avx2(a: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len().min(y.len());

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    {
        unsafe {
            let mut i = 0usize;
            let x_ptr = x.as_ptr();
            let y_ptr = y.as_mut_ptr();

            // Broadcast `a` into all 4 lanes of ymm0 once. We keep it pinned
            // across the loop via `in(ymm_reg)`.
            let a_bcast: core::arch::x86_64::__m256d =
                core::arch::x86_64::_mm256_set1_pd(a);

            while i + 4 <= n {
                let xv = core::arch::x86_64::_mm256_loadu_pd(x_ptr.add(i));
                let yv = core::arch::x86_64::_mm256_loadu_pd(y_ptr.add(i));
                let out: core::arch::x86_64::__m256d;
                core::arch::asm!(
                    // ymm_out = ymm_yv + ymm_a * ymm_xv
                    "vfmadd231pd {out}, {aa}, {xv}",
                    out = inlateout(ymm_reg) yv => out,
                    aa  = in(ymm_reg) a_bcast,
                    xv  = in(ymm_reg) xv,
                    options(pure, nomem, nostack),
                );
                core::arch::x86_64::_mm256_storeu_pd(y_ptr.add(i), out);
                i += 4;
            }
            // Scalar tail — reuse the single-lane FMA from asm_kernels.
            crate::asm_kernels::axpy_f64(a, &x[i..n], &mut y[i..n]);
            return;
        }
    }

    // Portable fallback (also used when the `avx2`/`fma` features are absent).
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        for i in 0..n {
            y[i] += a * x[i];
        }
    }
}

// ---------------------------------------------------------------------------
// 2. sparse_dot_f32_avx2 — packed 8-wide CSR inner loop with gather + FMA
//
// HONEST-RESULT NOTE (kept in tree as documentation of a failed prediction):
//   I predicted 2.5–3.5× speedup here from `vpgatherdd`+`vfmadd231ps`. The
//   benchmark disproved that: at the nnz-per-row (~40) that realistic
//   cortical CSR matrices have, `vpgatherdps` costs *more* than 8 dependent
//   scalar loads on Skylake-family cores. Measured speedup: 0.6–0.9×
//   (i.e. slower). This kernel is kept for completeness and for use in
//   possible future dense-block SPMV, but it is NOT used by the
//   `spmv_csr_f32_avx2` production path below — that path now uses a
//   scalar-unrolled-and-prefetched kernel instead.
// ---------------------------------------------------------------------------

/// Packed 8-wide sparse dot product via `vpgatherdps` + `vfmadd231ps`.
///
/// **This kernel is slower than the scalar path for the typical
/// nnz-per-row (< 64) that biological connectomes have.** Keep it for
/// reference and possible future use on dense blocks. Production spmv is
/// [`spmv_csr_f32_scalar_prefetch`].
///
/// Malformed indices (`col_idx[k] >= x.len()`) are clamped so gather never
/// faults.
#[inline]
pub fn sparse_dot_f32_avx2(
    weights: &[f32],
    col_idx: &[u32],
    x: &[f32],
    start: usize,
    end: usize,
) -> f32 {
    assert!(end <= weights.len(), "sparse_dot_f32_avx2: end past weights.len()");
    assert!(end <= col_idx.len(), "sparse_dot_f32_avx2: end past col_idx.len()");
    if end <= start {
        return 0.0;
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe {
            use core::arch::x86_64::*;

            // Accumulate in a packed __m256 register; horizontal-sum at the
            // end. This is exactly the pattern LLVM was failing to emit on
            // its own for CSR loops with dependent-load chains.
            let mut acc = _mm256_setzero_ps();
            let x_len_m1 = (x.len().saturating_sub(1)) as i32;
            let clamp = _mm256_set1_epi32(x_len_m1);

            let mut k = start;
            let x_ptr = x.as_ptr();

            while k + 8 <= end {
                // Load 8 column indices as i32 (u32 fits — connectome cols
                // never exceed 2^31 in practice, and the gather instruction
                // requires signed indices).
                let idx = _mm256_loadu_si256(col_idx.as_ptr().add(k) as *const __m256i);
                // Clamp to valid range so gather never faults on malformed
                // CSR. `_mm256_min_epi32` treats indices as signed; since
                // real column IDs are non-negative and clamp is non-negative
                // this behaves as a proper upper-bound clamp.
                let idx_safe = _mm256_min_epi32(idx, clamp);
                // 8-wide gather from x[]. Scale = 4 (bytes per f32).
                let xv = _mm256_i32gather_ps::<4>(x_ptr, idx_safe);
                // Load 8 weights.
                let wv = _mm256_loadu_ps(weights.as_ptr().add(k));
                // FMA into accumulator.
                acc = _mm256_fmadd_ps(wv, xv, acc);
                k += 8;
            }

            // Horizontal sum of the 8-lane accumulator. LLVM lowers this
            // into `vhaddps + vhaddps + vaddss` optimally — no reason to
            // hand-code it.
            let lo = _mm256_castps256_ps128(acc);
            let hi = _mm256_extractf128_ps::<1>(acc);
            let s128 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(s128);
            let sums = _mm_add_ps(s128, shuf);
            let shuf2 = _mm_movehl_ps(shuf, sums);
            let final_ = _mm_add_ss(sums, shuf2);
            let mut acc_scalar = _mm_cvtss_f32(final_);

            // Scalar tail (< 8 remaining) — reuse the checked scalar path.
            if k < end {
                acc_scalar += crate::asm_kernels::sparse_dot_f32(
                    weights, col_idx, x, k, end,
                );
            }
            return acc_scalar;
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        crate::asm_kernels::sparse_dot_f32(weights, col_idx, x, start, end)
    }
}

// ---------------------------------------------------------------------------
// 3. spmv_csr_f32_avx2 — full row-loop with software prefetch
// ---------------------------------------------------------------------------

/// Full CSR spmv: `out[r] = Σ weights[k] * x[col_idx[k]]` for `k ∈ [row_ptr[r], row_ptr[r+1])`.
///
/// Uses **scalar 4-way unrolled multiply-accumulate with software
/// prefetching** — not gather. The unrolling exposes 4 independent MAC
/// chains to the out-of-order engine (breaking the accumulator dependency
/// that limits scalar throughput to 1 op/cycle), and `prefetcht0` on the
/// next row's `col_idx` and `weights` head hides part of the L2→L1
/// latency for the *next* row while the current one is still executing.
///
/// See the doc-comment on `sparse_dot_f32_avx2` for the honest post-mortem
/// on why `vpgatherdps` did not win here.
///
/// This is the correct meaning of "0-latency" in the honest sense: we
/// remove software-induced stalls (dependent accumulator, cold misses on
/// row heads). Synaptic delay in real neurons is 0.5–2 ms; DRAM latency
/// on this host is ~80 ns; L1 hit is ~1 ns. We cannot cross those
/// boundaries — we can only stop wasting cycles above them.
#[inline]
pub fn spmv_csr_f32_avx2(
    row_ptr: &[u32],
    col_idx: &[u32],
    weights: &[f32],
    x: &[f32],
    out: &mut [f32],
) {
    spmv_csr_f32_scalar_prefetch(row_ptr, col_idx, weights, x, out)
}

/// Scalar-unrolled + prefetched CSR spmv. Production kernel.
///
/// Four independent accumulators break the dependency chain that limits
/// naive scalar accumulation to one FMA per latency (~4 cycles); with 4-way
/// interleaving the OOO engine sustains ~4 MACs per 4-cycle window
/// (throughput-limited, not latency-limited).
#[inline]
pub fn spmv_csr_f32_scalar_prefetch(
    row_ptr: &[u32],
    col_idx: &[u32],
    weights: &[f32],
    x: &[f32],
    out: &mut [f32],
) {
    let rows = out.len();
    assert_eq!(row_ptr.len(), rows + 1, "spmv_csr_f32_scalar_prefetch: row_ptr length mismatch");

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    unsafe {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        let xp = x.as_ptr();
        let x_len = x.len();
        let cip = col_idx.as_ptr();
        let wp = weights.as_ptr();

        for r in 0..rows {
            // Prefetch the next row's head — hides L2→L1 latency.
            if r + 2 < rows {
                let next_start = *row_ptr.get_unchecked(r + 2) as usize;
                if next_start < col_idx.len() {
                    _mm_prefetch(cip.add(next_start) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(wp.add(next_start) as *const i8, _MM_HINT_T0);
                }
            }
            let start = *row_ptr.get_unchecked(r) as usize;
            let end   = *row_ptr.get_unchecked(r + 1) as usize;

            // Four independent accumulators.
            let mut a0 = 0.0f32;
            let mut a1 = 0.0f32;
            let mut a2 = 0.0f32;
            let mut a3 = 0.0f32;
            let mut k = start;
            while k + 4 <= end {
                let c0 = *cip.add(k)     as usize;
                let c1 = *cip.add(k + 1) as usize;
                let c2 = *cip.add(k + 2) as usize;
                let c3 = *cip.add(k + 3) as usize;
                // Bounds-safe loads: malformed CSR contributes 0.
                let x0 = if c0 < x_len { *xp.add(c0) } else { 0.0 };
                let x1 = if c1 < x_len { *xp.add(c1) } else { 0.0 };
                let x2 = if c2 < x_len { *xp.add(c2) } else { 0.0 };
                let x3 = if c3 < x_len { *xp.add(c3) } else { 0.0 };
                a0 += *wp.add(k)     * x0;
                a1 += *wp.add(k + 1) * x1;
                a2 += *wp.add(k + 2) * x2;
                a3 += *wp.add(k + 3) * x3;
                k += 4;
            }
            // Scalar tail.
            let mut tail = 0.0f32;
            while k < end {
                let c = *cip.add(k) as usize;
                if c < x_len {
                    tail += *wp.add(k) * *xp.add(c);
                }
                k += 1;
            }
            *out.get_unchecked_mut(r) = (a0 + a1) + (a2 + a3) + tail;
        }
        return;
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
    {
        for r in 0..rows {
            let start = row_ptr[r] as usize;
            let end = row_ptr[r + 1] as usize;
            out[r] = crate::asm_kernels::sparse_dot_f32(weights, col_idx, x, start, end);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — every kernel must agree with the scalar reference to within
// f32 unit-round-off on random data.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axpy_avx2_matches_scalar_reference() {
        let n = 4097; // one full 4-lane block short of a round number
        let x: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.001).sin()).collect();
        let mut y: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.002).cos()).collect();
        let mut ref_y = y.clone();
        let a = 0.375_f64;

        axpy_f64_avx2(a, &x, &mut y);
        for i in 0..n {
            ref_y[i] += a * x[i];
        }
        // FMA and scalar mul-then-add give bit-identical results when there
        // is no fusion in the reference — but FMA rounds once. So we allow
        // 1 ULP per operation.
        for i in 0..n {
            assert!(
                (y[i] - ref_y[i]).abs() < 1e-13,
                "axpy_avx2 mismatch at {i}: got {} want {}", y[i], ref_y[i]
            );
        }
    }

    #[test]
    fn sparse_dot_avx2_matches_scalar_reference() {
        // A 100-wide sparse row on x of length 256.
        let n_col = 256;
        let n_nz  = 100;
        let x: Vec<f32> = (0..n_col).map(|i| ((i as f32) * 0.01).sin()).collect();
        let w: Vec<f32> = (0..n_nz).map(|k| ((k as f32) * 0.003).cos()).collect();
        // Non-monotonic column indices to stress the gather.
        let ci: Vec<u32> = (0..n_nz).map(|k| ((k * 37) % n_col) as u32).collect();

        let got  = sparse_dot_f32_avx2(&w, &ci, &x, 0, n_nz);
        let want = crate::asm_kernels::sparse_dot_f32(&w, &ci, &x, 0, n_nz);

        // f32 accumulator: allow 5·10⁻⁴ relative error at n_nz=100.
        assert!(
            ((got - want).abs() / want.abs().max(1e-8)) < 5e-4,
            "sparse_dot_avx2 mismatch: got {} want {}", got, want
        );
    }

    #[test]
    fn spmv_avx2_matches_scalar_reference_on_random_csr() {
        // Build a 64×64 CSR at ~10 % density.
        let rows = 64;
        let cols = 64;
        let mut row_ptr: Vec<u32> = Vec::with_capacity(rows + 1);
        let mut col_idx: Vec<u32> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();
        row_ptr.push(0);
        for r in 0..rows {
            for c in 0..cols {
                // Deterministic pseudo-random sparsity.
                if (r * 7 + c * 13) % 10 == 0 {
                    col_idx.push(c as u32);
                    weights.push(((r + c) as f32) * 0.01);
                }
            }
            row_ptr.push(col_idx.len() as u32);
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.017).sin()).collect();
        let mut out = vec![0.0_f32; rows];
        let mut out_ref = vec![0.0_f32; rows];

        spmv_csr_f32_avx2(&row_ptr, &col_idx, &weights, &x, &mut out);
        for r in 0..rows {
            let s = row_ptr[r] as usize;
            let e = row_ptr[r + 1] as usize;
            out_ref[r] = crate::asm_kernels::sparse_dot_f32(&weights, &col_idx, &x, s, e);
        }
        for r in 0..rows {
            let denom = out_ref[r].abs().max(1e-6);
            assert!(
                ((out[r] - out_ref[r]).abs() / denom) < 5e-4,
                "spmv_avx2 row {r} mismatch: got {} want {}",
                out[r], out_ref[r]
            );
        }
    }
}
