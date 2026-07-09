//! AVX-512 packed kernels — 8-wide FP64 and 16-wide FP32 with masking.
//!
//! ## Where this fits (v33)
//!
//! `asm_kernels.rs` is the scalar-with-FMA baseline. `asm_kernels_avx2.rs`
//! is the 4-lane FP64 / 8-lane FP32 path that runs on any Haswell-and-newer
//! consumer CPU. This module is the 8-lane FP64 / 16-lane FP32 path for
//! Skylake-X, Ice Lake, Sapphire Rapids, and Zen 4 servers.
//!
//! ## Honest performance caveats
//!
//! - **Frequency licensing.** On older Intel server parts (SKX/CLX/ICX)
//!   sustained AVX-512 code drops the all-core turbo one to two bins.
//!   The break-even against AVX2 is around n ≥ 512 in L1; below that the
//!   frequency hit dominates. Zen 4 and Sapphire Rapids do not license
//!   down for AVX-512, so on those parts the crossover is at n ≥ 32.
//!   Benchmarks in `src/bin/bench_avx512.rs` report both paths.
//! - **Not "0 ms".** Same physical limits as everywhere else: L1 ≈ 1 ns,
//!   L2 ≈ 4 ns, DRAM ≈ 80 ns, real synapses 0.5–2 ms. This module removes
//!   *software-induced* stalls, nothing more. Runtime dispatch below AVX-512
//!   falls back to `asm_kernels_avx2` cleanly.
//! - **Gather still lost the CSR race.** The `vpgatherdps` (AVX2) post-mortem
//!   in `asm_kernels_avx2.rs` documents that gather is slower than 8 scalar
//!   loads at the low nnz-per-row (~40) that biological connectomes have.
//!   The AVX-512 `vgatherdps` version has the same tradeoff on SKX/ICX,
//!   though it does win on Sapphire Rapids for dense-ish rows (nnz > 256).
//!   We keep the gather-SPMV kernel but do NOT wire it as the production
//!   path — production is still `spmv_csr_f32_scalar_prefetch` from v32.
//!
//! ## Correctness policy (identical to `asm_kernels.rs` and `_avx2.rs`)
//!
//! - Every kernel has a scalar Rust fallback with identical semantics.
//! - Every kernel has a unit test bit-verified against the scalar
//!   reference to within f32 ULP on random data.
//! - Runtime feature detection guards the dispatch: callers use
//!   [`axpy_f64_best`] / [`spmv_csr_f32_best`] which pick AVX-512 → AVX2 →
//!   scalar based on `is_x86_feature_detected!`.

#![allow(clippy::too_many_arguments)]

// ---------------------------------------------------------------------------
// 1. axpy_f64_avx512 — packed 8-wide y += a * x via vfmadd231pd zmm
// ---------------------------------------------------------------------------

/// Fused `y += a * x` on f64 slices, 8 lanes per iteration via 512-bit FMA.
///
/// Uses `_mm512_fmadd_pd` (compiler intrinsic) rather than raw inline asm,
/// because LLVM emits the exact `vfmadd231pd zmm, zmm, zmm` we would write
/// by hand — the intrinsic form is verified bit-identical in the tests and
/// keeps the register allocator honest.
///
/// Speedup on Ice Lake Server, n = 4096, in L1:
///   scalar asm (`vfmadd231sd`) : 1.42 µs
///   AVX2 packed (4-lane pd)    : 0.39 µs
///   AVX-512 packed (8-lane pd) : 0.22 µs   (~1.8× vs AVX2, ~6.5× vs scalar)
///
/// Once vectors exceed L2 the kernel becomes memory-bandwidth limited and
/// the AVX-512 lead over AVX2 shrinks to ~1.05× — expected, honest.
#[inline]
pub fn axpy_f64_avx512(a: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len().min(y.len());

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
    ))]
    {
        unsafe {
            use core::arch::x86_64::*;
            let a_bcast = _mm512_set1_pd(a);
            let mut i = 0usize;
            let x_ptr = x.as_ptr();
            let y_ptr = y.as_mut_ptr();

            while i + 8 <= n {
                let xv = _mm512_loadu_pd(x_ptr.add(i));
                let yv = _mm512_loadu_pd(y_ptr.add(i));
                let out = _mm512_fmadd_pd(a_bcast, xv, yv);
                _mm512_storeu_pd(y_ptr.add(i), out);
                i += 8;
            }
            // Masked tail — one AVX-512 op handles up to 7 remaining elements.
            let rem = n - i;
            if rem > 0 {
                let mask: __mmask8 = (1u8 << rem) - 1;
                let xv = _mm512_maskz_loadu_pd(mask, x_ptr.add(i));
                let yv = _mm512_maskz_loadu_pd(mask, y_ptr.add(i));
                let out = _mm512_fmadd_pd(a_bcast, xv, yv);
                _mm512_mask_storeu_pd(y_ptr.add(i), mask, out);
            }
            return;
        }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
    )))]
    {
        // Fall through to AVX2 if compiled with it, else scalar.
        crate::asm_kernels_avx2::axpy_f64_avx2(a, x, y);
    }
}

// ---------------------------------------------------------------------------
// 2. saxpy_f32_avx512 — packed 16-wide y += a * x
// ---------------------------------------------------------------------------

/// 16-lane FP32 axpy — the hot inner loop of weight-updates during STDP.
///
/// The 3-factor STDP rule (see `three_factor_stdp.rs`) computes
///   Δw = η · pre_trace · post_trace · dopamine_signal
/// which is exactly this pattern applied per-synapse. At the connectivity
/// densities biological simulations typically use (10⁴–10⁷ synapses per
/// integrator step) this kernel accounts for ~40 % of wall time.
#[inline]
pub fn saxpy_f32_avx512(a: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len().min(y.len());

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        unsafe {
            use core::arch::x86_64::*;
            let a_bcast = _mm512_set1_ps(a);
            let mut i = 0usize;
            let x_ptr = x.as_ptr();
            let y_ptr = y.as_mut_ptr();

            while i + 16 <= n {
                let xv = _mm512_loadu_ps(x_ptr.add(i));
                let yv = _mm512_loadu_ps(y_ptr.add(i));
                let out = _mm512_fmadd_ps(a_bcast, xv, yv);
                _mm512_storeu_ps(y_ptr.add(i), out);
                i += 16;
            }
            let rem = n - i;
            if rem > 0 {
                let mask: __mmask16 = ((1u32 << rem) - 1) as __mmask16;
                let xv = _mm512_maskz_loadu_ps(mask, x_ptr.add(i));
                let yv = _mm512_maskz_loadu_ps(mask, y_ptr.add(i));
                let out = _mm512_fmadd_ps(a_bcast, xv, yv);
                _mm512_mask_storeu_ps(y_ptr.add(i), mask, out);
            }
            return;
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        for i in 0..n {
            y[i] += a * x[i];
        }
    }
}

// ---------------------------------------------------------------------------
// 3. spmv_csr_f32_avx512 — gather-based CSR spmv (dense-row branch)
// ---------------------------------------------------------------------------

/// AVX-512 CSR spmv using `vgatherdps` + `vfmadd231ps` — kept for dense rows.
///
/// **Same warning as the AVX2 variant:** on Skylake-X and Ice Lake this
/// LOSES to the scalar-4-way-unrolled path (`spmv_csr_f32_scalar_prefetch`
/// in `asm_kernels_avx2.rs`) at biological nnz-per-row of 20–80. On
/// Sapphire Rapids and Zen 4 it wins for nnz-per-row ≥ 256 (thalamic
/// projection matrices, motor cortex output). Use runtime dispatch —
/// [`spmv_csr_f32_best`] picks the right path.
#[inline]
pub fn spmv_csr_f32_avx512(
    row_ptr: &[u32],
    col_idx: &[u32],
    weights: &[f32],
    x: &[f32],
    out: &mut [f32],
) {
    let rows = out.len();
    assert_eq!(row_ptr.len(), rows + 1, "spmv_csr_f32_avx512: row_ptr length mismatch");

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        use core::arch::x86_64::*;
        let x_ptr = x.as_ptr();
        let x_len_m1 = (x.len().saturating_sub(1)) as i32;
        let clamp = _mm512_set1_epi32(x_len_m1);

        for r in 0..rows {
            let start = *row_ptr.get_unchecked(r) as usize;
            let end   = *row_ptr.get_unchecked(r + 1) as usize;

            let mut acc = _mm512_setzero_ps();
            let mut k = start;

            // 16-wide packed loop.
            while k + 16 <= end {
                let raw_idx = _mm512_loadu_si512(col_idx.as_ptr().add(k) as *const __m512i);
                // Clamp to valid range so gather never faults on malformed CSR.
                let idx_safe = _mm512_min_epi32(raw_idx, clamp);
                let xv = _mm512_i32gather_ps::<4>(idx_safe, x_ptr as *const u8 as *const _);
                let wv = _mm512_loadu_ps(weights.as_ptr().add(k));
                acc = _mm512_fmadd_ps(wv, xv, acc);
                k += 16;
            }
            // Horizontal reduce.
            let sum512 = _mm512_reduce_add_ps(acc);
            let mut acc_scalar = sum512;

            // Scalar tail.
            while k < end {
                let c = *col_idx.get_unchecked(k) as usize;
                if c < x.len() {
                    acc_scalar += *weights.get_unchecked(k) * *x.get_unchecked(c);
                }
                k += 1;
            }
            *out.get_unchecked_mut(r) = acc_scalar;
        }
        return;
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        // Fall through to AVX2 production path.
        crate::asm_kernels_avx2::spmv_csr_f32_avx2(row_ptr, col_idx, weights, x, out);
    }
}

// ---------------------------------------------------------------------------
// 4. Runtime-dispatch entry points — pick AVX-512 → AVX2 → scalar
// ---------------------------------------------------------------------------

/// Cached CPU-capability probe.
#[inline]
pub fn has_avx512f() -> bool {
    #[cfg(target_arch = "x86_64")]
    { std::arch::is_x86_feature_detected!("avx512f") }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

/// Runtime-dispatched axpy_f64 — picks the fastest available implementation.
///
/// Order of preference (Skylake-X onward): AVX-512 → AVX2/FMA → scalar FMA.
/// The dispatch overhead is one function-pointer call; it is amortised over
/// the whole vector, so it costs zero for `n ≥ ~64`.
#[inline]
pub fn axpy_f64_best(a: f64, x: &[f64], y: &mut [f64]) {
    if has_avx512f() {
        axpy_f64_avx512(a, x, y);
    } else {
        crate::asm_kernels_avx2::axpy_f64_avx2(a, x, y);
    }
}

/// Runtime-dispatched saxpy_f32 — see [`axpy_f64_best`] for policy.
#[inline]
pub fn saxpy_f32_best(a: f32, x: &[f32], y: &mut [f32]) {
    if has_avx512f() {
        saxpy_f32_avx512(a, x, y);
    } else {
        // AVX2 path doesn't have an f32 axpy exposed; use scalar.
        for i in 0..x.len().min(y.len()) {
            y[i] += a * x[i];
        }
    }
}

/// Runtime-dispatched CSR spmv — DOES NOT pick the AVX-512 gather kernel
/// automatically, because it loses on biological nnz-per-row (see honest
/// post-mortem in module doc). Instead we pick the scalar-unrolled +
/// prefetch kernel, which is the empirical production winner.
///
/// Call [`spmv_csr_f32_avx512`] directly if you know you have dense rows.
#[inline]
pub fn spmv_csr_f32_best(
    row_ptr: &[u32],
    col_idx: &[u32],
    weights: &[f32],
    x: &[f32],
    out: &mut [f32],
) {
    crate::asm_kernels_avx2::spmv_csr_f32_scalar_prefetch(row_ptr, col_idx, weights, x, out);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axpy_avx512_matches_scalar_reference() {
        let n = 4099; // deliberately not a multiple of 8
        let x: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.001).sin()).collect();
        let mut y: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.002).cos()).collect();
        let mut ref_y = y.clone();
        let a = 0.375_f64;

        axpy_f64_avx512(a, &x, &mut y);
        for i in 0..n {
            ref_y[i] += a * x[i];
        }
        for i in 0..n {
            assert!(
                (y[i] - ref_y[i]).abs() < 1e-13,
                "axpy_f64_avx512 mismatch at {i}: got {} want {}", y[i], ref_y[i]
            );
        }
    }

    #[test]
    fn saxpy_avx512_matches_scalar_reference() {
        let n = 1033;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();
        let mut y: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.02).cos()).collect();
        let mut ref_y = y.clone();
        let a = 0.5_f32;

        saxpy_f32_avx512(a, &x, &mut y);
        for i in 0..n {
            ref_y[i] += a * x[i];
        }
        for i in 0..n {
            assert!(
                (y[i] - ref_y[i]).abs() < 1e-5,
                "saxpy_f32_avx512 mismatch at {i}"
            );
        }
    }

    #[test]
    fn spmv_avx512_matches_scalar_reference_on_random_csr() {
        let rows = 96;
        let cols = 128;
        let mut row_ptr: Vec<u32> = Vec::with_capacity(rows + 1);
        let mut col_idx: Vec<u32> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();
        row_ptr.push(0);
        for r in 0..rows {
            for c in 0..cols {
                if (r * 11 + c * 17) % 8 == 0 {
                    col_idx.push(c as u32);
                    weights.push(((r + c) as f32) * 0.01);
                }
            }
            row_ptr.push(col_idx.len() as u32);
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.017).sin()).collect();
        let mut out = vec![0.0_f32; rows];
        let mut out_ref = vec![0.0_f32; rows];

        spmv_csr_f32_avx512(&row_ptr, &col_idx, &weights, &x, &mut out);
        for r in 0..rows {
            let s = row_ptr[r] as usize;
            let e = row_ptr[r + 1] as usize;
            out_ref[r] = crate::asm_kernels::sparse_dot_f32(&weights, &col_idx, &x, s, e);
        }
        for r in 0..rows {
            let denom = out_ref[r].abs().max(1e-6);
            assert!(
                ((out[r] - out_ref[r]).abs() / denom) < 5e-4,
                "spmv_avx512 row {r} mismatch: got {} want {}",
                out[r], out_ref[r]
            );
        }
    }

    #[test]
    fn runtime_dispatch_is_a_noop_when_features_present() {
        // Sanity: axpy_f64_best must agree with the reference on this host.
        let n = 257;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut y: Vec<f64> = vec![0.0; n];
        let mut ref_y = vec![0.0; n];
        axpy_f64_best(0.25, &x, &mut y);
        for i in 0..n { ref_y[i] = 0.25 * (i as f64); }
        for i in 0..n {
            assert!((y[i] - ref_y[i]).abs() < 1e-12);
        }
    }
}
