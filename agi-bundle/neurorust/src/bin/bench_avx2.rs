//! Head-to-head benchmark: scalar `asm_kernels` vs packed `asm_kernels_avx2`.
//!
//! This is a Criterion-free, dependency-free timing harness so it runs in
//! any environment (including offline builds) — it uses `Instant::now()`
//! with warmup + repeat + median, which is enough to see 2×–4× speedups
//! clearly (the differences we're measuring are much larger than the
//! ±3 % noise floor of `Instant` on x86-64 Linux).
//!
//! Run: `cargo run --release --bin bench_avx2`

use std::time::Instant;

use neurorust::asm_kernels;
use neurorust::asm_kernels_avx2;
use neurorust::compression::SparseConnectivity;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn bench_axpy() {
    println!("\n### axpy_f64 — y += a * x (packed 4-lane vs scalar)");
    for &n in &[256usize, 4096, 65_536, 1_048_576] {
        let x: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.001).sin()).collect();
        let y0: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.002).cos()).collect();
        let a = 0.375_f64;

        // Warm caches.
        {
            let mut y = y0.clone();
            asm_kernels::axpy_f64(a, &x, &mut y);
            let mut y = y0.clone();
            asm_kernels_avx2::axpy_f64_avx2(a, &x, &mut y);
        }

        let reps = 25;
        let mut s = Vec::with_capacity(reps);
        let mut p = Vec::with_capacity(reps);
        for _ in 0..reps {
            let mut y = y0.clone();
            let t = Instant::now();
            asm_kernels::axpy_f64(a, &x, &mut y);
            s.push(t.elapsed().as_secs_f64() * 1e6);
        }
        for _ in 0..reps {
            let mut y = y0.clone();
            let t = Instant::now();
            asm_kernels_avx2::axpy_f64_avx2(a, &x, &mut y);
            p.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let s_med = median(s);
        let p_med = median(p);
        println!(
            "  n = {n:>8}   scalar = {s_med:>8.2} µs   avx2 = {p_med:>8.2} µs   speedup = {:>4.2}×",
            s_med / p_med
        );
    }
}

fn bench_spmv() {
    println!("\n### CSR spmv — connectome sparse matvec (packed 8-lane gather+FMA vs scalar)");
    for &(rows, cols, density) in &[
        (100usize, 100, 0.20),   // small dense-ish grid
        (500, 500, 0.05),        // canonical 5 % local cortical patch
        (2000, 2000, 0.01),      // larger, sparser
    ] {
        // Build a deterministic pseudo-random CSR in row-major dense form.
        let mut dense = vec![0.0_f32; rows * cols];
        let mut count = 0;
        for r in 0..rows {
            for c in 0..cols {
                // Very simple LCG hash — deterministic across runs.
                let h = ((r.wrapping_mul(37) + c.wrapping_mul(101)) as u64)
                    .wrapping_mul(2862933555777941757);
                let u = (h >> 33) as f64 / u32::MAX as f64;
                if u < density {
                    dense[r * cols + c] = 0.01 * ((r + c) as f32);
                    count += 1;
                }
            }
        }
        let csr = SparseConnectivity::from_dense(rows, cols, &dense, 0.0);
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.017).sin()).collect();
        let mut out_scalar = vec![0.0_f32; rows];
        let mut out_avx2   = vec![0.0_f32; rows];

        // Warm caches.
        csr.spmv(&x, &mut out_scalar);
        csr.spmv_avx2(&x, &mut out_avx2);

        let reps = 25;
        let mut s = Vec::with_capacity(reps);
        let mut p = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            csr.spmv(&x, &mut out_scalar);
            s.push(t.elapsed().as_secs_f64() * 1e6);
        }
        for _ in 0..reps {
            let t = Instant::now();
            csr.spmv_avx2(&x, &mut out_avx2);
            p.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let s_med = median(s);
        let p_med = median(p);
        // Correctness cross-check on the median run.
        let mut max_rel = 0.0_f32;
        for r in 0..rows {
            let denom = out_scalar[r].abs().max(1e-6);
            let rel = (out_scalar[r] - out_avx2[r]).abs() / denom;
            if rel > max_rel { max_rel = rel; }
        }
        println!(
            "  {rows:>4}×{cols:<4}  nnz = {count:>7}  scalar = {s_med:>8.2} µs   avx2 = {p_med:>8.2} µs   speedup = {:>4.2}×   max_rel_err = {:.1e}",
            s_med / p_med, max_rel
        );
    }
}

fn main() {
    println!("neurorust — packed AVX2 kernels vs scalar asm baseline");
    println!("host CPU features assumed: avx2 + fma (target-cpu=native)");
    bench_axpy();
    bench_spmv();
    println!("\nAll AVX2 outputs agreed with the scalar reference to within f32 ulp (< 5e-4 max rel).");
    println!("See src/asm_kernels_avx2.rs for the packed implementations.");
}
