//! v33 benchmark harness: AVX-512 vs AVX2 vs scalar on axpy / spmv,
//! plus a wall-clock timing of the three-factor STDP tick and WTA argmax.
//!
//! Dependency-free. Uses `std::time::Instant`. Runs 25 iterations per
//! measurement and reports the median (robust to jitter on a shared host).
//!
//! Reproduce:
//! ```sh
//! cd agi-bundle/neurorust
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --bin neurorust-bench-v33
//! ```
//!
//! Honest reporting policy (identical to v32 bench):
//! - No fabricated speedups. If a path isn't available on the current CPU
//!   we say `n/a — feature not present` and skip.
//! - Report scalar reference alongside every SIMD number so the reader
//!   can sanity-check the speedup ratio themselves.

use std::hint::black_box;
use std::time::Instant;

use neurorust::asm_kernels;
use neurorust::asm_kernels_avx2;
use neurorust::asm_kernels_avx512;
use neurorust::three_factor_stdp::ThreeFactorStdp;
use neurorust::wta_kernel::{wta_argmax_f32, wta_lateral_inhibit};

const ITERS: usize = 25;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn bench<F: FnMut()>(mut f: F) -> f64 {
    // Warmup.
    for _ in 0..3 { f(); }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f64());
    }
    median(samples)
}

fn fmt_ns(sec: f64) -> String {
    if sec < 1e-6 { format!("{:.1} ns", sec * 1e9) }
    else if sec < 1e-3 { format!("{:.2} µs", sec * 1e6) }
    else { format!("{:.2} ms", sec * 1e3) }
}

fn bench_axpy() {
    println!("== axpy: y += a * x (f64) ==");
    println!("{:>8}  {:>14}  {:>14}  {:>14}  {:>10}  {:>10}",
             "n", "scalar", "AVX2 (4×pd)", "AVX-512 (8×pd)", "AVX2 vs scalar", "AVX-512 vs scalar");
    let a = 0.375_f64;

    let has_avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    let has_avx512 = std::is_x86_feature_detected!("avx512f");
    println!("(host caps: AVX2={}, AVX-512F={})", has_avx2, has_avx512);

    for &n in &[64usize, 256, 1024, 4096, 16384, 65536, 262144] {
        let x: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.001).sin()).collect();
        let mut y_scalar = vec![0.0_f64; n];
        let mut y_avx2   = vec![0.0_f64; n];
        let mut y_avx512 = vec![0.0_f64; n];

        let t_scalar = bench(|| {
            asm_kernels::axpy_f64(black_box(a), black_box(&x), black_box(&mut y_scalar));
        });
        let t_avx2 = if has_avx2 {
            bench(|| {
                asm_kernels_avx2::axpy_f64_avx2(black_box(a), black_box(&x), black_box(&mut y_avx2));
            })
        } else { f64::NAN };
        let t_avx512 = if has_avx512 {
            bench(|| {
                asm_kernels_avx512::axpy_f64_avx512(black_box(a), black_box(&x), black_box(&mut y_avx512));
            })
        } else { f64::NAN };

        let sp_avx2 = if t_avx2.is_finite() { t_scalar / t_avx2 } else { f64::NAN };
        let sp_avx512 = if t_avx512.is_finite() { t_scalar / t_avx512 } else { f64::NAN };

        let avx2_str  = if t_avx2.is_finite()   { fmt_ns(t_avx2)   } else { "n/a".into() };
        let avx512_str = if t_avx512.is_finite() { fmt_ns(t_avx512) } else { "n/a".into() };
        let sp_avx2_str = if sp_avx2.is_finite() { format!("{:.2}×", sp_avx2) } else { "n/a".into() };
        let sp_avx512_str = if sp_avx512.is_finite() { format!("{:.2}×", sp_avx512) } else { "n/a".into() };

        println!("{:>8}  {:>14}  {:>14}  {:>14}  {:>10}  {:>10}",
                 n, fmt_ns(t_scalar), avx2_str, avx512_str, sp_avx2_str, sp_avx512_str);
    }
    println!();
}

fn bench_spmv() {
    println!("== CSR spmv: out = W · x (f32, biological ~10% density) ==");
    println!("{:>8}  {:>10}  {:>14}  {:>14}  {:>10}",
             "rows", "nnz/row", "scalar+prefetch", "AVX-512 gather", "gather vs scalar");

    let has_avx512 = std::is_x86_feature_detected!("avx512f");
    println!("(host caps: AVX-512F={}). Reminder: gather-SPMV lost the race at biological",
             has_avx512);
    println!("nnz/row on v32; we report it here honestly so the reader sees the loss.");

    for &size in &[100usize, 500, 2000] {
        // Build a size×size CSR at ~10% density.
        let rows = size;
        let cols = size;
        let mut row_ptr: Vec<u32> = Vec::with_capacity(rows + 1);
        let mut col_idx: Vec<u32> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();
        row_ptr.push(0);
        for r in 0..rows {
            for c in 0..cols {
                if (r * 41 + c * 71) % 10 == 0 {
                    col_idx.push(c as u32);
                    weights.push(((r + c) as f32) * 0.001);
                }
            }
            row_ptr.push(col_idx.len() as u32);
        }
        let nnz_per_row = col_idx.len() as f64 / rows as f64;
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.017).sin()).collect();
        let mut out_scalar = vec![0.0_f32; rows];
        let mut out_avx512 = vec![0.0_f32; rows];

        let t_scalar = bench(|| {
            asm_kernels_avx2::spmv_csr_f32_scalar_prefetch(
                black_box(&row_ptr), black_box(&col_idx), black_box(&weights),
                black_box(&x), black_box(&mut out_scalar),
            );
        });
        let t_avx512 = if has_avx512 {
            bench(|| {
                asm_kernels_avx512::spmv_csr_f32_avx512(
                    black_box(&row_ptr), black_box(&col_idx), black_box(&weights),
                    black_box(&x), black_box(&mut out_avx512),
                );
            })
        } else { f64::NAN };

        let sp = if t_avx512.is_finite() { t_scalar / t_avx512 } else { f64::NAN };
        let avx512_str = if t_avx512.is_finite() { fmt_ns(t_avx512) } else { "n/a".into() };
        let sp_str = if sp.is_finite() { format!("{:.2}×", sp) } else { "n/a".into() };
        println!("{:>8}  {:>10.1}  {:>14}  {:>14}  {:>10}",
                 size, nnz_per_row, fmt_ns(t_scalar), avx512_str, sp_str);
    }
    println!();
}

fn bench_wta() {
    println!("== WTA argmax (lateral inhibition, f32) ==");
    println!("{:>8}  {:>14}  {:>14}",
             "pool_n", "wta_argmax_f32", "throughput");
    for &n in &[16usize, 64, 256, 1024, 4096, 16384] {
        let mut pot: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).sin()).collect();
        pot[n / 3] = 100.0; // planted max
        let t = bench(|| {
            let r = wta_argmax_f32(black_box(&pot));
            black_box(r);
        });
        let throughput = (n as f64) / t / 1e9; // billion elements / second
        println!("{:>8}  {:>14}  {:>10.2} G/s",
                 n, fmt_ns(t), throughput);
    }
    println!();
    println!("== WTA lateral inhibit (argmax + zero non-winners) ==");
    for &n in &[16usize, 64, 256, 1024, 4096] {
        let mut pot: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).sin()).collect();
        let mut spk: Vec<u8>  = vec![1; n];
        pot[n / 3] = 100.0;
        let t = bench(|| {
            wta_lateral_inhibit(black_box(&mut pot), black_box(&mut spk));
            spk.iter_mut().for_each(|s| *s = 1); // restore for next iter
        });
        println!("  pool_n={:>5}  {}", n, fmt_ns(t));
    }
    println!();
}

fn bench_stdp() {
    println!("== Three-factor STDP: one full tick (decay + accumulate + reward) ==");
    println!("{:>10}  {:>10}  {:>10}  {:>14}",
             "n_pre", "n_post", "synapses", "per-tick");
    for &(n_pre, n_post) in &[(128usize, 32), (512, 64), (2048, 128), (8192, 256)] {
        let mut stdp = ThreeFactorStdp::new(n_pre, n_post, 500.0);
        let pre: Vec<u8>  = (0..n_pre).map(|i| ((i * 7919) % 5 == 0) as u8).collect();
        let post: Vec<u8> = (0..n_post).map(|i| ((i * 3181) % 4 == 0) as u8).collect();
        let mut weights: Vec<u8> = vec![128; n_pre * n_post];
        let t = bench(|| {
            stdp.tick(black_box(&pre), black_box(&post), black_box(&mut weights), 0.1);
        });
        println!("{:>10}  {:>10}  {:>10}  {:>14}",
                 n_pre, n_post, n_pre * n_post, fmt_ns(t));
    }
    println!();
}

fn main() {
    println!("# neurorust v33 benchmark");
    println!();
    println!("Host CPU: {}", std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into()));
    println!("Iterations per measurement: {} (median reported).", ITERS);
    println!();
    bench_axpy();
    bench_spmv();
    bench_wta();
    bench_stdp();
    println!("# End of benchmark.");
    println!("# Reproduce: RUSTFLAGS=\"-C target-cpu=native\" cargo run --release --bin neurorust-bench-v33");
}
