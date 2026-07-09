//! Micro-benchmark of the compression layer.
//!
//! Prints a Before-vs-After table so users can validate compression gains.

use neurorust::compression::{BitRaster, MembraneTrace, SparseConnectivity, SpikeTrain};
use rand::prelude::*;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let dt = 0.05;
    let ms = 5_000.0; // 5 s of data
    let n_steps = (ms / dt) as usize;

    // ----- spike train --------------------------------------------------
    let mut train = SpikeTrain::new(dt);
    let firing_rate_hz = 20.0;
    let expected_spikes = (firing_rate_hz * ms / 1000.0) as usize;
    let mut spikes = Vec::with_capacity(expected_spikes);
    let mut t = 0.0;
    while t < ms {
        // Poisson-ish ISI ~ Exp(rate).
        let isi = -(rng.gen::<f64>().max(1e-9)).ln() / firing_rate_hz * 1000.0;
        t += isi;
        if t < ms { spikes.push(t); }
    }
    let n_spikes = spikes.len();
    train.extend(spikes).unwrap();
    let train_naive = n_spikes * 8; // u64 timestamps

    // ----- V-trace ------------------------------------------------------
    let mut v = MembraneTrace::new(-100.0, 60.0, dt);
    let trace: Vec<f32> = (0..n_steps)
        .map(|i| -65.0 + 10.0 * ((i as f32) * 0.001).sin())
        .collect();
    v.extend(trace.iter().copied()).unwrap();
    let v_naive = n_steps * 4;

    // ----- connectivity -------------------------------------------------
    let rows = 500;
    let cols = 500;
    let density = 0.05;
    let dense: Vec<f32> = (0..rows * cols)
        .map(|_| if rng.gen::<f64>() < density { rng.gen::<f32>() } else { 0.0 })
        .collect();
    let csr = SparseConnectivity::from_dense(rows, cols, &dense, 1e-9);
    let csr_naive = rows * cols * 4;
    let csr_now = csr.weights.len() * 4 + csr.col_idx.len() * 4 + csr.row_ptr.len() * 4;

    // ----- raster -------------------------------------------------------
    let raster_n = 1024;
    let raster_bins = 4096;
    let _raster = BitRaster::zeros(raster_n, raster_bins);
    let raster_naive = raster_n * raster_bins; // 1 B per bool
    let raster_now = (raster_n * raster_bins).div_ceil(8);

    // ----- report -------------------------------------------------------
    println!("=== neurorust compression benchmark ===\n");
    println!("| stream                | naive bytes | after bytes | ratio |");
    println!("|-----------------------|-------------|-------------|-------|");
    println!("| Spike train ({:>4} sp) | {:>11} | {:>11} | {:>4.1}× |",
        n_spikes, train_naive, train.compressed_bytes(),
        train_naive as f64 / train.compressed_bytes().max(1) as f64);
    println!("| V-trace     ({:>4} sm) | {:>11} | {:>11} | {:>4.1}× |",
        n_steps, v_naive, v.compressed_bytes(), v.compression_ratio());
    println!("| Connectivity 500×500  | {:>11} | {:>11} | {:>4.1}× |",
        csr_naive, csr_now, csr_naive as f64 / csr_now as f64);
    println!("| Raster 1024×4096      | {:>11} | {:>11} | {:>4.1}× |",
        raster_naive, raster_now, raster_naive as f64 / raster_now as f64);
}
