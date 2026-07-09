//! Micro-benchmark showing the O(1)-amortised staging fix in
//! `MembraneTrace::extend` / `SpikeTrain::push`. The original code
//! decompressed + recompressed the ENTIRE zstd payload on every step,
//! giving O(n²) simulation cost.

use neurorust::models::HodgkinHuxleyNeuron;
use neurorust::neuron::Neuron;
use std::time::Instant;

fn main() {
    let dt = 0.01;
    for &n_steps in &[1_000usize, 5_000, 10_000, 20_000, 40_000] {
        let mut n = HodgkinHuxleyNeuron::new(dt);
        let t0 = Instant::now();
        for _ in 0..n_steps {
            n.step(dt, 10.0);
        }
        let elapsed = t0.elapsed();
        let ns_per_step = elapsed.as_nanos() as f64 / n_steps as f64;
        println!(
            "HH  steps={:>7}  wall={:>10.3} ms  per-step={:>10.1} ns  spikes={}",
            n_steps,
            elapsed.as_secs_f64() * 1e3,
            ns_per_step,
            n.memory().spikes.len(),
        );
    }
}
