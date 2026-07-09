//! `agikernel-demo` — a tiny "mini-AGI" cycle end-to-end.
//!
//! This spins up the full kernel with sensible small dimensions and runs 200
//! ticks (~5 seconds of simulated brain time). It prints per-tick prediction
//! error, dopamine, arousal, and any emitted motor commands.
//!
//! The purpose is to demonstrate that the biological invariants encoded in
//! the type system compile and run — not to solve a task.

use agikernel::bus::Drive;
use agikernel::Syscall;

fn main() {
    env_logger::init();
    println!("agikernel v{} — mini-AGI demo\n", agikernel::VERSION);

    // depth=3 (like V1→V2→PFC), 32 cells/layer, 8 sensory channels, top-K=4.
    let mut sys = Syscall::new(3, 32, 8, 4, 25.0);

    // A trivial "environment": every tick, we present a random 8-D feature
    // vector on modality 0 and, on a small fraction of ticks, deliver a
    // scalar reward.
    use rand::Rng;
    let mut rng = rand::thread_rng();

    println!(" tick   t_ms   err     DA    NE    arousal   action");
    println!("-----------------------------------------------------");
    for t in 0..200 {
        // Present features.
        let features: Vec<f32> = (0..8).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        sys.sense(0, features);

        // Occasionally refuel (feeding) so drives don't collapse in demo.
        if t % 40 == 0 { sys.refuel(Drive::Energy, 0.1); }
        // Every ~50 ticks, deliver a reward.
        if t % 53 == 0 { sys.reward(0.3); }

        let (err, action) = sys.tick();

        if t % 10 == 0 {
            println!(
                " {:>4}  {:>6.0}  {:>5.3}  {:>4.2}  {:>4.2}  {:>7.3}   {:?}",
                t, sys.t_ms(), err,
                sys.neuromod.dopamine(),
                sys.neuromod.noradrenaline(),
                sys.brainstem.arousal,
                action,
            );
        }
    }

    println!("\nCortex compressed memory footprint: {} bytes",
             sys.cortex.memory_bytes());
    println!("Hippocampal episodes stored: {}", sys.hippocampus.len());
    println!("\nDemo complete. Type-level biological invariants held throughout.");
}
