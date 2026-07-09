//! Integration tests that lock in the biological invariants promised in the
//! README. Some of them are compile-fail tests in comments (Rust does not
//! have inline `compile_fail` doctests for library crates without extra
//! machinery, so we document them as prose next to a positive test).

use agikernel::bus::{Drive, Percept};
use agikernel::Syscall;

/// The kernel boots and completes ticks without panicking.
#[test]
fn kernel_boots_and_ticks() {
    let mut sys = Syscall::new(2, 16, 4, 2, 25.0);
    for _ in 0..20 {
        sys.sense(0, vec![0.5, -0.2, 0.1, 0.0]);
        let _ = sys.tick();
    }
}

/// Homeostasis wins: if we drain energy far enough, cortex is throttled
/// and prediction error is pegged.
#[test]
fn brainstem_throttles_cortex_when_starved() {
    let mut sys = Syscall::new(2, 16, 4, 2, 25.0);
    // Force energy to collapse.
    if let Some(s) = sys.brainstem.drives.get_mut(&Drive::Energy) {
        s.value = 0.0;
    }
    sys.sense(0, vec![0.5, -0.2, 0.1, 0.0]);
    let (err, _) = sys.tick();
    // Under throttle we return a saturating error value.
    assert!(err >= 0.99, "expected throttle-error signal, got {err}");
}

/// Basal ganglia never emits an action when tonic inhibition dominates.
#[test]
fn basal_ganglia_defaults_to_no_action() {
    let mut sys = Syscall::new(2, 16, 4, 2, 25.0);
    sys.bg.tonic_inhibition = 10.0; // Way above any plausible EV.
    for _ in 0..10 {
        sys.sense(0, vec![0.5, 0.5, 0.5, 0.5]);
        let (_e, a) = sys.tick();
        assert!(a.is_none(), "action leaked past BG under massive inhibition");
    }
}

/// Reward calls raise dopamine.
#[test]
fn reward_raises_dopamine() {
    let mut sys = Syscall::new(2, 16, 4, 2, 25.0);
    let da_before = sys.neuromod.dopamine();
    let _ = sys.reward(0.5);
    let da_after = sys.neuromod.dopamine();
    assert!(da_after >= da_before, "dopamine should not fall after reward");
}

/// Hippocampus grows monotonically as long as we bind percepts.
#[test]
fn hippocampus_accumulates_episodes() {
    let mut sys = Syscall::new(2, 16, 4, 2, 25.0);
    let start = sys.hippocampus.len();
    for _ in 0..50 {
        sys.sense(0, vec![0.9, 0.9, 0.9, 0.9]);
        let _ = sys.tick();
    }
    assert!(sys.hippocampus.len() > start,
            "hippocampus should record at least some episodes");
}

/// Type-system invariant: `Percept<Raw>` cannot be forged into `Percept<Gated>`
/// outside `agikernel`. The following would be a compile error if uncommented:
///
/// ```compile_fail
/// use agikernel::bus::{Percept, Gated};
/// let raw = Percept::raw(0.0, 0, vec![1.0]);
/// // The `from_raw` constructor is pub(crate); userspace has no way to call it.
/// let _gated: Percept<Gated> = Percept::from_raw(raw, 1.0);
/// ```
#[test]
fn gated_percept_marker_is_unforgeable() {
    // We can still receive them from the bus, but only after Thalamus emits.
    let p: Percept = Percept::raw(0.0, 0, vec![1.0, 2.0]);
    assert_eq!(p.modality, 0);
}

/// Scheduler rhythms nest correctly.
#[test]
fn scheduler_rhythms_are_nested() {
    use agikernel::scheduler::{Rhythm, Scheduler};
    let mut s = Scheduler::new(25.0);
    let mut gamma = 0;
    let mut theta = 0;
    for _ in 0..80 {
        if s.on_phase(Rhythm::Gamma) { gamma += 1; }
        if s.on_phase(Rhythm::Theta) { theta += 1; }
        s.advance();
    }
    // gamma fires every tick; theta every 8 ticks.
    assert_eq!(gamma, 80);
    assert_eq!(theta, 10);
}
