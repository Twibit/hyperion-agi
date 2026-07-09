//! # scheduler — oscillatory global time base
//!
//! Biological homologue: **thalamo-cortical rhythms**. The mammalian brain
//! does not free-run its regions; they are entrained by nested oscillations:
//!
//! | Rhythm  | Frequency    | Function                                     |
//! |---------|--------------|----------------------------------------------|
//! | Gamma γ | 30–80 Hz     | Local binding, cortical microcircuit clock   |
//! | Beta β  | 13–30 Hz     | Top-down predictions, motor preparation      |
//! | Alpha α | 8–13 Hz      | Cortical inhibition / attentional gating     |
//! | Theta θ | 4–8 Hz       | Hippocampal encoding vs. replay phase        |
//!
//! In this kernel, one **gamma tick ≈ 25 ms** is the smallest scheduling
//! quantum. Longer rhythms are derived by counting gamma ticks, so any
//! subsystem can ask "am I in a theta-trough right now?" and phase-lock its
//! work to it — mirroring, for example, how hippocampal replay preferentially
//! occurs during theta troughs and sharp-wave ripples.

use serde::{Deserialize, Serialize};

/// Which cortical rhythm phase a subsystem may want to phase-lock to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rhythm {
    /// Gamma — every tick.
    Gamma,
    /// Beta — every 2 gamma ticks (~50 ms → 20 Hz, in-band).
    Beta,
    /// Alpha — every 4 gamma ticks (~100 ms → 10 Hz).
    Alpha,
    /// Theta — every 8 gamma ticks (~200 ms → 5 Hz).
    Theta,
}

/// The global scheduler. Not a preemptive OS scheduler — it hands out
/// **phase information** to subsystems that then schedule their own work.
#[derive(Debug, Clone)]
pub struct Scheduler {
    tick_ms: f64,
    tick_count: u64,
}

impl Scheduler {
    /// Build a scheduler with the given tick length (defaults to 25 ms
    /// elsewhere).
    pub fn new(tick_ms: f64) -> Self { Self { tick_ms, tick_count: 0 } }

    /// Advance to the next gamma tick.
    pub fn advance(&mut self) { self.tick_count = self.tick_count.wrapping_add(1); }

    /// Current tick index.
    pub fn tick(&self) -> u64 { self.tick_count }

    /// Current simulation time (ms) since scheduler start.
    pub fn time_ms(&self) -> f64 { self.tick_count as f64 * self.tick_ms }

    /// Length of one tick in ms.
    pub fn tick_ms(&self) -> f64 { self.tick_ms }

    /// Whether the current tick coincides with the given rhythm's cycle start.
    pub fn on_phase(&self, r: Rhythm) -> bool {
        let period = match r {
            Rhythm::Gamma => 1,
            Rhythm::Beta  => 2,
            Rhythm::Alpha => 4,
            Rhythm::Theta => 8,
        };
        self.tick_count % period == 0
    }

    /// Is the current tick in the "trough" half of the theta cycle? This is
    /// when hippocampal replay is biologically most active.
    pub fn is_theta_trough(&self) -> bool {
        // Second half of an 8-tick theta cycle.
        (self.tick_count % 8) >= 4
    }
}

impl Default for Scheduler {
    fn default() -> Self { Self::new(crate::GAMMA_TICK_MS) }
}
