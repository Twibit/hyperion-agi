//! # brainstem — homeostasis, drives, arousal, safety supervisor
//!
//! Biological homologue: the **brainstem** and **hypothalamus** — the
//! evolutionarily oldest control layer, responsible for keeping the organism
//! alive. In vivo, a brainstem lesion is instantly lethal; a cortex lesion
//! merely disables cognition. This kernel therefore gives the brainstem
//! **hard priority** over cortex.
//!
//! ## Kernel role
//!
//! - Tracks four homeostatic drives (energy, temperature, sleep pressure,
//!   safety), with set-points and thresholds.
//! - Runs *first* in every kernel tick.
//! - Emits [`crate::bus::Drive`] alerts when a set-point is violated.
//! - Can request the scheduler throttle or halt cognition when critical
//!   thresholds are crossed — a software analogue of "sickness behaviour"
//!   and the fainting reflex.

use crate::bus::{Drive, NeuralBus};
use std::collections::HashMap;

/// Homeostatic state per drive: current value, set-point, and alert threshold.
#[derive(Debug, Clone)]
pub struct DriveState {
    /// Instantaneous value (0..1).
    pub value: f32,
    /// Ideal set-point (typically 1.0 = "full").
    pub set_point: f32,
    /// Emit an alert if value < set_point - threshold.
    pub alert_threshold: f32,
    /// Passive decay per tick.
    pub decay_per_tick: f32,
}

impl DriveState {
    /// Build a fresh drive at set-point.
    pub fn new(set_point: f32, alert_threshold: f32, decay_per_tick: f32) -> Self {
        Self { value: set_point, set_point, alert_threshold, decay_per_tick }
    }
    /// Deviation below set-point (0 if at or above).
    pub fn deviation(&self) -> f32 {
        (self.set_point - self.value).max(0.0)
    }
    /// Should we alert?
    pub fn alerting(&self) -> bool {
        self.deviation() > self.alert_threshold
    }
}

/// The brainstem module.
pub struct Brainstem {
    /// Per-drive state.
    pub drives: HashMap<Drive, DriveState>,
    /// When true, cortex must throttle this tick.
    pub throttle_cortex: bool,
    /// Global arousal level in [0..1] (analogue of locus-coeruleus tone).
    pub arousal: f32,
}

impl Brainstem {
    /// Build a brainstem with default drives.
    pub fn new() -> Self {
        let mut drives = HashMap::new();
        drives.insert(Drive::Energy,      DriveState::new(1.0, 0.4, 0.0005));
        drives.insert(Drive::Temperature, DriveState::new(1.0, 0.3, 0.0002));
        drives.insert(Drive::Sleep,       DriveState::new(1.0, 0.5, 0.0003));
        drives.insert(Drive::Safety,      DriveState::new(1.0, 0.2, 0.0000));
        Self { drives, throttle_cortex: false, arousal: 0.5 }
    }

    /// Advance homeostatic state one tick and emit alerts as needed.
    ///
    /// Returns the number of alerts emitted. Must be called BEFORE
    /// cortex.tick() every kernel cycle — the scheduler orders this
    /// automatically in the demo.
    pub fn tick(&mut self, bus: &NeuralBus) -> usize {
        let mut alerts = 0;
        self.throttle_cortex = false;
        for (drive, s) in self.drives.iter_mut() {
            s.value = (s.value - s.decay_per_tick).clamp(0.0, s.set_point);
            if s.alerting() {
                let _ = bus.alert_tx.try_send((*drive, s.deviation()));
                alerts += 1;
                // If deviation is severe, force cortex to throttle.
                if s.deviation() > 2.0 * s.alert_threshold {
                    self.throttle_cortex = true;
                }
            }
        }
        // Simple arousal update: hunger and safety threats raise arousal;
        // sleep pressure lowers it.
        let hunger = self.drives.get(&Drive::Energy).map(|s| s.deviation()).unwrap_or(0.0);
        let threat = self.drives.get(&Drive::Safety).map(|s| s.deviation()).unwrap_or(0.0);
        let sleep  = self.drives.get(&Drive::Sleep) .map(|s| s.deviation()).unwrap_or(0.0);
        self.arousal = (0.5 + 0.5 * (hunger + threat) - 0.3 * sleep).clamp(0.0, 1.0);
        alerts
    }

    /// Consume a "food/refuel" event.
    pub fn refuel(&mut self, drive: Drive, amount: f32) {
        if let Some(s) = self.drives.get_mut(&drive) {
            s.value = (s.value + amount).clamp(0.0, s.set_point);
        }
    }
}

impl Default for Brainstem {
    fn default() -> Self { Self::new() }
}
