//! # neuromod — dopamine, noradrenaline, serotonin, acetylcholine
//!
//! Biological homologue: the **neuromodulatory nuclei** — VTA/SNc for
//! dopamine, locus coeruleus (LC) for noradrenaline, raphé nuclei for
//! serotonin, basal forebrain (NBM) for acetylcholine. These nuclei
//! **broadcast** rather than point-to-point; a small number of cells project
//! diffusely across cortex, modulating gain and plasticity globally.
//!
//! Canonical mappings adopted here (well-established in systems
//! neuroscience):
//!
//! - **Dopamine (DA)** – reward prediction error, plasticity gate
//!   (Schultz, Dayan & Montague, 1997).
//! - **Noradrenaline (NE)** – uncertainty, exploration, gain (Aston-Jones &
//!   Cohen, 2005; Yu & Dayan, 2005).
//! - **Serotonin (5-HT)** – patience, aversive prediction, punishment
//!   learning (Doya, 2002; Cools, Nakamura & Daw, 2011).
//! - **Acetylcholine (ACh)** – top-down attention, cortical signal-to-noise
//!   (Sarter, Hasselmo, Bruno & Givens, 2005).
//!
//! ## Kernel role
//!
//! Any call in the kernel that changes weights (i.e. learning) must present a
//! [`DopamineTag`]. `DopamineTag` has a `pub(crate)` constructor, produced
//! only inside this module by [`NeuromodulatorySystem::vta_burst`]. This
//! encodes at compile-time that STDP is dopamine-gated.

use crate::bus::{Drive, NeuralBus, Neuromodulator, NeuromodBroadcast};

/// An unforgeable dopamine authorisation to run a learning update.
#[derive(Debug, Clone, Copy)]
pub struct DopamineTag {
    da_level: f32,
}

impl DopamineTag {
    /// Current dopamine level captured by this tag (0..1 typical).
    pub fn level(&self) -> f32 { self.da_level }
}

/// The neuromodulatory system.
pub struct NeuromodulatorySystem {
    /// Current global levels.
    pub da: f32,
    /// Current global noradrenaline level.
    pub ne: f32,
    /// Current global serotonin level.
    pub ht: f32,
    /// Current global acetylcholine level.
    pub ach: f32,
    /// Exponential-average of prediction error, used as RPE baseline.
    ep_baseline: f32,
}

impl NeuromodulatorySystem {
    /// Build a fresh neuromodulatory system at physiological rest.
    pub fn new() -> Self {
        Self { da: 0.2, ne: 0.2, ht: 0.2, ach: 0.2, ep_baseline: 0.0 }
    }

    /// Compute one tick of neuromodulator dynamics from the current
    /// prediction error and the brainstem's drive alerts.
    ///
    /// This is the routine that also produces a [`DopamineTag`] — the only
    /// key that unlocks cortical plasticity.
    pub fn broadcast(&mut self, prediction_error: f32,
                     drive_alerts: &[(Drive, f32)],
                     bus: &NeuralBus, t_ms: f64) -> DopamineTag {
        // Dopamine ~ reward-prediction error = (current err - baseline).
        // Convention: falling error = "better than expected" = phasic burst.
        let rpe = self.ep_baseline - prediction_error;
        self.ep_baseline = 0.95 * self.ep_baseline + 0.05 * prediction_error;
        self.da = (0.2 + rpe.abs().tanh()).clamp(0.0, 1.0);

        // Noradrenaline scales with total drive-alert urgency and error.
        let total_alert: f32 = drive_alerts.iter().map(|(_, u)| *u).sum();
        self.ne = (0.2 + 0.4 * total_alert + 0.4 * prediction_error).clamp(0.0, 1.0);

        // Serotonin — high sleep debt / safety threats raise it.
        let sleep_urgency: f32 = drive_alerts.iter()
            .filter(|(d, _)| matches!(d, Drive::Sleep | Drive::Safety))
            .map(|(_, u)| *u).sum();
        self.ht = (0.2 + 0.6 * sleep_urgency).clamp(0.0, 1.0);

        // Acetylcholine ~ inverse of noradrenaline in this simple model.
        self.ach = (0.8 - 0.5 * self.ne).clamp(0.0, 1.0);

        // Broadcast on the bus so any subsystem can subscribe.
        for (kind, level) in [
            (Neuromodulator::Dopamine,      self.da),
            (Neuromodulator::Noradrenaline, self.ne),
            (Neuromodulator::Serotonin,     self.ht),
            (Neuromodulator::Acetylcholine, self.ach),
        ] {
            let _ = bus.neuromod_tx.try_send(NeuromodBroadcast { t_ms, kind, level });
        }

        // Mint the dopamine authorisation for this tick.
        DopamineTag { da_level: self.da }
    }

    /// Explicit phasic VTA burst — used e.g. on unexpected reward.
    /// Kept separate from `broadcast` so external reward signals (from the
    /// [`crate::syscall`] API) can also mint a tag.
    pub fn vta_burst(&mut self, magnitude: f32) -> DopamineTag {
        self.da = (self.da + magnitude).clamp(0.0, 1.0);
        DopamineTag { da_level: self.da }
    }

    /// Snapshot for logging / debugging.
    pub fn dopamine(&self) -> f32 { self.da }
    /// Noradrenaline snapshot.
    pub fn noradrenaline(&self) -> f32 { self.ne }
    /// Serotonin snapshot.
    pub fn serotonin(&self) -> f32 { self.ht }
    /// Acetylcholine snapshot.
    pub fn acetylcholine(&self) -> f32 { self.ach }
}

impl Default for NeuromodulatorySystem {
    fn default() -> Self { Self::new() }
}
