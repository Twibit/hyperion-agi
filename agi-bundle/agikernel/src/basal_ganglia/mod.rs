//! # basal_ganglia — action selection
//!
//! Biological homologue: the **basal ganglia** — striatum (D1 direct pathway,
//! D2 indirect pathway), GPi/SNr (tonic inhibitory output), STN
//! (hyperdirect), and SNc/VTA (dopamine).
//!
//! Anatomically the BG output nucleus (GPi/SNr) is **tonically active**,
//! keeping all downstream motor plans inhibited. Action selection is
//! implemented by *disinhibition*: cortex proposes many candidate actions,
//! the direct D1 pathway suppresses GPi for the winning candidate, and the
//! motor plan is released. Dopamine from SNc modulates D1 vs D2 balance
//! and gates learning (Redgrave, Prescott & Gurney, 1999; Frank, 2005).
//!
//! ## Kernel role
//!
//! The [`BasalGanglia::select`] method is the **only place** in this crate
//! that can produce an `ActionCommand<Selected>`. Downstream effectors accept
//! only selected commands, so cortex cannot bypass BG gating at the type
//! level.

use crate::bus::{ActionCommand, NeuralBus, Selected, Proposed};

/// The basal ganglia module.
pub struct BasalGanglia {
    /// A tonic inhibition offset — proposals must exceed this to be released.
    pub tonic_inhibition: f32,
    /// Dopamine-modulated gain on the direct pathway (D1). 1.0 = normal.
    pub d1_gain: f32,
    /// Dopamine-modulated gain on the indirect pathway (D2). 1.0 = normal.
    pub d2_gain: f32,
}

impl BasalGanglia {
    /// Build a basal-ganglia module with a mild tonic inhibition.
    pub fn new() -> Self {
        Self { tonic_inhibition: 0.2, d1_gain: 1.0, d2_gain: 1.0 }
    }

    /// Update D1/D2 gains from an ambient dopamine level (0..1).
    ///
    /// High DA → D1 gain up, D2 gain down → more "go" (exploration or
    /// exploitation of high-value proposals).
    pub fn set_dopamine(&mut self, da: f32) {
        self.d1_gain = 1.0 + da;
        self.d2_gain = 1.0 - 0.5 * da;
    }

    /// Winner-take-all action selection among cortical proposals.
    ///
    /// Returns at most one [`ActionCommand<Selected>`] and forwards it on the
    /// motor bus. The returned option is `None` when nothing exceeds the
    /// tonic-inhibition threshold — the biologically correct default is
    /// "no movement".
    pub fn select(&self, proposals: Vec<ActionCommand<Proposed>>, bus: &NeuralBus)
                  -> Option<ActionCommand<Selected>> {
        if proposals.is_empty() { return None; }
        // Score = D1(gain)·ev - D2(gain)·(inhibition).
        let mut best = None;
        let mut best_score = f32::MIN;
        for p in proposals.into_iter() {
            let score = self.d1_gain * p.expected_value
                      - self.d2_gain * self.tonic_inhibition;
            if score > best_score {
                best_score = score;
                best = Some(p);
            }
        }
        if best_score <= 0.0 { return None; } // Nothing beat tonic inhibition.
        let chosen = best.unwrap();
        let selected = ActionCommand::<Selected>::from_proposed(chosen);
        let _ = bus.motor_tx.try_send(selected.clone());
        Some(selected)
    }
}

impl Default for BasalGanglia {
    fn default() -> Self { Self::new() }
}
