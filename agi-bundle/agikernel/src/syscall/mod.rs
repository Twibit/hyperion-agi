//! # syscall — prefrontal-cortex-like external API
//!
//! Biological homologue: the **prefrontal cortex (PFC)** — the "userspace
//! interface" of the brain. In humans, the PFC is what allows verbal
//! instructions, goals, and abstract rules to influence otherwise-automatic
//! behaviour (Miller & Cohen, 2001).
//!
//! ## Kernel role
//!
//! `Syscall` is the ONLY struct that userspace code should hold. It exposes
//! four operations, all of which route through the internal subsystems while
//! respecting biological invariants:
//!
//! 1. `sense(...)` – inject a raw percept into the sensory bus.
//! 2. `poll_action()` – pop a selected motor command (if any).
//! 3. `reward(magnitude)` – deliver an unexpected reward → VTA burst.
//! 4. `refuel(drive, amount)` – restore a homeostatic drive (e.g. "feed").

use crate::basal_ganglia::BasalGanglia;
use crate::brainstem::Brainstem;
use crate::bus::{ActionCommand, Drive, NeuralBus, Percept, Selected};
use crate::cerebellum::Cerebellum;
use crate::cortex::Cortex;
use crate::hippocampus::Hippocampus;
use crate::neuromod::NeuromodulatorySystem;
use crate::scheduler::Scheduler;
use crate::thalamus::Thalamus;

/// The full kernel, wired up.
pub struct Syscall {
    /// Sensory-gating relay.
    pub thalamus: Thalamus,
    /// Cortex hierarchy.
    pub cortex: Cortex,
    /// Episodic memory.
    pub hippocampus: Hippocampus,
    /// Action selector.
    pub bg: BasalGanglia,
    /// Fine-motor refiner.
    pub cerebellum: Cerebellum,
    /// Homeostat.
    pub brainstem: Brainstem,
    /// Neuromodulatory broadcaster.
    pub neuromod: NeuromodulatorySystem,
    /// Message bus.
    pub bus: NeuralBus,
    /// Global oscillatory scheduler.
    pub scheduler: Scheduler,
    /// Last selected action, cached for `poll_action`.
    last_action: Option<ActionCommand<Selected>>,
}

impl Syscall {
    /// Build a mini-brain with the given dimensions. Reasonable defaults:
    /// depth=3, n_per_layer=32, n_channels=8, top_k=4, dt_ms=25.0.
    pub fn new(depth: usize, n_per_layer: usize, n_channels: usize,
               top_k: usize, dt_ms: f64) -> Self {
        Self {
            thalamus: Thalamus::new(n_channels, top_k, dt_ms),
            cortex: Cortex::new(depth, n_per_layer, dt_ms),
            hippocampus: Hippocampus::new(4096, n_per_layer, 128, 0xC0FFEE),
            bg: BasalGanglia::new(),
            cerebellum: Cerebellum::new(1, n_per_layer),
            brainstem: Brainstem::new(),
            neuromod: NeuromodulatorySystem::new(),
            bus: NeuralBus::default(),
            scheduler: Scheduler::new(dt_ms),
            last_action: None,
        }
    }

    /// Userspace → kernel: submit a raw sensory percept.
    pub fn sense(&self, modality: u8, features: Vec<f32>) {
        let t = self.scheduler.time_ms();
        let _ = self.bus.sensory_tx.try_send(Percept::raw(t, modality, features));
    }

    /// Userspace ← kernel: fetch the most recent selected action, if any.
    pub fn poll_action(&mut self) -> Option<ActionCommand<Selected>> {
        self.last_action.take()
    }

    /// Deliver an unexpected reward — triggers VTA phasic dopamine burst,
    /// consolidates the last few episodes, and returns a `DopamineTag`
    /// snapshot for logging.
    pub fn reward(&mut self, magnitude: f32) -> f32 {
        let tag = self.neuromod.vta_burst(magnitude);
        self.hippocampus.tag_recent_reward(magnitude, 16);
        tag.level()
    }

    /// Refuel a homeostatic drive.
    pub fn refuel(&mut self, drive: Drive, amount: f32) {
        self.brainstem.refuel(drive, amount);
    }

    /// Run one full kernel tick — the "master loop" from the README. Returns
    /// (mean_prediction_error, emitted_action_id_or_none).
    pub fn tick(&mut self) -> (f32, Option<u32>) {
        // 1. Homeostasis first — brainstem always wins.
        let alerts_before = self.brainstem.tick(&self.bus);
        let throttle = self.brainstem.throttle_cortex;

        // Collect drive alerts for neuromod.
        let mut drive_alerts: Vec<(Drive, f32)> = Vec::new();
        while let Ok(a) = self.bus.alert_rx.try_recv() { drive_alerts.push(a); }

        // 2. Thalamic gating.
        let _ = self.thalamus.gate(&self.bus);

        // 3. Cortex tick (unless throttled by brainstem).
        let (workspace, err, gated_percepts) = if throttle {
            (self.cortex.workspace.clone(), 1.0, Vec::new())
        } else {
            self.cortex.tick(&self.bus)
        };

        // 4. Cortex proposes actions; basal ganglia selects one.
        let proposals = self.cortex.propose_actions(self.scheduler.time_ms());
        let da_for_bg = self.neuromod.dopamine();
        self.bg.set_dopamine(da_for_bg);
        let selected = self.bg.select(proposals, &self.bus);

        // 5. Cerebellum refines timing & parameters.
        let mut refined = selected;
        if let Some(a) = refined.as_mut() {
            self.cerebellum.refine(a, &workspace);
        }
        let action_id = refined.as_ref().map(|a| a.action_id);
        self.last_action = refined;

        // 6. Hippocampus binds each gated percept from this tick ↔ the
        //    latest selected action. Cortex handed us the consumed percepts
        //    directly to avoid racing on the bus channel.
        for p in &gated_percepts {
            self.hippocampus.bind(p, self.last_action.as_ref());
        }

        // 7. Neuromod broadcast — this is what mints the DopamineTag used
        //    for cortical consolidation this tick.
        let tag = self.neuromod.broadcast(err, &drive_alerts, &self.bus,
                                          self.scheduler.time_ms());

        // 8. Offline replay + consolidation at theta troughs (~5 Hz).
        if self.scheduler.is_theta_trough() && !self.hippocampus.is_empty() {
            let batch = self.hippocampus.replay(8);
            self.cortex.consolidate(batch, &tag);
        }

        // 9. Close the cortico-thalamic loop.
        let bias = self.cortex.attn_bias(self.thalamus.n_channels());
        self.thalamus.set_attn_bias(bias);

        // 10. Advance clock.
        self.bus.flush();
        self.scheduler.advance();

        let _ = alerts_before; // silence unused-var warning
        (err, action_id)
    }

    /// Simulation time (ms) since kernel start.
    pub fn t_ms(&self) -> f64 { self.scheduler.time_ms() }
}
