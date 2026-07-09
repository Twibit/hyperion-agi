//! # bus — typed inter-region message passing
//!
//! Biological homologue: **long-range white-matter fiber tracts** (e.g. corpus
//! callosum, internal capsule, cingulum) and diffuse neuromodulatory
//! projections. In vivo these are the only way one brain region talks to
//! another; in this kernel they are the only way one subsystem talks to
//! another.
//!
//! ## Design
//!
//! - **Typed channels.** Different message families ride on different
//!   channels, mirroring the fact that thalamocortical afferents, cortico-
//!   cortical fibers, and neuromodulatory projections use different axon
//!   classes and target different layers.
//! - **Phantom-type tokens.** Some channels only accept messages that have
//!   passed a specific gate (`Gated`, `Selected`, `Consolidated`). This
//!   makes biologically illegal shortcuts a **compile error**.
//! - **Bounded queues.** Every channel is bounded — a biological network has
//!   finite bandwidth, and unbounded queues in this crate would silently
//!   accumulate the same latency pathologies that in vivo produce hallucinated
//!   percepts and runaway loops.

use crossbeam_channel::{bounded, Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Marker: a message that has passed thalamic attentional gating.
#[derive(Debug, Clone, Copy)]
pub struct Gated;
/// Marker: a message before thalamic gating (raw sensory input).
#[derive(Debug, Clone, Copy)]
pub struct Raw;
/// Marker: an action selected by basal ganglia.
#[derive(Debug, Clone, Copy)]
pub struct Selected;
/// Marker: an action proposal not yet selected.
#[derive(Debug, Clone, Copy)]
pub struct Proposed;
/// Marker: a memory that has been replayed and consolidated.
#[derive(Debug, Clone, Copy)]
pub struct Consolidated;

/// A sensory percept flowing from receptors → thalamus → cortex.
///
/// The `S` marker enforces that only [`crate::thalamus::Thalamus`] can produce
/// `Percept<Gated>`, because the `Gated` constructor is `pub(crate)` and
/// invoked only inside the `thalamus` module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Percept<S = Raw> {
    /// Simulation time (ms) at which this percept entered the kernel.
    pub t_ms: f64,
    /// Modality: 0=vision, 1=audition, 2=somatosensation, … (extend as needed).
    pub modality: u8,
    /// Feature vector — spike rates or embeddings.
    pub features: Vec<f32>,
    /// Attentional salience (0..1) assigned by thalamus.
    pub salience: f32,
    #[serde(skip)]
    _stage: PhantomData<S>,
}

impl Percept<Raw> {
    /// Any subsystem may mint a raw percept — this is the "receptor" API.
    pub fn raw(t_ms: f64, modality: u8, features: Vec<f32>) -> Self {
        Self { t_ms, modality, features, salience: 0.0, _stage: PhantomData }
    }
}

impl Percept<Gated> {
    /// Only callable from inside `agikernel` (thalamus module).
    pub(crate) fn from_raw(raw: Percept<Raw>, salience: f32) -> Self {
        Self { t_ms: raw.t_ms, modality: raw.modality, features: raw.features,
               salience, _stage: PhantomData }
    }
}

/// An action proposed by cortex, awaiting basal-ganglia selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCommand<S = Proposed> {
    /// Time proposed.
    pub t_ms: f64,
    /// Discrete action id (index into a repertoire).
    pub action_id: u32,
    /// Continuous parameter vector (e.g. target angle, force).
    pub params: Vec<f32>,
    /// Cortical estimate of expected value (used by basal-ganglia gating).
    pub expected_value: f32,
    #[serde(skip)]
    _stage: PhantomData<S>,
}

impl ActionCommand<Proposed> {
    /// Cortex mints proposals.
    pub fn propose(t_ms: f64, action_id: u32, params: Vec<f32>, expected_value: f32) -> Self {
        Self { t_ms, action_id, params, expected_value, _stage: PhantomData }
    }
}

impl ActionCommand<Selected> {
    /// Only callable from inside `agikernel` (basal_ganglia module).
    pub(crate) fn from_proposed(p: ActionCommand<Proposed>) -> Self {
        Self { t_ms: p.t_ms, action_id: p.action_id, params: p.params,
               expected_value: p.expected_value, _stage: PhantomData }
    }
}

/// Any message that circulates on the neural bus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BusMessage {
    /// Raw sensory input from receptors → thalamus.
    Sensory(Percept<Raw>),
    /// Gated percept from thalamus → cortex.
    Gated(Percept<Gated>),
    /// Motor command emitted by basal ganglia → effectors.
    Motor(ActionCommand<Selected>),
    /// Neuromodulatory broadcast (dopamine reward-prediction error, etc.).
    Neuromod(NeuromodBroadcast),
    /// Prediction error broadcast from cortex → hippocampus / cerebellum.
    PredictionError {
        /// Time of emission (ms).
        t_ms: f64,
        /// Magnitude of the error.
        magnitude: f32,
    },
    /// Homeostatic alert from brainstem → all subsystems.
    HomeostaticAlert {
        /// Time of alert (ms).
        t_ms: f64,
        /// Which drive is out of range.
        drive: Drive,
        /// Urgency (deviation below set-point).
        urgency: f32,
    },
}

/// A neuromodulator broadcast on the bus. See [`crate::neuromod`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodBroadcast {
    /// Time of release.
    pub t_ms: f64,
    /// Which neuromodulator.
    pub kind: Neuromodulator,
    /// Concentration (arbitrary units, 0..1 typical).
    pub level: f32,
}

/// The four canonical neuromodulators handled by this kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Neuromodulator {
    /// Dopamine — reward prediction error, plasticity gate.
    Dopamine,
    /// Noradrenaline — arousal, uncertainty, gain.
    Noradrenaline,
    /// Serotonin — mood, patience, aversive prediction.
    Serotonin,
    /// Acetylcholine — attention, cortical signal-to-noise.
    Acetylcholine,
}

/// Homeostatic drives supervised by the brainstem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Drive {
    /// Energy / metabolic budget.
    Energy,
    /// Thermal balance.
    Temperature,
    /// Sleep pressure / cumulative uptime.
    Sleep,
    /// Threat / integrity.
    Safety,
}

/// The typed neural bus: a bundle of bounded channels, one per traffic class.
///
/// Cloning `NeuralBus` clones the senders/receivers cheaply — every subsystem
/// keeps its own handle so hot-path locking is avoided.
#[derive(Clone)]
pub struct NeuralBus {
    /// Raw sensory ingress (userspace → thalamus).
    pub sensory_tx: Sender<Percept<Raw>>,
    /// Raw sensory ingress reader (thalamus).
    pub sensory_rx: Receiver<Percept<Raw>>,

    /// Gated percepts (thalamus → cortex).
    pub gated_tx: Sender<Percept<Gated>>,
    /// Gated percept reader (cortex).
    pub gated_rx: Receiver<Percept<Gated>>,

    /// Motor egress (basal ganglia → effectors).
    pub motor_tx: Sender<ActionCommand<Selected>>,
    /// Motor egress reader (userspace).
    pub motor_rx: Receiver<ActionCommand<Selected>>,

    /// Neuromodulator broadcast (VTA/LC/Raphé/NBM → everyone).
    pub neuromod_tx: Sender<NeuromodBroadcast>,
    /// Neuromodulator broadcast reader.
    pub neuromod_rx: Receiver<NeuromodBroadcast>,

    /// Homeostatic alerts (brainstem → everyone).
    pub alert_tx: Sender<(Drive, f32)>,
    /// Homeostatic alerts reader.
    pub alert_rx: Receiver<(Drive, f32)>,
}

impl NeuralBus {
    /// Build a fresh bus with sensible default channel capacities.
    ///
    /// Bounded queues force back-pressure — exactly like a real brain, where a
    /// saturated pathway must drop or queue signals rather than accumulate
    /// them without bound.
    pub fn new(sensory_cap: usize, gated_cap: usize, motor_cap: usize) -> Self {
        let (sensory_tx, sensory_rx) = bounded(sensory_cap);
        let (gated_tx,   gated_rx)   = bounded(gated_cap);
        let (motor_tx,   motor_rx)   = bounded(motor_cap);
        let (neuromod_tx, neuromod_rx) = bounded(1024);
        let (alert_tx,   alert_rx)   = bounded(64);
        Self { sensory_tx, sensory_rx, gated_tx, gated_rx,
               motor_tx, motor_rx, neuromod_tx, neuromod_rx,
               alert_tx, alert_rx }
    }

    /// Deliver whatever is pending on unbuffered internal channels. This is
    /// intentionally a no-op today because all channels are already
    /// crossbeam-backed; the method exists so that future NIC-like batching
    /// can be added without changing callers.
    pub fn flush(&self) {}
}

impl Default for NeuralBus {
    fn default() -> Self { Self::new(4096, 4096, 1024) }
}
