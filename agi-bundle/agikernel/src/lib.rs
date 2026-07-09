//! # agikernel
//!
//! A biologically-grounded kernel/hardware design concept for a theoretical
//! AGI, layered on the [`neurorust`] spiking-neuron substrate.
//!
//! ## Overview
//!
//! `agikernel` treats the mammalian brain as a **multi-computer operating
//! system**: the neocortex, thalamus, hippocampus, basal ganglia, cerebellum,
//! brainstem and neuromodulatory nuclei are modelled as **kernel subsystems**
//! that communicate over a **typed message bus** under a shared **oscillatory
//! scheduler**.
//!
//! Every subsystem consists of populations of [`neurorust::Neuron`] trait
//! objects — biologically accurate Hodgkin–Huxley, Izhikevich, or
//! multi-compartment cells — so the design is grounded in real biophysics
//! rather than analogy.
//!
//! ## Module map
//!
//! - [`cortex`]      – 6-layer canonical microcircuit, hierarchical columns,
//!                     predictive coding.
//! - [`thalamus`]    – Sensory gating, attention (TRN), cortico-cortical relay.
//! - [`hippocampus`] – Episodic binding, pattern-separation (DG),
//!                     pattern-completion (CA3), replay to cortex.
//! - [`basal_ganglia`] – Action selection via direct/indirect pathways.
//! - [`cerebellum`]  – Forward models, adaptive fine timing.
//! - [`brainstem`]   – Homeostasis, drives, arousal, safety supervisor.
//! - [`neuromod`]    – Dopamine, noradrenaline, serotonin, acetylcholine
//!                     broadcast (global gain and plasticity gating).
//! - [`bus`]         – Typed inter-region message passing (white matter).
//! - [`scheduler`]   – Global oscillatory time base (θ, α, β, γ).
//! - [`syscall`]     – Prefrontal-cortex-like external API for "userspace".
//!
//! ## Biological invariants enforced at compile time
//!
//! See `README.md` §3. In short, the type system encodes:
//!
//! 1. Sensory data cannot reach cortex without passing thalamic gating.
//! 2. Cortical consolidation requires a hippocampal replay batch.
//! 3. Motor output requires basal-ganglia selection.
//! 4. Plasticity requires a neuromodulator token.
//! 5. Homeostasis has hard priority over cognition.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod basal_ganglia;
pub mod brainstem;
pub mod bus;
pub mod cerebellum;
pub mod cortex;
pub mod hippocampus;
pub mod neuromod;
pub mod scheduler;
pub mod syscall;
pub mod thalamus;

// Convenient re-exports.
pub use crate::brainstem::Brainstem;
pub use crate::bus::NeuralBus;
pub use crate::cerebellum::Cerebellum;
pub use crate::cortex::{Cortex, CorticalColumn};
pub use crate::hippocampus::Hippocampus;
pub use crate::basal_ganglia::BasalGanglia;
pub use crate::neuromod::NeuromodulatorySystem;
pub use crate::scheduler::{Rhythm, Scheduler};
pub use crate::syscall::Syscall;
pub use crate::thalamus::Thalamus;

/// Wall-clock length of one "gamma cycle" tick — the finest kernel quantum.
/// 25 ms mirrors the ~40 Hz gamma band that binds cortical activity in vivo.
pub const GAMMA_TICK_MS: f64 = 25.0;

/// Compile-time version tag.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Convenience type alias: a boxed biological neuron.
pub type BoxedNeuron = Box<dyn neurorust::Neuron>;
