//! Core `Neuron` trait and shared state types.

use serde::{Deserialize, Serialize};

use crate::compression::NeuronMemory;

/// Which biophysical family of neuron model to instantiate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronKind {
    /// Full ion-channel Hodgkin–Huxley (Na⁺, K⁺, leak).
    HodgkinHuxley,
    /// 2-variable Izhikevich model.
    Izhikevich,
    /// Multi-compartment cable-equation model.
    MultiCompartment,
}

/// Coarse label for the observed firing regime — used by tests and
/// classification code. Names follow Izhikevich (2003).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FiringPattern {
    /// Regular spiking.
    RegularSpiking,
    /// Intrinsically bursting.
    IntrinsicallyBursting,
    /// Chattering.
    Chattering,
    /// Fast spiking.
    FastSpiking,
    /// Low-threshold spiking.
    LowThresholdSpiking,
    /// Quiescent (subthreshold only).
    Quiescent,
}

/// A snapshot of a neuron's dynamical state at one instant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronState {
    /// Simulation time (ms) at which this snapshot was taken.
    pub t_ms: f64,
    /// Membrane potential of the soma (mV).
    pub v_mem: f64,
    /// Free-form auxiliary variables (gating vars, recovery, compartments …).
    pub aux: Vec<f64>,
    /// Whether a spike was emitted during the step that produced this snapshot.
    pub spiked: bool,
}

impl NeuronState {
    /// Convenience constructor for a resting-state snapshot.
    pub fn resting(v_rest_mv: f64, aux_len: usize) -> Self {
        Self {
            t_ms: 0.0,
            v_mem: v_rest_mv,
            aux: vec![0.0; aux_len],
            spiked: false,
        }
    }
}

/// Uniform interface every neuron model must implement.
///
/// Implementations own their own compression-backed memory so that long
/// simulations do not blow up RAM. `step` advances the state by `dt` ms under
/// injected current `i_inj` (µA/cm²) and appends the resulting state to the
/// internal [`NeuronMemory`].
pub trait Neuron: Send {
    /// Which family this implementer belongs to.
    fn kind(&self) -> NeuronKind;

    /// Advance the simulation by one time step.
    fn step(&mut self, dt_ms: f64, i_inj: f64) -> NeuronState;

    /// Read-only handle to the compressed memory arena.
    fn memory(&self) -> &NeuronMemory;

    /// Mutable handle (mostly for tests and network wiring).
    fn memory_mut(&mut self) -> &mut NeuronMemory;

    /// Current membrane potential (mV) without stepping.
    fn v_mem(&self) -> f64;

    /// Reset to resting state, keeping model parameters.
    fn reset(&mut self);
}
