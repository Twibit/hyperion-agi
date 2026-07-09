//! # neurorust
//!
//! Biologically accurate human-neuron simulation in Rust.
//!
//! ## Design pillars
//!
//! 1. **Biological fidelity via a trait-based model zoo.** The [`Neuron`] trait
//!    unifies three canonical models used in computational neuroscience:
//!    - Hodgkin–Huxley (ion-channel-level Na⁺/K⁺/leak dynamics, gold standard),
//!    - Izhikevich (2-ODE quadratic model reproducing 20+ firing regimes),
//!    - Multi-compartment cable equation (soma + dendrites + axon, spatial).
//!    Client code selects a model via [`NeuronKind`] or by writing generic
//!    code against the trait.
//!
//! 2. **Hybrid per-datum compression.** Each neuron owns a
//!    [`compression::NeuronMemory`] arena that routes each state stream to the
//!    codec that fits it best:
//!    - Spike trains  → delta-encoded inter-spike intervals + zstd (lossless),
//!    - Membrane traces V(t) → f32 → i16 affine quantisation (lossy but tight),
//!    - Connectivity matrices → sparse CSR,
//!    - Population raster plots → bit-packed via `bitvec`.
//!
//! 3. **Heavy math lives in Julia.** Rust owns simulation control, memory
//!    layout, and safety. Matrix ops, ODE Jacobians, cable-equation tridiagonal
//!    solves, and spectral analyses are shipped over the FFI in
//!    [`ffi::julia`], behind the `julia` cargo feature. When Julia is not
//!    available, [`ffi::pure_rust`] provides equivalent fallback kernels so
//!    everything still compiles and tests pass.
//!
//! See `README.md` for the full architectural narrative and validation report.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod asm_kernels;
pub mod asm_kernels_avx2;
pub mod asm_kernels_avx512;
pub mod asm_kernels_neon;
pub mod compression;
pub mod ffi;
pub mod models;
pub mod network;
pub mod neuron;
pub mod three_factor_stdp;
pub mod wta_kernel;

pub use crate::neuron::{FiringPattern, Neuron, NeuronKind, NeuronState};
pub use crate::network::Network;

/// Physical / numerical constants shared across models.
pub mod constants {
    /// Default integration step (ms). Small enough for stiff HH dynamics.
    pub const DT_MS: f64 = 0.01;
    /// Faraday constant (C/mol) — used by future extensions with Nernst terms.
    pub const F_CONST: f64 = 96_485.332_12;
    /// Gas constant (J/(mol·K)).
    pub const R_CONST: f64 = 8.314_462_618;
    /// Physiological temperature (K) — 37 °C.
    pub const T_HUMAN_K: f64 = 310.15;
}
