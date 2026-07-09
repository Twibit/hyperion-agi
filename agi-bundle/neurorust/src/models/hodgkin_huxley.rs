//! Hodgkin–Huxley (1952) neuron.
//!
//! Four-state ODE system:
//! ```text
//! C_m dV/dt = I_inj − ḡ_Na m³ h (V − E_Na)
//!                    − ḡ_K  n⁴   (V − E_K)
//!                    − ḡ_L        (V − E_L)
//! dx/dt = α_x(V)(1 − x) − β_x(V) x        for x ∈ {m, h, n}
//! ```
//!
//! Integrated with 4th-order Runge–Kutta at 0.01 ms; a spike is registered on
//! the up-crossing of `V = 0 mV`.

use crate::compression::NeuronMemory;
use crate::neuron::{Neuron, NeuronKind, NeuronState};

/// Standard mammalian HH parameter set (µF/cm², mS/cm², mV).
#[derive(Debug, Clone, Copy)]
pub struct HHParams {
    /// Membrane capacitance.
    pub c_m: f64,
    /// Maximum Na⁺ conductance.
    pub g_na: f64,
    /// Maximum K⁺ conductance.
    pub g_k: f64,
    /// Leak conductance.
    pub g_l: f64,
    /// Na⁺ reversal potential.
    pub e_na: f64,
    /// K⁺ reversal potential.
    pub e_k: f64,
    /// Leak reversal potential.
    pub e_l: f64,
}

impl Default for HHParams {
    fn default() -> Self {
        // Squid-giant-axon canonical values (Hodgkin & Huxley 1952),
        // temperature-scaled toward mammalian ranges is left to the caller.
        Self {
            c_m: 1.0,
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.4,
        }
    }
}

/// A single Hodgkin–Huxley neuron.
pub struct HodgkinHuxleyNeuron {
    /// Biophysical parameters.
    pub p: HHParams,
    v: f64,       // membrane potential (mV)
    m: f64,       // Na activation
    h: f64,       // Na inactivation
    n: f64,       // K  activation
    t_ms: f64,
    prev_v: f64,
    mem: NeuronMemory,
}

impl HodgkinHuxleyNeuron {
    /// Build a resting-state HH neuron with default parameters.
    pub fn new(dt_ms: f64) -> Self {
        let v0 = -65.0;
        Self {
            p: HHParams::default(),
            v: v0,
            m: alpha_m(v0) / (alpha_m(v0) + beta_m(v0)),
            h: alpha_h(v0) / (alpha_h(v0) + beta_h(v0)),
            n: alpha_n(v0) / (alpha_n(v0) + beta_n(v0)),
            t_ms: 0.0,
            prev_v: v0,
            mem: NeuronMemory::new(dt_ms, -100.0, 60.0),
        }
    }

    /// RHS of the HH ODE. Public for use by external integrators / Julia FFI.
    pub fn rhs(&self, v: f64, m: f64, h: f64, n: f64, i_inj: f64) -> (f64, f64, f64, f64) {
        let i_na = self.p.g_na * m.powi(3) * h * (v - self.p.e_na);
        let i_k  = self.p.g_k  * n.powi(4)     * (v - self.p.e_k);
        let i_l  = self.p.g_l  *                 (v - self.p.e_l);
        let dv = (i_inj - i_na - i_k - i_l) / self.p.c_m;
        let dm = alpha_m(v) * (1.0 - m) - beta_m(v) * m;
        let dh = alpha_h(v) * (1.0 - h) - beta_h(v) * h;
        let dn = alpha_n(v) * (1.0 - n) - beta_n(v) * n;
        (dv, dm, dh, dn)
    }
}

impl Neuron for HodgkinHuxleyNeuron {
    fn kind(&self) -> NeuronKind { NeuronKind::HodgkinHuxley }

    fn step(&mut self, dt_ms: f64, i_inj: f64) -> NeuronState {
        // RK4 integration.
        let (v, m, h, n) = (self.v, self.m, self.h, self.n);
        let (k1v, k1m, k1h, k1n) = self.rhs(v, m, h, n, i_inj);
        let (k2v, k2m, k2h, k2n) = self.rhs(
            v + 0.5 * dt_ms * k1v,
            m + 0.5 * dt_ms * k1m,
            h + 0.5 * dt_ms * k1h,
            n + 0.5 * dt_ms * k1n,
            i_inj,
        );
        let (k3v, k3m, k3h, k3n) = self.rhs(
            v + 0.5 * dt_ms * k2v,
            m + 0.5 * dt_ms * k2m,
            h + 0.5 * dt_ms * k2h,
            n + 0.5 * dt_ms * k2n,
            i_inj,
        );
        let (k4v, k4m, k4h, k4n) = self.rhs(
            v + dt_ms * k3v,
            m + dt_ms * k3m,
            h + dt_ms * k3h,
            n + dt_ms * k3n,
            i_inj,
        );

        self.prev_v = self.v;
        self.v += dt_ms * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0;
        self.m += dt_ms * (k1m + 2.0 * k2m + 2.0 * k3m + k4m) / 6.0;
        self.h += dt_ms * (k1h + 2.0 * k2h + 2.0 * k3h + k4h) / 6.0;
        self.n += dt_ms * (k1n + 2.0 * k2n + 2.0 * k3n + k4n) / 6.0;
        self.t_ms += dt_ms;

        // Threshold-crossing spike detection at V = 0 mV.
        let spiked = self.prev_v < 0.0 && self.v >= 0.0;
        let state = NeuronState {
            t_ms: self.t_ms,
            v_mem: self.v,
            aux: vec![self.m, self.h, self.n],
            spiked,
        };
        let _ = self.mem.record(self.t_ms, self.v as f32, spiked);
        state
    }

    fn memory(&self) -> &NeuronMemory { &self.mem }
    fn memory_mut(&mut self) -> &mut NeuronMemory { &mut self.mem }
    fn v_mem(&self) -> f64 { self.v }

    fn reset(&mut self) {
        let v0 = -65.0;
        self.v = v0;
        self.prev_v = v0;
        self.m = alpha_m(v0) / (alpha_m(v0) + beta_m(v0));
        self.h = alpha_h(v0) / (alpha_h(v0) + beta_h(v0));
        self.n = alpha_n(v0) / (alpha_n(v0) + beta_n(v0));
        self.t_ms = 0.0;
    }
}

// ---- HH rate constants (Hodgkin & Huxley 1952, in ms⁻¹) --------------------

fn alpha_m(v: f64) -> f64 {
    // Standard HH form: α_m(V) = 0.1 (V + 40) / (1 − exp(−(V + 40) / 10)).
    // Handle the removable singularity at V = −40 mV with an L'Hôpital limit.
    let x = -(v + 40.0) / 10.0;
    if x.abs() < 1e-6 {
        1.0
    } else {
        0.1 * (v + 40.0) / (1.0 - x.exp())
    }
}

fn beta_m(v: f64)  -> f64 { 4.0 * (-(v + 65.0) / 18.0).exp() }
fn alpha_h(v: f64) -> f64 { 0.07 * (-(v + 65.0) / 20.0).exp() }
fn beta_h(v: f64)  -> f64 { 1.0 / (((-(v + 35.0)) / 10.0).exp() + 1.0) }

fn alpha_n(v: f64) -> f64 {
    let denom = 1.0 - ((-(v + 55.0)) / 10.0).exp();
    if denom.abs() < 1e-9 { 0.1 } else { 0.01 * (v + 55.0) / denom }
}
fn beta_n(v: f64)  -> f64 { 0.125 * (-(v + 65.0) / 80.0).exp() }
