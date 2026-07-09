//! Multi-compartment cable-equation neuron.
//!
//! Discretises a passive dendritic tree + active soma into `N` cylindrical
//! compartments coupled by axial conductances derived from the cable equation:
//!
//! ```text
//! C_m dV_i/dt = I_ion,i(V_i) + g_ax (V_{i-1} − 2 V_i + V_{i+1}) + I_inj,i
//! ```
//!
//! The soma compartment carries an Izhikevich-style active mechanism so the
//! whole cell still spikes; dendrites are passive RC compartments. This is a
//! reasonable stand-in for a reduced-morphology pyramidal cell.

use crate::compression::NeuronMemory;
use crate::neuron::{Neuron, NeuronKind, NeuronState};

/// Parameters of the multi-compartment model.
#[derive(Debug, Clone, Copy)]
pub struct MultiCompartmentParams {
    /// Number of compartments (index 0 = soma; rest = dendrites).
    pub n_comp: usize,
    /// Membrane capacitance per compartment (µF/cm²).
    pub c_m: f64,
    /// Passive leak conductance per compartment (mS/cm²).
    pub g_leak: f64,
    /// Leak reversal potential (mV).
    pub e_leak: f64,
    /// Axial coupling conductance between neighbouring compartments (mS/cm²).
    pub g_ax: f64,
    /// Soma Izhikevich `a`.
    pub a: f64,
    /// Soma Izhikevich `b`.
    pub b: f64,
    /// Soma Izhikevich `c` (reset).
    pub c_reset: f64,
    /// Soma Izhikevich `d` (recovery jump).
    pub d: f64,
}

impl Default for MultiCompartmentParams {
    fn default() -> Self {
        Self {
            n_comp: 5,
            c_m: 1.0,
            g_leak: 0.05,
            e_leak: -65.0,
            g_ax: 0.5,
            a: 0.02,
            b: 0.2,
            c_reset: -65.0,
            d: 8.0,
        }
    }
}

/// A multi-compartment neuron: soma with active Izhikevich dynamics and `n-1`
/// passive dendritic compartments.
pub struct MultiCompartmentNeuron {
    /// Model parameters.
    pub p: MultiCompartmentParams,
    v: Vec<f64>,   // per-compartment voltage
    u: f64,        // soma recovery
    t_ms: f64,
    mem: NeuronMemory,
}

impl MultiCompartmentNeuron {
    /// Build a resting-state multi-compartment neuron.
    pub fn new(dt_ms: f64, params: MultiCompartmentParams) -> Self {
        let v0 = params.e_leak;
        Self {
            v: vec![v0; params.n_comp],
            u: params.b * v0,
            p: params,
            t_ms: 0.0,
            mem: NeuronMemory::new(dt_ms, -100.0, 40.0),
        }
    }
}

impl Neuron for MultiCompartmentNeuron {
    fn kind(&self) -> NeuronKind { NeuronKind::MultiCompartment }

    fn step(&mut self, dt_ms: f64, i_inj: f64) -> NeuronState {
        let n = self.p.n_comp;
        let mut dv = vec![0.0; n];

        // Passive leak + axial coupling on every compartment.
        for i in 0..n {
            let leak = -self.p.g_leak * (self.v[i] - self.p.e_leak);
            let axial = if i == 0 {
                self.p.g_ax * (self.v.get(1).copied().unwrap_or(self.v[0]) - self.v[0])
            } else if i == n - 1 {
                self.p.g_ax * (self.v[i - 1] - self.v[i])
            } else {
                self.p.g_ax * (self.v[i - 1] - 2.0 * self.v[i] + self.v[i + 1])
            };
            dv[i] = (leak + axial) / self.p.c_m;
        }

        // Soma is active: Izhikevich quadratic + injected current.
        let v0 = self.v[0];
        dv[0] += 0.04 * v0 * v0 + 5.0 * v0 + 140.0 - self.u + i_inj;

        // Injection only at the distal dendrite tip in this reference impl,
        // to demonstrate propagation — comment out if you prefer soma-only.
        if n > 1 {
            dv[n - 1] += 0.05 * i_inj;
        }

        for i in 0..n {
            self.v[i] += dt_ms * dv[i];
        }
        self.u += dt_ms * self.p.a * (self.p.b * self.v[0] - self.u);

        // Somatic spike detection & reset.
        let spiked = self.v[0] >= 30.0;
        if spiked {
            self.v[0] = self.p.c_reset;
            self.u += self.p.d;
        }
        self.t_ms += dt_ms;

        let state = NeuronState {
            t_ms: self.t_ms,
            v_mem: self.v[0],
            aux: {
                let mut a = self.v.clone();
                a.push(self.u);
                a
            },
            spiked,
        };
        let _ = self.mem.record(self.t_ms, self.v[0] as f32, spiked);
        state
    }

    fn memory(&self) -> &NeuronMemory { &self.mem }
    fn memory_mut(&mut self) -> &mut NeuronMemory { &mut self.mem }
    fn v_mem(&self) -> f64 { self.v[0] }

    fn reset(&mut self) {
        let v0 = self.p.e_leak;
        for x in self.v.iter_mut() { *x = v0; }
        self.u = self.p.b * v0;
        self.t_ms = 0.0;
    }
}
