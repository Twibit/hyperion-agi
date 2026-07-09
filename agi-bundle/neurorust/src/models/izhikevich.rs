//! Izhikevich (2003) neuron.
//!
//! ```text
//! dv/dt = 0.04 v² + 5 v + 140 − u + I
//! du/dt = a (b v − u)
//! if v ≥ 30 mV: v ← c, u ← u + d
//! ```
//!
//! Four parameters `(a, b, c, d)` reproduce >20 cortical firing regimes.

use crate::compression::NeuronMemory;
use crate::neuron::{FiringPattern, Neuron, NeuronKind, NeuronState};

/// The four canonical Izhikevich parameters.
#[derive(Debug, Clone, Copy)]
pub struct IzhikevichParams {
    /// Recovery time constant.
    pub a: f64,
    /// Sensitivity of `u` to subthreshold `v`.
    pub b: f64,
    /// After-spike reset for `v` (mV).
    pub c: f64,
    /// After-spike jump for `u`.
    pub d: f64,
}

impl IzhikevichParams {
    /// Return canonical parameters for a named firing regime.
    pub fn preset(p: FiringPattern) -> Self {
        match p {
            FiringPattern::RegularSpiking       => Self { a: 0.02, b: 0.2,  c: -65.0, d: 8.0  },
            FiringPattern::IntrinsicallyBursting=> Self { a: 0.02, b: 0.2,  c: -55.0, d: 4.0  },
            FiringPattern::Chattering           => Self { a: 0.02, b: 0.2,  c: -50.0, d: 2.0  },
            FiringPattern::FastSpiking          => Self { a: 0.10, b: 0.2,  c: -65.0, d: 2.0  },
            FiringPattern::LowThresholdSpiking  => Self { a: 0.02, b: 0.25, c: -65.0, d: 2.0  },
            FiringPattern::Quiescent            => Self { a: 0.02, b: 0.2,  c: -65.0, d: 8.0  },
        }
    }
}

/// A single Izhikevich neuron.
pub struct IzhikevichNeuron {
    /// Parameters `(a, b, c, d)`.
    pub p: IzhikevichParams,
    v: f64,
    u: f64,
    t_ms: f64,
    mem: NeuronMemory,
}

impl IzhikevichNeuron {
    /// Construct with a named preset.
    pub fn preset(pattern: FiringPattern, dt_ms: f64) -> Self {
        let p = IzhikevichParams::preset(pattern);
        Self {
            v: -65.0,
            u: p.b * -65.0,
            p,
            t_ms: 0.0,
            mem: NeuronMemory::new(dt_ms, -100.0, 40.0),
        }
    }
}

impl Neuron for IzhikevichNeuron {
    fn kind(&self) -> NeuronKind { NeuronKind::Izhikevich }

    fn step(&mut self, dt_ms: f64, i_inj: f64) -> NeuronState {
        // Semi-implicit Euler as in Izhikevich (2003), split into two 0.5 ms
        // substeps for numerical stability.
        let sub = 2;
        let h = dt_ms / sub as f64;
        for _ in 0..sub {
            self.v += h * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + i_inj);
            self.u += h * (self.p.a * (self.p.b * self.v - self.u));
        }
        let spiked = self.v >= 30.0;
        if spiked {
            self.v = self.p.c;
            self.u += self.p.d;
        }
        self.t_ms += dt_ms;

        let state = NeuronState {
            t_ms: self.t_ms,
            v_mem: self.v,
            aux: vec![self.u],
            spiked,
        };
        let _ = self.mem.record(self.t_ms, self.v as f32, spiked);
        state
    }

    fn memory(&self) -> &NeuronMemory { &self.mem }
    fn memory_mut(&mut self) -> &mut NeuronMemory { &mut self.mem }
    fn v_mem(&self) -> f64 { self.v }

    fn reset(&mut self) {
        self.v = -65.0;
        self.u = self.p.b * -65.0;
        self.t_ms = 0.0;
    }
}
