//! Small-scale network abstraction over the [`Neuron`] trait.

use crate::compression::SparseConnectivity;
use crate::neuron::Neuron;

/// A network wires many `dyn Neuron` cells through a CSR connectivity matrix.
///
/// The current implementation uses instantaneous current-based synapses
/// (`I_syn = W · s`, where `s` is a 0/1 spike vector). More biologically
/// realistic conductance-based synapses with rise/decay kinetics are a natural
/// extension and can be added without changing this API.
pub struct Network {
    /// The neurons.
    pub neurons: Vec<Box<dyn Neuron>>,
    /// Row-i, col-j = weight from j to i (post ← pre).
    pub weights: SparseConnectivity,
    /// Scratch buffer for last-step spike vector (0.0 / 1.0).
    spike_vec: Vec<f32>,
    /// Scratch buffer for last-step synaptic current per neuron.
    syn_current: Vec<f32>,
}

impl Network {
    /// Build a network from neurons + connectivity.
    pub fn new(neurons: Vec<Box<dyn Neuron>>, weights: SparseConnectivity) -> Self {
        let n = neurons.len();
        assert_eq!(weights.rows, n);
        assert_eq!(weights.cols, n);
        Self {
            neurons,
            weights,
            spike_vec: vec![0.0; n],
            syn_current: vec![0.0; n],
        }
    }

    /// Number of neurons.
    pub fn len(&self) -> usize { self.neurons.len() }

    /// Whether the network is empty.
    pub fn is_empty(&self) -> bool { self.neurons.is_empty() }

    /// Advance every neuron by one step under the given external input vector.
    pub fn step(&mut self, dt_ms: f64, i_ext: &[f64]) -> Vec<bool> {
        assert_eq!(i_ext.len(), self.neurons.len());
        // Synaptic contribution from last step's spikes.
        self.weights.spmv(&self.spike_vec, &mut self.syn_current);
        // Step each neuron with (external + synaptic) input.
        let mut spikes = Vec::with_capacity(self.neurons.len());
        for (i, n) in self.neurons.iter_mut().enumerate() {
            let i_in = i_ext[i] + self.syn_current[i] as f64;
            let s = n.step(dt_ms, i_in);
            spikes.push(s.spiked);
        }
        // Refresh spike vector for next step.
        for (i, s) in spikes.iter().enumerate() {
            self.spike_vec[i] = if *s { 1.0 } else { 0.0 };
        }
        spikes
    }
}
