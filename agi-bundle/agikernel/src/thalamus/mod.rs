//! # thalamus — sensory gating and attentional relay
//!
//! Biological homologue: the **thalamus** (LGN, MGN, VPL/VPM, pulvinar, MD)
//! and the **thalamic reticular nucleus (TRN)**.
//!
//! Every sensory pathway (vision, audition, somatosensation …) passes
//! through the thalamus before reaching cortex. The TRN forms an inhibitory
//! shell around the thalamus that **selectively suppresses** channels that
//! are not attended, implementing a biological "attentional gate". Cortex
//! reciprocally projects back to the thalamus, which is why thalamo-cortical
//! loops are the substrate for top-down attention (Sherman & Guillery, 2006;
//! Halassa & Kastner, 2017).
//!
//! ## Kernel role
//!
//! - Consume [`Percept<Raw>`] messages from `bus.sensory_rx`.
//! - Compute per-channel salience using fast-spiking Izhikevich cells
//!   (a well-known caricature of TRN interneurons).
//! - Emit [`Percept<Gated>`] on `bus.gated_tx`, but **only** for the top-K
//!   most salient channels each tick.
//!
//! The [`Gated`] marker constructor on [`Percept<Gated>`] is `pub(crate)`, so
//! no code outside `agikernel` can synthesise a gated percept: cortex is
//! guaranteed at compile time to only receive thalamically-vetted input.

use crate::bus::{Gated, NeuralBus, Percept};
use neurorust::models::izhikevich::IzhikevichNeuron;
use neurorust::neuron::FiringPattern;
use neurorust::Neuron;

/// The thalamus module.
pub struct Thalamus {
    /// One "relay + TRN" cell per sensory channel.
    channels: Vec<Box<dyn Neuron>>,
    /// Top-down attentional bias vector (length = n_channels). Values ≥ 0.
    attn_bias: Vec<f32>,
    /// How many channels are allowed through per tick (attentional spotlight).
    top_k: usize,
    /// Integration step (ms).
    dt_ms: f64,
}

impl Thalamus {
    /// Build a thalamus with `n_channels` sensory relay cells.
    pub fn new(n_channels: usize, top_k: usize, dt_ms: f64) -> Self {
        let channels: Vec<Box<dyn Neuron>> = (0..n_channels)
            .map(|_| Box::new(IzhikevichNeuron::preset(FiringPattern::FastSpiking, dt_ms))
                       as Box<dyn Neuron>)
            .collect();
        Self { channels, attn_bias: vec![1.0; n_channels], top_k, dt_ms }
    }

    /// Number of sensory channels this thalamus relays.
    pub fn n_channels(&self) -> usize { self.channels.len() }

    /// Update the top-down attentional bias (set by cortex L6 each tick).
    pub fn set_attn_bias(&mut self, bias: Vec<f32>) {
        assert_eq!(bias.len(), self.attn_bias.len(),
                   "attentional bias must match thalamic channel count");
        self.attn_bias = bias;
    }

    /// Gate one tick's worth of raw percepts, forwarding gated ones on the
    /// bus. Returns the number of gated percepts emitted.
    pub fn gate(&mut self, bus: &NeuralBus) -> usize {
        let mut raws: Vec<Percept> = Vec::new();
        while let Ok(p) = bus.sensory_rx.try_recv() {
            raws.push(p);
        }
        if raws.is_empty() { return 0; }

        // Score each raw percept via its channel's relay cell.
        let mut scored: Vec<(f32, Percept)> = Vec::with_capacity(raws.len());
        for p in raws {
            let ch = (p.modality as usize) % self.channels.len();
            let drive: f64 = p.features.iter().map(|f| f.abs() as f64).sum::<f64>()
                / (p.features.len().max(1) as f64);
            let scaled = drive * (self.attn_bias[ch] as f64) * 5.0;
            let state = self.channels[ch].step(self.dt_ms, scaled);
            // Salience proxy: normalised depolarisation above -60 mV.
            let sal = ((state.v_mem + 60.0) / 30.0).clamp(0.0, 1.0) as f32;
            scored.push((sal, p));
        }

        // Keep only the top-K most salient this tick.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.truncate(self.top_k);

        let mut emitted = 0;
        for (sal, raw) in scored {
            let gated: Percept<Gated> = Percept::from_raw(raw, sal);
            if bus.gated_tx.try_send(gated).is_ok() {
                emitted += 1;
            }
        }
        emitted
    }
}
