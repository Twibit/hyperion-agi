//! Three-factor STDP — Hebbian × neuromodulator gated plasticity.
//!
//! ## Why this module exists (v33)
//!
//! Vanilla STDP (spike-timing-dependent plasticity) implements the
//! Hebbian rule:
//!
//!   Δw_ij ∝ + η_LTP · pre_i · post_j   if post fires shortly after pre
//!   Δw_ij ∝ − η_LTD · pre_i · post_j   if pre fires shortly after post
//!
//! In vivo, cortical STDP is not this pure. Every measurement since the
//! late-2000s (Pawlak & Kerr 2008; Izhikevich 2007; Frémaux & Gerstner
//! 2016) shows that the weight update is *also* gated by a global
//! neuromodulator signal — dopamine for reward, ACh for attention,
//! noradrenaline for surprise. The rule becomes:
//!
//!   Δw_ij = η · e_ij(t) · M(t)
//!
//! where `e_ij(t)` is an *eligibility trace* (the accumulated Hebbian
//! coincidence signal), and `M(t)` is the neuromodulator level, which
//! can be positive (dopamine reward) or negative (aversive signal).
//!
//! ## Why this matters for learning
//!
//! Two-factor STDP is unsupervised — it does whatever the input statistics
//! push it toward. Combined with WTA (see `wta_kernel.rs`) it does
//! pattern separation. But it *cannot* solve a task that requires
//! "learn to fire pattern-A on stimulus-1 and pattern-B on stimulus-2
//! based on external reward" — because the weight update has no way to
//! know which was rewarded.
//!
//! Three-factor STDP solves that. The eligibility trace decays with a
//! biological ~1 s time constant, so a reward delivered up to a second
//! after the causal spike pair still gates the correct synapses.
//!
//! ## Honest scope
//!
//! Three-factor STDP is well-established biology and well-established
//! computation — see Frémaux & Gerstner 2016 for a full review. It gives
//! you reward-modulated Hebbian learning. It does NOT give you deep
//! reasoning, transfer learning, or a coherent conversation. Those
//! require architectural scale (mm² of cortex, billions of synapses),
//! curricula, and structure that this module does not attempt.

#![allow(clippy::too_many_arguments)]

/// State for a three-factor STDP synapse population.
///
/// One instance is created per connectivity block (e.g. layer-4 →
/// layer-2/3, or input → cortical column). The eligibility trace `e_ij`
/// is stored densely as f32 — this is memory-cheap next to weights.
pub struct ThreeFactorStdp {
    /// Number of presynaptic neurons.
    pub n_pre: usize,
    /// Number of postsynaptic neurons.
    pub n_post: usize,

    /// Eligibility trace, shape `(n_post, n_pre)`, row-major.
    /// Decays exponentially with time constant `tau_e`.
    pub eligibility: Vec<f32>,

    /// Time constant of the eligibility trace, in simulation ticks.
    /// Biological cortex: 0.5–2 s ~ 500–2000 ticks at 1 kHz.
    pub tau_e: f32,

    /// Decay factor per tick, precomputed = exp(-1 / tau_e).
    pub decay: f32,

    /// LTP gain per coincident (pre_active, post_fired) pair, applied to
    /// the *eligibility trace* not the weight. Weight updates happen only
    /// when a neuromodulator signal arrives.
    pub eta_ltp: f32,

    /// LTD gain per (pre_active, post_silent) pair. Negative sign is
    /// applied by [`accumulate_eligibility`], so pass a positive number.
    pub eta_ltd: f32,

    /// Learning rate for the actual weight update: Δw = lr · e · M.
    pub lr: f32,
}

impl ThreeFactorStdp {
    /// Create a fresh eligibility trace bank for `n_post × n_pre` synapses.
    pub fn new(n_pre: usize, n_post: usize, tau_e_ticks: f32) -> Self {
        let decay = (-1.0_f32 / tau_e_ticks.max(1e-3)).exp();
        Self {
            n_pre,
            n_post,
            eligibility: vec![0.0; n_pre * n_post],
            tau_e: tau_e_ticks,
            decay,
            eta_ltp: 0.02,
            eta_ltd: 0.005,
            lr: 0.5,
        }
    }

    /// Advance the eligibility trace one tick — pure decay, no input.
    /// This is called every tick, even when the network is idle, to model
    /// the biological decay of the calcium/CaMKII "tag".
    #[inline]
    pub fn decay_step(&mut self) {
        // Use the SAXPY kernel from AVX-512 when available: e *= decay
        // is expressible as e += (decay - 1) * e, but a straight scalar
        // multiply is actually faster because we avoid a load of e.
        // Leave it as a plain multiply and trust the compiler to
        // vectorise the tight loop.
        for v in self.eligibility.iter_mut() {
            *v *= self.decay;
        }
    }

    /// Accumulate Hebbian coincidences into the eligibility trace.
    ///
    /// - `pre_spikes[i]` is 1 if presynaptic neuron `i` fired this tick.
    /// - `post_spikes[j]` is 1 if postsynaptic neuron `j` fired this tick.
    ///
    /// For every (pre fired, post fired) pair we add +η_LTP.
    /// For every (pre fired, post silent) pair we add −η_LTD (LTD).
    ///
    /// This is the Hebb rule as applied to the *trace*, not the weight.
    /// The weight update comes later, when `apply_neuromodulator` runs.
    #[inline]
    pub fn accumulate_eligibility(&mut self, pre_spikes: &[u8], post_spikes: &[u8]) {
        assert_eq!(pre_spikes.len(), self.n_pre);
        assert_eq!(post_spikes.len(), self.n_post);
        for j in 0..self.n_post {
            let post_fired = post_spikes[j] > 0;
            let row_off = j * self.n_pre;
            if post_fired {
                // LTP row.
                for i in 0..self.n_pre {
                    if pre_spikes[i] > 0 {
                        self.eligibility[row_off + i] += self.eta_ltp;
                    }
                }
            } else {
                // LTD row: only synapses whose pre fired but post didn't
                // are candidates for LTD (Bienenstock-Cooper-Munro flavour).
                for i in 0..self.n_pre {
                    if pre_spikes[i] > 0 {
                        self.eligibility[row_off + i] -= self.eta_ltd;
                    }
                }
            }
        }
    }

    /// Apply a global neuromodulator signal to convert eligibility into
    /// weight change. `weights` is uint8 memristor-style storage, i.e.
    /// the same format used by `MemristorLayer` in `embodied_agent`.
    ///
    /// Positive `modulator` (dopamine/reward) commits LTP.
    /// Negative `modulator` (aversive) commits LTD.
    /// Zero modulator = no weight change even if eligibility is high.
    ///
    /// This is the whole point of three-factor STDP: the weight only
    /// changes when the *global* reward signal arrives, but the
    /// eligibility trace already knows *which* synapses were causally
    /// involved thanks to its accumulation phase.
    pub fn apply_neuromodulator(&mut self, weights: &mut [u8], modulator: f32) {
        assert_eq!(weights.len(), self.n_pre * self.n_post);
        let scale = self.lr * modulator;
        if scale.abs() < 1e-6 {
            return;
        }
        // Δw (float) = scale * e; then quantise into the u8 memristor range.
        for (w, &e) in weights.iter_mut().zip(self.eligibility.iter()) {
            let delta = scale * e * 255.0; // uint8 range
            let new_w = (*w as f32) + delta;
            *w = new_w.clamp(0.0, 255.0) as u8;
        }
    }

    /// Convenience: full step. Called by the SNN driver every tick.
    ///
    /// 1. Decay eligibility.
    /// 2. Accumulate coincidences from this tick's spikes.
    /// 3. If a neuromodulator arrived, commit eligibility → weights.
    pub fn tick(
        &mut self,
        pre_spikes: &[u8],
        post_spikes: &[u8],
        weights: &mut [u8],
        modulator: f32,
    ) {
        self.decay_step();
        self.accumulate_eligibility(pre_spikes, post_spikes);
        if modulator.abs() > 1e-6 {
            self.apply_neuromodulator(weights, modulator);
        }
    }
}

// ---------------------------------------------------------------------------
// FFI shim — lets the Python `embodied_agent` layer call these directly.
// See `src/ffi/mod.rs` for wiring.
// ---------------------------------------------------------------------------

/// Opaque handle for the Python side.
#[repr(C)]
pub struct StdpHandle {
    inner: ThreeFactorStdp,
}

/// # Safety
/// Caller must pass valid pointers and matching sizes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stdp_new(n_pre: usize, n_post: usize, tau_e: f32) -> *mut StdpHandle {
    let s = Box::new(StdpHandle {
        inner: ThreeFactorStdp::new(n_pre, n_post, tau_e),
    });
    Box::into_raw(s)
}

/// # Safety
/// Caller must have obtained `h` from [`stdp_new`] and not have already freed it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stdp_free(h: *mut StdpHandle) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// # Safety
/// - `h` valid non-null handle
/// - `pre_spikes` / `post_spikes` / `weights` point to correctly-sized buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stdp_tick(
    h: *mut StdpHandle,
    pre_spikes: *const u8,
    n_pre: usize,
    post_spikes: *const u8,
    n_post: usize,
    weights: *mut u8,
    modulator: f32,
) {
    let handle = &mut *h;
    let pre = core::slice::from_raw_parts(pre_spikes, n_pre);
    let post = core::slice::from_raw_parts(post_spikes, n_post);
    let w = core::slice::from_raw_parts_mut(weights, n_pre * n_post);
    handle.inner.tick(pre, post, w, modulator);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eligibility_decays_toward_zero() {
        let mut s = ThreeFactorStdp::new(2, 2, tau_e_ticks_1000_ish());
        s.eligibility.fill(1.0);
        for _ in 0..10_000 {
            s.decay_step();
        }
        assert!(s.eligibility.iter().all(|&v| v.abs() < 1e-3));
    }
    fn tau_e_ticks_1000_ish() -> f32 { 1000.0 }

    #[test]
    fn coincident_pre_post_grows_eligibility() {
        let mut s = ThreeFactorStdp::new(3, 2, 1000.0);
        let pre  = [1u8, 0, 1];
        let post = [1u8, 0];
        s.accumulate_eligibility(&pre, &post);
        // post neuron 0 fired, so synapses (post=0, pre=0) and (post=0, pre=2)
        // should get +eta_ltp; (post=0, pre=1) should be untouched.
        assert!((s.eligibility[0] - s.eta_ltp).abs() < 1e-6);
        assert!(s.eligibility[1].abs() < 1e-6);
        assert!((s.eligibility[2] - s.eta_ltp).abs() < 1e-6);
        // post neuron 1 did NOT fire: pre-fired synapses should get −eta_ltd.
        assert!((s.eligibility[3] + s.eta_ltd).abs() < 1e-6);
        assert!(s.eligibility[4].abs() < 1e-6);
        assert!((s.eligibility[5] + s.eta_ltd).abs() < 1e-6);
    }

    #[test]
    fn reward_commits_eligibility_to_weights() {
        let mut s = ThreeFactorStdp::new(2, 2, 1000.0);
        s.eligibility = vec![0.5, -0.3, 0.0, 0.1];
        let mut w: Vec<u8> = vec![128, 128, 128, 128];
        s.apply_neuromodulator(&mut w, 1.0);
        // Positive modulator + positive eligibility → weight grows.
        assert!(w[0] > 128);
        // Positive modulator + negative eligibility → weight shrinks.
        assert!(w[1] < 128);
        assert_eq!(w[2], 128);
        assert!(w[3] > 128);
    }

    #[test]
    fn zero_modulator_is_noop() {
        let mut s = ThreeFactorStdp::new(2, 2, 1000.0);
        s.eligibility = vec![0.5, -0.3, 0.0, 0.1];
        let mut w: Vec<u8> = vec![128, 128, 128, 128];
        s.apply_neuromodulator(&mut w, 0.0);
        assert_eq!(w, vec![128, 128, 128, 128]);
    }
}
