//! # cerebellum — forward model & adaptive fine timing
//!
//! Biological homologue: the **cerebellum** — a stereotyped 3-layer cortex
//! (granule / Purkinje / deep nuclei) that receives efference copies of
//! motor commands and predicts their sensory consequences. Its plasticity is
//! famously driven by climbing-fiber "error" signals from the inferior olive
//! (Marr, 1969; Albus, 1971; Ito, 1984). The cerebellum is not required for
//! movement per se, but movements without it are slow, uncoordinated, and
//! miss their targets.
//!
//! ## Kernel role
//!
//! [`Cerebellum::refine`] takes a coarse action command from the basal
//! ganglia and returns a **temporally-refined** version whose parameters are
//! nudged by a linear forward model. The model is a small learned regression
//! from cortical workspace state to action-parameter correction — the same
//! math a real cerebellum implements at scale via granule → Purkinje
//! synapses.

use crate::bus::{ActionCommand, Selected};

/// A small linear forward model.
pub struct Cerebellum {
    /// Weights: correction = W · workspace + b.
    weights: Vec<Vec<f32>>,
    /// Bias term per action-parameter dimension.
    bias: Vec<f32>,
    /// Learning rate for climbing-fiber-driven updates.
    pub lr: f32,
    /// Cached last workspace input (used by `learn` after `refine`).
    last_workspace: Vec<f32>,
    /// Cached last correction applied (used by `learn`).
    last_correction: Vec<f32>,
}

impl Cerebellum {
    /// Build a cerebellum with output dimension `n_action_params` and input
    /// dimension `n_workspace_dims`.
    pub fn new(n_action_params: usize, n_workspace_dims: usize) -> Self {
        Self {
            weights: vec![vec![0.0; n_workspace_dims]; n_action_params],
            bias: vec![0.0; n_action_params],
            lr: 1e-3,
            last_workspace: vec![0.0; n_workspace_dims],
            last_correction: vec![0.0; n_action_params],
        }
    }

    /// Refine a selected action given the current cortical workspace.
    ///
    /// The refinement is in-place on the `params` vector — cerebellum does
    /// not select actions, it only sharpens them.
    pub fn refine(&mut self, action: &mut ActionCommand<Selected>, workspace: &[f32]) {
        self.last_workspace = workspace.to_vec();
        // Predict correction per param.
        for k in 0..self.weights.len() {
            let mut s = self.bias[k];
            for (j, w) in self.weights[k].iter().enumerate() {
                if j < workspace.len() { s += w * workspace[j]; }
            }
            self.last_correction.resize(self.weights.len(), 0.0);
            self.last_correction[k] = s;
            if k < action.params.len() {
                action.params[k] += s;
            }
        }
    }

    /// Climbing-fiber-driven update. The teacher signal is the observed
    /// prediction error along each action-parameter dimension.
    pub fn learn(&mut self, target_error: &[f32]) {
        // Δw_kj = -lr · error_k · workspace_j (delta rule).
        for k in 0..self.weights.len().min(target_error.len()) {
            let e = target_error[k];
            self.bias[k] -= self.lr * e;
            for j in 0..self.weights[k].len().min(self.last_workspace.len()) {
                self.weights[k][j] -= self.lr * e * self.last_workspace[j];
            }
        }
    }
}
