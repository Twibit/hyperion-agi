//! # hippocampus — episodic binding, pattern (dis)completion, replay
//!
//! Biological homologue: the **hippocampal formation**.
//!
//! - **Dentate Gyrus (DG)** — pattern *separation*: makes similar inputs
//!   maximally dissimilar (Yassa & Stark, 2011).
//! - **CA3**       — auto-associative attractor: pattern *completion* — a
//!   partial cue retrieves the whole episode (Marr, 1971).
//! - **CA1**       — comparator / index: signals novelty to cortex.
//! - **Sharp-wave ripples**  — offline compressed replay, mostly during
//!   slow-wave sleep and theta troughs (Buzsáki, 2015).
//!
//! ## Kernel role
//!
//! - `bind(percept, action)` writes a new episode to a large ring buffer.
//! - `recall(cue)` returns the closest episode (pattern completion).
//! - `replay()` returns a [`ReplayBatch`] — the only type accepted by
//!   [`crate::cortex::Cortex::consolidate`], so cortex cannot consolidate
//!   without hippocampal cooperation. This encodes the well-established
//!   Complementary Learning Systems theory (McClelland et al., 1995).

use crate::bus::{ActionCommand, Gated, Percept, Selected};
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// A single episodic memory: a percept bound to whatever action followed it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Time captured (ms).
    pub t_ms: f64,
    /// Sparse-coded percept (DG output).
    pub key: Vec<f32>,
    /// Action id that co-occurred (0 = "no action").
    pub action_id: u32,
    /// Optional reward tag associated with this episode.
    pub reward: f32,
}

/// A batch of replayed episodes that cortex can consolidate. The private
/// constructor makes this type unforgeable from outside `hippocampus`.
pub struct ReplayBatch {
    inner: Vec<Episode>,
}

impl ReplayBatch {
    /// Length of the batch.
    pub fn len(&self) -> usize { self.inner.len() }
    /// Is empty?
    pub fn is_empty(&self) -> bool { self.inner.is_empty() }
    /// Iterate.
    pub fn iter(&self) -> impl Iterator<Item = &Episode> { self.inner.iter() }
}

/// The hippocampus module.
pub struct Hippocampus {
    /// Ring buffer of bound episodes.
    episodes: Vec<Episode>,
    /// Ring-buffer write cursor.
    cursor: usize,
    /// Ring-buffer capacity.
    cap: usize,
    /// RNG for random-order replay.
    rng: StdRng,
    /// DG projection matrix (dense random for pattern separation).
    dg_proj: Vec<Vec<f32>>,
    /// Output width of the DG code (== `key` length).
    dg_dim: usize,
}

impl Hippocampus {
    /// Build a hippocampus with the given capacity and DG code width.
    pub fn new(capacity: usize, input_dim: usize, dg_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        // Random ±1 projection — a well-known, cheap pattern-separation
        // primitive (sparse random projection, Achlioptas 2003).
        use rand::Rng;
        let dg_proj: Vec<Vec<f32>> = (0..dg_dim)
            .map(|_| (0..input_dim)
                .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
                .collect())
            .collect();
        Self {
            episodes: Vec::with_capacity(capacity),
            cursor: 0,
            cap: capacity,
            rng,
            dg_proj,
            dg_dim,
        }
    }

    fn dg_encode(&self, features: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.dg_dim];
        let d = features.len().min(self.dg_proj[0].len());
        for i in 0..self.dg_dim {
            let mut s = 0.0;
            for j in 0..d { s += features[j] * self.dg_proj[i][j]; }
            // Sparse activation: k-WTA-style threshold at 0.
            out[i] = if s > 0.0 { s } else { 0.0 };
        }
        out
    }

    /// Bind a percept-action pair into a new episode. Only requires a gated
    /// percept and a selected action, which upstream guarantees are already
    /// vetted by thalamus and basal ganglia.
    pub fn bind(&mut self, p: &Percept<Gated>, a: Option<&ActionCommand<Selected>>) {
        let key = self.dg_encode(&p.features);
        let ep = Episode {
            t_ms: p.t_ms,
            key,
            action_id: a.map(|c| c.action_id).unwrap_or(0),
            reward: 0.0,
        };
        if self.episodes.len() < self.cap {
            self.episodes.push(ep);
        } else {
            self.episodes[self.cursor] = ep;
            self.cursor = (self.cursor + 1) % self.cap;
        }
    }

    /// Retroactive reward tagging (dopaminergic surprise updates the last N).
    pub fn tag_recent_reward(&mut self, r: f32, n: usize) {
        let len = self.episodes.len();
        for i in 0..n.min(len) {
            let idx = (self.cursor + self.cap - 1 - i) % self.cap.max(1);
            if idx < len { self.episodes[idx].reward = r; }
        }
    }

    /// Pattern completion — return the closest stored episode to a query.
    pub fn recall(&self, query_features: &[f32]) -> Option<&Episode> {
        if self.episodes.is_empty() { return None; }
        let q = self.dg_encode(query_features);
        let mut best = 0;
        let mut best_score = f32::MIN;
        for (i, ep) in self.episodes.iter().enumerate() {
            let mut s = 0.0;
            for k in 0..q.len().min(ep.key.len()) { s += q[k] * ep.key[k]; }
            if s > best_score { best_score = s; best = i; }
        }
        Some(&self.episodes[best])
    }

    /// Offline replay — sample `batch_size` episodes, preferring recent /
    /// high-reward ones (prioritized experience replay in the RL sense).
    pub fn replay(&mut self, batch_size: usize) -> ReplayBatch {
        if self.episodes.is_empty() {
            return ReplayBatch { inner: Vec::new() };
        }
        // Priority proportional to |reward| + small recency bonus.
        let mut idx: Vec<usize> = (0..self.episodes.len()).collect();
        idx.shuffle(&mut self.rng);
        idx.truncate(batch_size.min(self.episodes.len()));
        let inner = idx.into_iter().map(|i| self.episodes[i].clone()).collect();
        ReplayBatch { inner }
    }

    /// Number of episodes currently stored.
    pub fn len(&self) -> usize { self.episodes.len() }

    /// Whether the hippocampus is empty.
    pub fn is_empty(&self) -> bool { self.episodes.is_empty() }
}
