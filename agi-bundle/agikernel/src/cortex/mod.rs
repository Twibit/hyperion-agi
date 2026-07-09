//! # cortex — hierarchical 6-layer canonical microcircuit
//!
//! Biological homologue: the **neocortex** organised into
//! [Mountcastle-1978 canonical columns](https://en.wikipedia.org/wiki/Cortical_column):
//! roughly 10⁴ neurons across six layers with a highly conserved intra-column
//! wiring diagram (L4 receives thalamic input, L2/3 does cortico-cortical
//! communication, L5 sends motor/sub-cortical output, L6 feeds back to
//! thalamus). This microcircuit is repeated ~10⁶ times across the human
//! neocortex and is generally accepted as the substrate of hierarchical
//! predictive processing (Bastos et al., 2012; Larkum, 2013).
//!
//! ## Kernel role
//!
//! [`Cortex`] owns many [`CorticalColumn`]s arranged in a hierarchy. Each
//! column implements **predictive coding**:
//!
//! - L4 receives feed-forward input (from thalamus or from the lower column).
//! - L2/3 computes a prediction of L4 activity.
//! - The **prediction error** = L4 − L2/3 prediction is propagated up.
//! - L6 sends predictions **down** to the previous stage (and to thalamus,
//!   closing the cortico-thalamic loop).
//! - L5 emits candidate motor actions to basal ganglia.
//!
//! Plasticity is gated by a [`crate::neuromod::DopamineTag`] — a token that
//! only the neuromodulatory system can mint. Learning without a valid tag
//! is a **compile error**, mirroring the fact that STDP in vivo is strongly
//! modulated by dopamine (Reynolds & Wickens, 2002).

use crate::bus::{ActionCommand, Percept, Gated, NeuralBus};
use crate::hippocampus::ReplayBatch;
use crate::neuromod::DopamineTag;
use neurorust::compression::SparseConnectivity;
use neurorust::models::hodgkin_huxley::HodgkinHuxleyNeuron;
use neurorust::models::izhikevich::IzhikevichNeuron;
use neurorust::network::Network;
use neurorust::neuron::FiringPattern;
use neurorust::Neuron;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Which cortical layer a neuron belongs to inside a canonical column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// Layer 2/3 — cortico-cortical prediction pyramidal cells.
    L23,
    /// Layer 4 — thalamo-recipient stellate cells.
    L4,
    /// Layer 5 — sub-cortical output pyramidal cells.
    L5,
    /// Layer 6 — corticothalamic feedback cells.
    L6,
}

/// A single canonical column: six-layer microcircuit.
pub struct CorticalColumn {
    /// The full population, laid out as [L2/3 | L4 | L5 | L6]. L1 is axonal
    /// input territory only and holds no cell bodies in this simplification.
    pub network: Network,
    /// (start, end) index ranges into `network.neurons` for L2/3, L4, L5, L6.
    layer_ranges: [(usize, usize); 4],
    /// Running L2/3 rate estimate (the column's *prediction*).
    prediction: Vec<f32>,
    /// Latest L4 prediction error, per L4 cell.
    pub last_error: Vec<f32>,
    /// Expected value of the most recent action proposal (L5 activity).
    ev: f32,
    /// Column id (position in a cortical hierarchy).
    pub id: usize,
    /// Time step in ms.
    dt_ms: f64,
}

impl CorticalColumn {
    /// Build a canonical column with `n_per_layer` cells per layer.
    ///
    /// L2/3 and L5 pyramidal cells use Hodgkin–Huxley (gold-standard
    /// biophysics). L4 stellates and L6 corticothalamic cells use Izhikevich
    /// for speed.
    pub fn new(id: usize, n_per_layer: usize, dt_ms: f64) -> Self {
        let mut neurons: Vec<Box<dyn Neuron>> = Vec::with_capacity(n_per_layer * 4);
        for _ in 0..n_per_layer { neurons.push(Box::new(HodgkinHuxleyNeuron::new(dt_ms))); }
        for _ in 0..n_per_layer { neurons.push(Box::new(IzhikevichNeuron::preset(FiringPattern::RegularSpiking, dt_ms))); }
        for _ in 0..n_per_layer { neurons.push(Box::new(HodgkinHuxleyNeuron::new(dt_ms))); }
        for _ in 0..n_per_layer { neurons.push(Box::new(IzhikevichNeuron::preset(FiringPattern::RegularSpiking, dt_ms))); }

        let n = neurons.len();
        // ~5 % sparse random weights, consistent with cortical connection
        // probabilities (Braitenberg & Schüz, 1998).
        let weights = random_csr(n, n, 0.05, 0.4, 42 + id as u64);

        let ranges = [
            (0,               n_per_layer),        // L2/3
            (n_per_layer,     2 * n_per_layer),    // L4
            (2 * n_per_layer, 3 * n_per_layer),    // L5
            (3 * n_per_layer, 4 * n_per_layer),    // L6
        ];

        Self {
            network: Network::new(neurons, weights),
            layer_ranges: ranges,
            prediction: vec![0.0; n_per_layer],
            last_error: vec![0.0; n_per_layer],
            ev: 0.0,
            id,
            dt_ms,
        }
    }

    /// Number of cells per layer.
    pub fn n_per_layer(&self) -> usize {
        self.layer_ranges[1].1 - self.layer_ranges[1].0
    }

    /// Advance the column by one tick with an external current injected
    /// into L4 (feed-forward input).
    pub fn step(&mut self, l4_input: &[f64]) {
        let n = self.network.neurons.len();
        let mut ext = vec![0.0_f64; n];
        let (l4_lo, l4_hi) = self.layer_ranges[1];
        for (i, x) in l4_input.iter().enumerate().take(l4_hi - l4_lo) {
            ext[l4_lo + i] = *x;
        }
        let spikes = self.network.step(self.dt_ms, &ext);

        // L2/3 firing → prediction rate (exponential moving average).
        let (l23_lo, l23_hi) = self.layer_ranges[0];
        for (i, s) in spikes[l23_lo..l23_hi].iter().enumerate() {
            self.prediction[i] = 0.9 * self.prediction[i] + if *s { 0.1 } else { 0.0 };
        }

        // L4 prediction error = |L4 rate - prediction|.
        for (i, s) in spikes[l4_lo..l4_hi].iter().enumerate() {
            let rate = if *s { 1.0 } else { 0.0 };
            self.last_error[i] = (rate - self.prediction[i]).abs();
        }

        // L5 fraction firing → expected-value estimate for action gating.
        let (l5_lo, l5_hi) = self.layer_ranges[2];
        let fired_l5 = spikes[l5_lo..l5_hi].iter().filter(|s| **s).count();
        self.ev = fired_l5 as f32 / (l5_hi - l5_lo) as f32;
    }

    /// Emit a proposed action (from the L5 pyramidal population).
    pub fn propose(&self, t_ms: f64) -> ActionCommand {
        ActionCommand::propose(t_ms, self.id as u32, vec![self.ev], self.ev)
    }

    /// Apply plasticity. Requires a [`DopamineTag`] because in vivo STDP is
    /// dopamine-gated. Without the tag this call does not compile.
    pub fn learn(&mut self, tag: &DopamineTag, lr: f32) {
        let da = tag.level();
        for w in self.network.weights.weights.iter_mut() {
            *w += lr * da * 0.01_f32;
            *w = w.clamp(-1.0, 1.0); // rough Dale's-law-ish clip.
        }
    }
}

/// Build a sparse random CSR connectivity of given density.
fn random_csr(rows: usize, cols: usize, density: f64, w_std: f32, seed: u64)
              -> SparseConnectivity {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut dense = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            if r != c && rng.gen::<f64>() < density {
                // Signed normal weights.
                let x: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                dense[r * cols + c] = (x as f32) * w_std;
            }
        }
    }
    SparseConnectivity::from_dense(rows, cols, &dense, 1e-6)
}

/// A cortical hierarchy: a stack of columns feeding forward, with feedback
/// via L6 → thalamus (closed by [`crate::thalamus::Thalamus::set_attn_bias`]).
pub struct Cortex {
    /// Columns, ordered from V1-like (bottom) to prefrontal-like (top).
    pub columns: Vec<CorticalColumn>,
    /// Cross-column broadcast buffer — a coarse analogue of Global Workspace
    /// "ignition" (Dehaene, 2011).
    pub workspace: Vec<f32>,
}

impl Cortex {
    /// Build a hierarchy of `depth` columns × `n_per_layer` cells / layer.
    pub fn new(depth: usize, n_per_layer: usize, dt_ms: f64) -> Self {
        let columns = (0..depth).map(|i| CorticalColumn::new(i, n_per_layer, dt_ms)).collect();
        Self { columns, workspace: vec![0.0; n_per_layer] }
    }

    /// Number of columns.
    pub fn depth(&self) -> usize { self.columns.len() }

    /// Advance the whole hierarchy under a batch of gated percepts.
    ///
    /// Returns `(workspace_content, mean_prediction_error_at_bottom,
    /// consumed_percepts)`. The percepts are returned so downstream
    /// subsystems (hippocampus in particular) can bind them without racing
    /// cortex on the same crossbeam channel.
    pub fn tick(&mut self, bus: &NeuralBus)
                -> (Vec<f32>, f32, Vec<Percept<Gated>>) {
        // Drain gated percepts.
        let mut incoming: Vec<Percept<Gated>> = Vec::new();
        while let Ok(p) = bus.gated_rx.try_recv() { incoming.push(p); }

        let depth = self.columns.len();
        let n_per_layer = self.columns[0].n_per_layer();

        // Aggregate features per column: modality mod depth selects column.
        let mut per_column_input: Vec<Vec<f64>> = vec![vec![0.0; n_per_layer]; depth];
        for p in &incoming {
            let c = (p.modality as usize) % depth;
            for (i, f) in p.features.iter().enumerate().take(n_per_layer) {
                per_column_input[c][i] += (*f as f64) * (p.salience as f64) * 10.0;
            }
        }

        // Feed-forward: bottom column takes external input, each higher
        // column takes the prediction from the lower one.
        for c in 0..depth {
            let input: Vec<f64> = if c == 0 {
                per_column_input[0].clone()
            } else {
                self.columns[c - 1].prediction.iter().map(|x| *x as f64 * 5.0).collect()
            };
            self.columns[c].step(&input);
        }

        // Broadcast: workspace = top column L2/3 prediction.
        self.workspace = self.columns.last().unwrap().prediction.clone();

        let bottom = &self.columns[0];
        let mean_err: f32 = bottom.last_error.iter().sum::<f32>()
            / bottom.last_error.len().max(1) as f32;
        (self.workspace.clone(), mean_err, incoming)
    }

    /// Consolidate replayed episodes into cortical weights. Requires a
    /// [`ReplayBatch`] — which only hippocampus can produce — encoding the
    /// two-stage memory theory (Marr, 1971; McClelland, McNaughton &
    /// O'Reilly, 1995).
    pub fn consolidate(&mut self, batch: ReplayBatch, tag: &DopamineTag) {
        for _episode in batch.iter() {
            for col in &mut self.columns {
                col.learn(tag, 1e-4);
            }
        }
    }

    /// Ask cortex what motor action each column wants this tick.
    pub fn propose_actions(&self, t_ms: f64) -> Vec<ActionCommand> {
        self.columns.iter().map(|c| c.propose(t_ms)).collect()
    }

    /// Attentional bias for thalamus: derived from the top column's L2/3
    /// prediction. L6 in vivo carries this feedback; we approximate.
    pub fn attn_bias(&self, n_channels: usize) -> Vec<f32> {
        let top = self.columns.last().unwrap();
        let mut bias = vec![1.0; n_channels];
        for (i, v) in top.prediction.iter().enumerate().take(n_channels) {
            bias[i] = 1.0 + 2.0 * v;
        }
        bias
    }

    /// Estimate of compressed memory footprint across all columns.
    pub fn memory_bytes(&self) -> usize {
        self.columns.iter()
            .flat_map(|c| c.network.neurons.iter())
            .map(|n| n.memory().total_compressed_bytes())
            .sum()
    }
}
