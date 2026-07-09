use pyo3::prelude::*;
use pyo3::types::PyComplex;
use pyo3::Py;
use std::hash::Hasher;
use fnv::FnvHasher;
use std::fs::File;
use std::io::{BufRead, BufReader};
use ai_eye::Eye;
use neurorust::neuron::{FiringPattern, Neuron};
use neurorust::models::izhikevich::IzhikevichNeuron;
use agikernel::Syscall;

mod attention;

/// Stable 64-bit hash. NOTE: the pure-Python shim uses BLAKE2b; this uses
/// FNV-1a. Both are stable 64-bit hashes. `virtual_100t_substrate` only uses
/// the result as a neuron address, so the algorithm difference is
/// functionally irrelevant (no persisted state is keyed by it). Kept as FNV
/// to avoid pulling in a crypto dep for an offline build.
#[pyfunction]
fn hash_neuron_id(concept_bytes: &[u8], domain_bytes: &[u8], offset: i32) -> PyResult<u64> {
    let mut hasher = FnvHasher::default();
    hasher.write(concept_bytes);
    hasher.write(domain_bytes);
    hasher.write_i32(offset);
    Ok(hasher.finish())
}

/// Integrates Astrocyte Calcium based on Tripartite Synapse Biology.
#[pyfunction]
fn calcium_integrate(
    current_calcium: f64, pre_spike: f64, post_spike: f64, dopamine: f64,
    decay_rate: Option<f64>, influx_scale: Option<f64>, calcium_max: Option<f64>,
) -> PyResult<f64> {
    let decay = decay_rate.unwrap_or(0.95);
    let scale = influx_scale.unwrap_or(0.1);
    let cap = calcium_max.unwrap_or(1.0);
    let calcium_influx = (pre_spike * post_spike) * scale;
    let modulated_influx = calcium_influx * dopamine;
    let mut new_calcium = (current_calcium * decay) + modulated_influx;
    if new_calcium > cap {
        new_calcium = cap;
    }
    Ok(new_calcium)
}

/// Computes a full N-point complex FFT, matching `numpy.fft.fft(samples).tolist()`.
/// Implemented as a naive O(n^2) DFT so we need no `rustfft`/`num-complex` deps
/// (inputs here are small audio chunks). Returns a list of Python complex numbers.
#[pyfunction]
fn compute_fft(py: Python, samples: Vec<f64>) -> PyResult<Vec<Py<PyComplex>>> {
    let n = samples.len();
    let mut out = Vec::with_capacity(n);
    let pi = std::f64::consts::PI;
    for k in 0..n {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (i, &x) in samples.iter().enumerate() {
            let theta = -2.0 * pi * (k as f64) * (i as f64) / (n as f64);
            re += x * theta.cos();
            im += x * theta.sin();
        }
        out.push(PyComplex::from_doubles(py, re, im).into());
    }
    Ok(out)
}

/// Reward-modulated STDP delta — EXACT match to rust_backend_shim.py semantics
/// so the native module is a drop-in fast-path with identical behavior.
#[pyfunction]
fn reward_modulated_stdp(
    trace_a: f64, trace_b: f64, dopamine: f64, calcium: f64,
    calcium_midpoint: Option<f64>, calcium_steepness: Option<f64>,
) -> PyResult<f64> {
    let midpoint = calcium_midpoint.unwrap_or(0.2);
    let steepness = calcium_steepness.unwrap_or(-8.0);
    let coactivation = (trace_a * trace_b).tanh();
    let dopamine_gain = dopamine.max(-1.0).min(1.0);
    let calcium_gate = 1.0 / (1.0 + (steepness * (calcium - midpoint)).exp());
    Ok(coactivation * dopamine_gain * calcium_gate)
}

/// Prediction error given predicted and actual activation.
#[pyfunction]
fn predictive_error(predicted_activation: f64, actual_activation: f64) -> PyResult<f64> {
    Ok(actual_activation - predicted_activation)
}

/// NATIVE VISION: Reads an entire massive file (13GB+) natively in Rust,
/// bypassing the Python GIL and string allocation overhead.
/// Pre-filters lines by checking for basic keyword presence, reducing the
/// 400-million line payload down to a few hundred lines instantly.
#[pyfunction]
fn native_omni_vision_scan(
    filepath: &str, keywords: Vec<String>,
    buffer_bytes: Option<usize>, telemetry_interval: Option<u64>,
) -> PyResult<Vec<String>> {
    let buf_size = buffer_bytes.unwrap_or(16 * 1024 * 1024);
    let telem_interval = telemetry_interval.unwrap_or(10_000_000);

    let file = File::open(filepath)?;
    let reader = BufReader::with_capacity(buf_size, file);
    let mut highly_salient_nodes = Vec::new();

    let lower_keywords: Vec<String> = keywords.iter().map(|k| k.to_lowercase()).collect();
    let mut nodes_seen: u64 = 0;

    for line_result in reader.lines() {
        nodes_seen += 1;
        
        if nodes_seen % telem_interval == 0 {
            println!("      [Rust Vision Core] Natively scanned {} lines...", nodes_seen);
        }

        if let Ok(line) = line_result {
            let lower_line = line.to_lowercase();
            for kw in &lower_keywords {
                if lower_line.contains(kw) {
                    highly_salient_nodes.push(line);
                    break;
                }
            }
        }
    }
    Ok(highly_salient_nodes)
}

/// The Python Module (named `hyperion_rust_core` so it never shadows the
/// pure-Python `rust_backend_shim.py` fallback).
#[pymodule]
fn hyperion_rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash_neuron_id, m)?)?;
    m.add_function(wrap_pyfunction!(calcium_integrate, m)?)?;
    m.add_function(wrap_pyfunction!(compute_fft, m)?)?;
    m.add_function(wrap_pyfunction!(reward_modulated_stdp, m)?)?;
    m.add_function(wrap_pyfunction!(predictive_error, m)?)?;
    m.add_function(wrap_pyfunction!(attention::sparse_attention, m)?)?;
    m.add_function(wrap_pyfunction!(native_omni_vision_scan, m)?)?;
    m.add_class::<NativeUiEye>()?;
    m.add_class::<NativeMotorSystem>()?;
    m.add_class::<NativeIzhikevich>()?;
    m.add_class::<NativeAgiKernel>()?;
    Ok(())
}

/// NATIVE UI EYE: A biological UI perception engine that mirrors the mammalian visual system.
/// Keeps state between glances so that looking at the same page twice costs ~0ms.
#[pyclass]
struct NativeUiEye {
    inner: Eye,
}

#[pymethods]
impl NativeUiEye {
    #[new]
    fn new() -> Self {
        NativeUiEye { inner: Eye::new() }
    }

    fn glance_url(&mut self, url: &str) -> PyResult<String> {
        match self.inner.glance_url(url) {
            Ok(percept) => {
                match serde_json::to_string(&percept) {
                    Ok(json) => Ok(json),
                    Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("JSON error: {}", e))),
                }
            },
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Vision error: {}", e))),
        }
    }
}

/// NATIVE MOTOR SYSTEM: A biological actor mesh for safe, rollback-protected operations.
#[pyclass]
struct NativeMotorSystem {
    rt: tokio::runtime::Runtime,
    hand: telekinesis::limb::LimbHandle,
}

#[pymethods]
impl NativeMotorSystem {
    #[new]
    fn new() -> PyResult<Self> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Tokio error: {}", e)))?;

        let hand = rt.block_on(async {
            let mut ns = telekinesis::actor::NervousSystem::new().await.expect("Failed to create nervous system");
            ns.grow_limb(telekinesis::limb::FileLimb::new()).await
        });

        Ok(Self { rt, hand })
    }

    fn reach_file(&self, intent_json: &str) -> PyResult<String> {
        let intent: telekinesis::intent::Intent = serde_json::from_str(intent_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid intent: {}", e)))?;
        
        let outcome = self.rt.block_on(async {
            self.hand.reach(intent).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Motor error: {}", e)))?;
        
        let outcome_json = serde_json::to_string(&outcome)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Outcome serialization error: {}", e)))?;
            
            
        Ok(outcome_json)
    }
}

/// NATIVE IZHIKEVICH: Phenomenological fast-path biological neuron from neurorust.
#[pyclass]
struct NativeIzhikevich {
    inner: IzhikevichNeuron,
}

#[pymethods]
impl NativeIzhikevich {
    #[new]
    fn new(pattern: &str, dt_ms: f64) -> PyResult<Self> {
        let pat = match pattern {
            "RegularSpiking" => FiringPattern::RegularSpiking,
            "IntrinsicallyBursting" => FiringPattern::IntrinsicallyBursting,
            "Chattering" => FiringPattern::Chattering,
            "FastSpiking" => FiringPattern::FastSpiking,
            "LowThresholdSpiking" => FiringPattern::LowThresholdSpiking,
            "Quiescent" => FiringPattern::Quiescent,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown FiringPattern")),
        };
        Ok(Self { inner: IzhikevichNeuron::preset(pat, dt_ms) })
    }

    fn step(&mut self, dt_ms: f64, i_inj: f64) -> PyResult<String> {
        let state = self.inner.step(dt_ms, i_inj);
        Ok(format!("{},{}", state.v_mem, state.spiked))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// NATIVE AGI KERNEL: Exposes the full multi-region biologically-grounded agikernel to Python.
#[pyclass(unsendable)]
struct NativeAgiKernel {
    inner: Syscall,
}

#[pymethods]
impl NativeAgiKernel {
    #[new]
    fn new(depth: i32, n_per_layer: i32, n_channels: i32, top_k: i32, dt_ms: f64) -> Self {
        Self {
            inner: Syscall::new(depth as usize, n_per_layer as usize, n_channels as usize, top_k as usize, dt_ms),
        }
    }

    fn sense(&self, modality: u8, features: Vec<f32>) {
        self.inner.sense(modality, features);
    }

    fn reward(&mut self, magnitude: f32) -> f32 {
        self.inner.reward(magnitude)
    }

    fn refuel(&mut self, drive_str: &str, amount: f32) -> PyResult<()> {
        let drive = match drive_str {
            "Energy" => agikernel::bus::Drive::Energy,
            "Temperature" => agikernel::bus::Drive::Temperature,
            "Sleep" => agikernel::bus::Drive::Sleep,
            "Safety" => agikernel::bus::Drive::Safety,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unknown Drive: {}", drive_str))),
        };
        self.inner.refuel(drive, amount);
        Ok(())
    }

    fn tick(&mut self) -> PyResult<String> {
        let (err, action_id) = self.inner.tick();
        let json_str = format!(
            "{{\"err\":{},\"action_id\":{}}}",
            err,
            match action_id {
                Some(id) => id.to_string(),
                None => "null".to_string(),
            }
        );
        Ok(json_str)
    }

    fn t_ms(&self) -> f64 {
        self.inner.t_ms()
    }

    // Neuromodulatory System getters
    #[getter]
    fn dopamine(&self) -> f32 {
        self.inner.neuromod.da
    }

    #[getter]
    fn noradrenaline(&self) -> f32 {
        self.inner.neuromod.ne
    }

    #[getter]
    fn serotonin(&self) -> f32 {
        self.inner.neuromod.ht
    }

    #[getter]
    fn acetylcholine(&self) -> f32 {
        self.inner.neuromod.ach
    }

    // Brainstem getters
    #[getter]
    fn arousal(&self) -> f32 {
        self.inner.brainstem.arousal
    }

    #[getter]
    fn throttle_cortex(&self) -> bool {
        self.inner.brainstem.throttle_cortex
    }

    fn drive_value(&self, drive_str: &str) -> PyResult<f32> {
        let drive = match drive_str {
            "Energy" => agikernel::bus::Drive::Energy,
            "Temperature" => agikernel::bus::Drive::Temperature,
            "Sleep" => agikernel::bus::Drive::Sleep,
            "Safety" => agikernel::bus::Drive::Safety,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unknown Drive: {}", drive_str))),
        };
        if let Some(state) = self.inner.brainstem.drives.get(&drive) {
            Ok(state.value)
        } else {
            Ok(0.0)
        }
    }
}
