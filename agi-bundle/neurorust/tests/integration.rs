//! Integration tests: biological plausibility + compression correctness.

use approx::assert_relative_eq;
use neurorust::compression::{BitRaster, MembraneTrace, SparseConnectivity, SpikeTrain};
use neurorust::models::multi_compartment::MultiCompartmentParams;
use neurorust::models::{HodgkinHuxleyNeuron, IzhikevichNeuron, MultiCompartmentNeuron};
use neurorust::neuron::{FiringPattern, Neuron, NeuronKind};

// ---- Neuron dynamics -------------------------------------------------------

#[test]
fn hh_fires_under_suprathreshold_current() {
    let dt = 0.01;
    let mut n = HodgkinHuxleyNeuron::new(dt);
    for _ in 0..((200.0 / dt) as usize) {
        n.step(dt, 10.0);
    }
    // 200 ms at I_inj = 10 µA/cm² should produce several spikes.
    assert!(n.memory().spikes.len() >= 3, "expected ≥3 spikes, got {}", n.memory().spikes.len());
}

#[test]
fn hh_is_quiet_at_rest() {
    let dt = 0.01;
    let mut n = HodgkinHuxleyNeuron::new(dt);
    for _ in 0..((100.0 / dt) as usize) {
        n.step(dt, 0.0);
    }
    assert_eq!(n.memory().spikes.len(), 0);
    assert!(n.v_mem() < -50.0 && n.v_mem() > -90.0);
}

#[test]
fn izh_regular_spiking_fires_within_expected_rate() {
    let dt = 0.05;
    let mut n = IzhikevichNeuron::preset(FiringPattern::RegularSpiking, dt);
    for _ in 0..((1000.0 / dt) as usize) {
        n.step(dt, 10.0);
    }
    let spikes = n.memory().spikes.len();
    // RS at I=10 → roughly 20–80 Hz.
    assert!((10..=120).contains(&spikes), "expected 10..=120 spikes, got {spikes}");
}

#[test]
fn multi_compartment_propagates_and_spikes() {
    let dt = 0.05;
    let mut n = MultiCompartmentNeuron::new(dt, MultiCompartmentParams::default());
    for _ in 0..((500.0 / dt) as usize) {
        n.step(dt, 12.0);
    }
    assert_eq!(n.kind(), NeuronKind::MultiCompartment);
    assert!(n.memory().spikes.len() >= 1);
}

// ---- Compression correctness ----------------------------------------------

#[test]
fn spike_train_roundtrip_is_lossless() {
    let mut s = SpikeTrain::new(0.05);
    let inputs = vec![1.0, 5.25, 5.30, 100.0, 100.05, 999.95];
    s.extend(inputs.iter().copied()).unwrap();
    let back = s.to_times_ms().unwrap();
    assert_eq!(back.len(), inputs.len());
    for (a, b) in inputs.iter().zip(back.iter()) {
        assert_relative_eq!(a, b, epsilon = 0.05);
    }
}

#[test]
fn v_trace_roundtrip_below_biological_noise() {
    let mut v = MembraneTrace::new(-100.0, 60.0, 0.05);
    let src: Vec<f32> = (0..1_000).map(|i| -65.0 + 20.0 * ((i as f32) * 0.01).sin()).collect();
    v.extend(src.iter().copied()).unwrap();
    let back = v.to_samples().unwrap();
    assert_eq!(back.len(), src.len());
    // ≤ 5 µV error (well below 1 mV membrane noise).
    for (a, b) in src.iter().zip(back.iter()) {
        assert!((a - b).abs() < 5e-3, "reconstruction error {} exceeded 5 µV", (a - b).abs());
    }
    assert!(v.compression_ratio() >= 1.5);
}

#[test]
fn csr_matches_dense_spmv() {
    let rows = 8;
    let cols = 8;
    let mut dense = vec![0.0f32; rows * cols];
    dense[0 * cols + 3] = 1.5;
    dense[4 * cols + 4] = -2.0;
    dense[7 * cols + 0] = 0.5;
    let csr = SparseConnectivity::from_dense(rows, cols, &dense, 1e-9);
    let x: Vec<f32> = (0..cols).map(|i| i as f32).collect();

    let mut out_sparse = vec![0.0; rows];
    csr.spmv(&x, &mut out_sparse);

    let mut out_dense = vec![0.0f32; rows];
    for r in 0..rows {
        for c in 0..cols {
            out_dense[r] += dense[r * cols + c] * x[c];
        }
    }
    for (a, b) in out_sparse.iter().zip(out_dense.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-6);
    }
}

#[test]
fn bit_raster_compression_is_8x() {
    let mut r = BitRaster::zeros(100, 1000);
    r.set(3, 7, true);
    assert!(r.get(3, 7));
    assert!(!r.get(0, 0));
    assert_relative_eq!(r.compression_ratio(), 8.0, epsilon = 0.1);
}

// ---- FFI fallback ----------------------------------------------------------

#[test]
fn tridiag_solver_matches_reference() {
    // Toy 3×3: T = [[2,-1,0],[-1,2,-1],[0,-1,2]], b = [1,0,1] → x = [1,1,1].
    let sub  = vec![-1.0, -1.0];
    let diag = vec![ 2.0,  2.0, 2.0];
    let sup  = vec![-1.0, -1.0];
    let b    = vec![ 1.0,  0.0, 1.0];
    let x = neurorust::ffi::solve_tridiag(&sub, &diag, &sup, &b);
    for xi in &x { assert_relative_eq!(*xi, 1.0, epsilon = 1e-9); }
}
