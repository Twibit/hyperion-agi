//! End-to-end demo: instantiate one of each neuron kind, drive it with a
//! constant current, and report firing rate + memory footprint.

use neurorust::models::{HodgkinHuxleyNeuron, IzhikevichNeuron, MultiCompartmentNeuron};
use neurorust::neuron::{FiringPattern, Neuron};
use neurorust::models::multi_compartment::MultiCompartmentParams;

fn simulate<N: Neuron>(name: &str, neuron: &mut N, dt: f64, ms: f64, i_inj: f64) {
    let steps = (ms / dt) as usize;
    for _ in 0..steps {
        neuron.step(dt, i_inj);
    }
    let mem = neuron.memory();
    let spikes = mem.spikes.len();
    println!(
        "{:>20}  |  {:>6} spikes  |  V-trace {:>6.2}× compressed  |  {:>6} B on wire",
        name,
        spikes,
        mem.v_trace.compression_ratio(),
        mem.total_compressed_bytes()
    );
}

fn main() {
    env_logger::try_init().ok();
    println!("neurorust demo — 1 s of simulation, dt = 0.05 ms, I_inj = 10 µA/cm²\n");

    let dt = 0.05;
    let ms = 1000.0;
    let i_inj = 10.0;

    let mut hh = HodgkinHuxleyNeuron::new(dt);
    let mut izh = IzhikevichNeuron::preset(FiringPattern::RegularSpiking, dt);
    let mut mc = MultiCompartmentNeuron::new(dt, MultiCompartmentParams::default());

    simulate("Hodgkin–Huxley", &mut hh,  dt, ms, i_inj);
    simulate("Izhikevich (RS)", &mut izh, dt, ms, i_inj);
    simulate("Multi-compartment", &mut mc, dt, ms, i_inj);
}
