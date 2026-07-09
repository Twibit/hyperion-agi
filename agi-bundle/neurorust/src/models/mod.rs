//! Concrete neuron implementations.

pub mod hodgkin_huxley;
pub mod izhikevich;
pub mod multi_compartment;

pub use hodgkin_huxley::HodgkinHuxleyNeuron;
pub use izhikevich::IzhikevichNeuron;
pub use multi_compartment::MultiCompartmentNeuron;
