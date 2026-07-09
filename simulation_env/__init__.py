"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  B R A I N   S I M   —  Biologically Plausible Cortical Simulation        ║
║                                                                             ║
║  A multi-scale brain simulation with:                                      ║
║    • Multi-compartment Hodgkin-Huxley neurons (soma + dendrite + AIS)      ║
║    • STDP synapses with AMPA/NMDA/GABA_A/GABA_B receptors                  ║
║    • 6-layer cortical column (L1-L6) with 4 cell types                     ║
║    • Thalamocortical relay + hippocampal formation                         ║
║    • Neuromodulation (dopamine, acetylcholine, serotonin, norepinephrine)  ║
║    • ATP energy accounting per spike and per synapse                       ║
║                                                                             ║
║  Reference: Hodgkin & Huxley (1952), Markram et al. (2015),                ║
║  Blue Brain Project column model, Izhikevich (2007)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from .models import (
    EnergyBudget, GatingKinetics, HodgkinHuxleySoma,
    MultiCompartmentNeuron, STDPSynapse,
    NeuronGroup, SynapseMatrix,
    CellType, Layer, CorticalColumn,
    Neuromodulator, NeuromodulationSystem,
    ThalamocorticalRelay, HippocampalFormation,
    simulate_step, run_simulation
)

__version__ = "0.1.0"
