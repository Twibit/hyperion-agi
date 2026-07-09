# Biology → Silicon mapping table

This document is the single source of truth for *why every design choice in
CortexOS exists*. Each row is a biological finding, an engineering primitive,
and the file/kernel that implements it.

| # | Biological substrate | Reference | CortexOS primitive | File |
|---|----------------------|-----------|--------------------|------|
| 1 | Ionic action potential (Na⁺, K⁺, leak) | Hodgkin & Huxley 1952 | `HHState`, `hh_step_rk4` | `src/cuda/hh_step_kernel.cu` |
| 2 | Two-variable minimal spiking model | Izhikevich 2003 | `IzhState`, `izh_step` | `src/cuda/izh_step_kernel.cu` |
| 3 | Cable equation on dendrites | Rall 1959 | `Compartment`, `cable_step_kernel` | `src/triton/cable_step.py` |
| 4 | NMDA plateau potentials on apical tuft | Larkum et al. 1999 | `apical_plateau_flag` on `Compartment` | `src/triton/cable_step.py` |
| 5 | Cortical minicolumn (~100 neurons) | Mountcastle 1957 | `Minicolumn` | `include/agi/column.hpp` |
| 6 | Cortical column (~10⁴ neurons, 6 layers) | Douglas & Martin 2004 | `Column`, `column_step` | `src/triton/column_step.py` |
| 7 | Canonical microcircuit (L2/3 → L5 → L6 → thalamus) | Bastos et al. 2012 | `LayerRouting` inside `Column::step` | `src/host/column.cpp` |
| 8 | Small-world + rich-club topology | Bullmore & Sporns 2009 | `HubOverlay` on top of local CSR | `src/host/connectivity.cpp` |
| 9 | Log-normal synaptic weight distribution | Song et al. 2005 | 8-bit log-quantised weights + LUT | `src/cuda/csr_spmv_spike.cu` |
| 10 | Axonal delays 0.1–25 ms | Swadlow 2000 | 8-bit delay in AER packet | `include/agi/core.hpp` |
| 11 | Thalamic relay + attentional gain | Sherman & Guillery 2006 | `Thalamus` router | `src/cuda/thalamus_route.cu` |
| 12 | Pulvinar cross-cortical routing | Saalmann et al. 2012 | `pulvinar_route()` method | `src/host/thalamus.cpp` |
| 13 | Basal-ganglia action selection | Redgrave et al. 1999 | `wta_select` kernel | `src/cuda/wta_select.cu` |
| 14 | Dopaminergic RPE broadcast | Schultz 1998 | `da` scalar on rail 0 | `src/host/neuromod.cpp` |
| 15 | Noradrenergic gain modulation | Aston-Jones 2005 | `ne` scales `g_leak⁻¹` in kernels | `src/host/neuromod.cpp` |
| 16 | Cholinergic novelty gating | Sarter 2005 | `ach` lowers spike threshold | `src/host/neuromod.cpp` |
| 17 | Serotonergic patience modulation | Doya 2002 | `ht5` scales RL γ | `src/host/neuromod.cpp` |
| 18 | Pair-based STDP | Bi & Poo 1998 | `stdp_update` kernel | `src/cuda/stdp_update.cu` |
| 19 | Three-factor learning (Hebb × neuromod) | Frémaux & Gerstner 2016 | `da`-gated STDP | `src/cuda/stdp_update.cu` |
| 20 | Eligibility trace, τ ≈ 500 ms | Izhikevich 2007 | `e_trace` decayed each step | `src/plasticity/eligibility.cpp` |
| 21 | Hippocampal replay during sleep | Wilson & McNaughton 1994 | `ReplayEngine` | `src/cuda/replay.cu` |
| 22 | Synaptic homeostasis / pruning during sleep | Tononi & Cirelli 2014 | Nightly prune step | `src/host/structural.cpp` |
| 23 | Predictive coding hierarchy | Rao & Ballard 1999 | Per-column `(prediction, error)` pair | `src/host/column.cpp` |
| 24 | Free-energy minimisation | Friston 2010 | Global objective (optional application layer) | example only |
| 25 | Poisson firing / rate coding of sensory inputs | Softky & Koch 1993 | `PoissonEncoder` | `src/io/sensor_encoders.hpp` |
| 26 | Sparse cortical firing rate (~1–5 Hz) | Attwell & Laughlin 2001 | AER bandwidth budget assumes 2 Hz mean | design doc §7 |
| 27 | ~10¹⁰ neurons, 10¹⁴ synapses (1 % human) | Herculano-Houzel 2009 | Target scale in ARCHITECTURE.md §7 | design doc |
| 28 | Membrane compression (i16 quant, 2.4 µV err) | from `neurorust` | `MembraneTrace` codec | `src/memory/membrane_codec.cpp` |
| 29 | Spike-train compression (delta-ISI + zstd) | from `neurorust` | `SpikeTrainCodec` | `src/memory/spike_codec.cpp` |
| 30 | Bit-packed population raster | from `neurorust` | `BitRasterCodec` | `src/memory/raster_codec.cpp` |
