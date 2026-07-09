# CortexOS — A Biologically-Grounded Hardware/Kernel Stack for Theoretical AGI

**Languages:** C++20 (host runtime & IR), CUDA C++ (device kernels), OpenAI Triton (fused block-level kernels).
**Reference neuron model:** [`neurorust 0.1.0`](../reference/neurorust/) — Hodgkin–Huxley, Izhikevich, and multi‑compartment cable neurons with hybrid per‑datum compression.

> CortexOS is a **design concept**: a full hardware/kernel blueprint for an AGI substrate whose primitives are proven biological ones — spiking neurons, cortical columns, thalamo‑cortical loops, neuromodulation, and predictive coding — rather than dense matmul.
> Every architectural choice is anchored in a specific, cited biological finding.

---

## 1. Design pillars

| # | Biological principle | Engineering realisation |
|---|----------------------|--------------------------|
| 1 | **Spiking, event-driven communication** (Hodgkin & Huxley 1952) | Sparse spike packets on an AER (Address-Event Representation) fabric; no dense activations on the wire |
| 2 | **Cortical column as canonical circuit** (Mountcastle 1957; Douglas & Martin 2004) | Column = SM tile of ~10⁴ neurons × 6 layers, exposed as a first-class kernel object |
| 3 | **Multi-compartment dendritic computation** (London & Häusser 2005) | Cable-equation Triton kernel per column; dendritic NMDA plateaus supported |
| 4 | **Thalamo-cortical loop for attention & routing** (Sherman & Guillery 2006) | On-chip "thalamic" router with pulvinar-style relay & attentional gating |
| 5 | **Neuromodulation as global scalars** (Schultz 1998; Aston-Jones 2005) | Four broadcast rails: DA (reward), 5-HT (patience), NE (gain), ACh (uncertainty) |
| 6 | **Hebbian + STDP + three-factor learning** (Bi & Poo 1998; Frémaux & Gerstner 2016) | On-line CUDA STDP kernel gated by the neuromodulatory rails |
| 7 | **Predictive coding / free energy** (Rao & Ballard 1999; Friston 2010) | Every column emits `(prediction, error)` pairs; error drives learning |
| 8 | **Hippocampal replay & consolidation** (Wilson & McNaughton 1994) | Off-line replay engine drives cortical STDP during "sleep" phases |
| 9 | **Sparse, small-world connectivity** (Bullmore & Sporns 2009) | CSR fabric with hub/rich-club overlay; ~1–10 % density like real neocortex |
| 10| **Hybrid per-datum memory compression** (from `neurorust`) | Same four codecs (delta-ISI, i16 quantised V, CSR, bit-raster) at exascale |

---

## 2. Stack overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Application layer   (Python / C++ — cognitive skills, planners, LLMs)  │
├─────────────────────────────────────────────────────────────────────────┤
│  CortexOS IR          (C++20 — MLIR-style dialect: column, spike,       │
│                        synapse, neuromod, plasticity ops)               │
├─────────────────────────────────────────────────────────────────────────┤
│  Scheduler & Runtime  (C++20 — event-driven step engine, thalamic       │
│                        router, replay engine, neuromod broadcast)       │
├─────────────────────────────────────────────────────────────────────────┤
│  Device kernels                                                         │
│    • CUDA C++   → HH / Izhikevich integrators, AER routing, STDP        │
│    • Triton     → fused cable-eq + synaptic drive + spike gen           │
├─────────────────────────────────────────────────────────────────────────┤
│  Memory subsystem     (C++20 — hybrid codec arenas, HBM/DDR/NVMe tiers) │
├─────────────────────────────────────────────────────────────────────────┤
│  Hardware abstraction (GPUs today; neuromorphic ASIC target tomorrow)   │
└─────────────────────────────────────────────────────────────────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full spec and
[`docs/BIO_MAPPING.md`](docs/BIO_MAPPING.md) for the biology→silicon mapping table.

---

## 3. Repository layout

```
agi-kernel/
├── docs/
│   ├── ARCHITECTURE.md           # 30-page design doc
│   ├── BIO_MAPPING.md            # biology → silicon mapping table
│   ├── IR_SPEC.md                # CortexOS IR spec
│   └── ROADMAP.md                # 5-phase build plan
├── include/agi/                  # public C++ headers
│   ├── core.hpp                  # tensor/spike types, error codes
│   ├── neuron.hpp                # column, neuron, compartment structs
│   ├── column.hpp                # cortical column API
│   ├── connectivity.hpp          # CSR/hub-and-spoke wiring
│   ├── thalamus.hpp              # thalamo-cortical router API
│   ├── neuromod.hpp              # DA/5-HT/NE/ACh broadcast rails
│   ├── plasticity.hpp            # STDP + three-factor rules
│   ├── memory.hpp                # hybrid codec arena
│   ├── ir.hpp                    # IR ops
│   ├── scheduler.hpp             # event-driven engine
│   └── runtime.hpp               # top-level session API
├── src/host/                     # C++ host implementation
├── src/cuda/                     # CUDA device kernels (.cu)
├── src/triton/                   # Triton kernels (.py)
├── src/ir/                       # IR passes
├── src/scheduler/                # event engine
├── src/memory/                   # codec arenas
├── src/plasticity/               # learning rules
├── src/io/                       # AER I/O, checkpointing
├── tests/                        # gtest suites
├── benchmarks/                   # scaling & throughput
├── examples/                     # small runnable demos
└── cmake/                        # CUDA + Triton build integration
```

---

## 4. Build (design-time skeleton)

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DAGI_ENABLE_CUDA=ON \
  -DAGI_ENABLE_TRITON=ON \
  -DCMAKE_CUDA_ARCHITECTURES="90"        # Hopper / H100
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Triton kernels are AOT-compiled into a shared object under `build/triton_cache/`.

---

## 5. Provenance & license

* Neuron biophysics: derived from [`neurorust`](../reference/neurorust/) (MIT / Apache-2.0).
* Design doc: Genspark AGI hardware/kernel concept, 2026.
* License: MIT.
