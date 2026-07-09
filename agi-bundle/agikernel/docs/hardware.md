# Theoretical hardware profile

`agikernel` is software today, but every subsystem was chosen with an
implementable **neuromorphic hardware target** in mind. Below is the mapping
we recommend for a hypothetical AGI ASIC / heterogeneous system.

## Overview

```
+----------------------------+   +---------------------------+
| Cortex tile array          |   | Cerebellar systolic array |
|  10^4-10^6 spiking columns |   |  dense MAC + BLAS offload |
|  Loihi-2 / SpiNNaker-2 like|   |  handles cable eq, expm   |
+-------------▲--------------+   +------------▲--------------+
              │                                │
              │  gated percepts                │  motor refinement
              │                                │
+-------------┴--------------+   +-------------┴--------------+
| Thalamic crossbar          |◀──│ Basal-ganglia WTA analog    │
|  per-channel gain + TRN    │   │ selection core              │
|  single-cycle gating       │   │                             │
+-------------▲--------------+   +-----------------------------+
              │                                ▲
              │                                │
+-------------┴--------------+   +-------------┴--------------+
| Hippocampal CAM            │   │ Neuromod broadcast fabric  │
|  memristor / PCM associative│   │ optical or high-fan-out wire│
|  memory, huge fan-in       │   │ delivers DA/NE/5-HT/ACh globally │
+----------------------------+   +---------------------------+

                     ┌────────────────────────────┐
                     │ Brainstem safety MCU       │
                     │ hard-real-time,            │
                     │ hardware watchdog          │
                     └────────────────────────────┘
```

## Region-by-region rationale

### Cortex tile array

- **Substrate**: massively parallel spiking cores, e.g. Intel Loihi-2 or
  SpiNNaker-2 tiles. Each tile hosts one canonical column.
- **Why**: cortex is 10⁴–10⁶ copies of the *same* microcircuit. Silicon that
  replicates one small mesh 10⁶ times is far cheaper than a monolithic ASIC.
- **Communication**: 2-D mesh NoC, packetised spike routing.

### Thalamic crossbar

- **Substrate**: a low-latency N × M crossbar with per-channel programmable
  gain, plus a small array of TRN-like inhibitory cells.
- **Why**: attentional gating must be single-cycle — the thalamus does not
  buffer.

### Hippocampal CAM

- **Substrate**: high-bandwidth content-addressable memory. Emerging
  candidates: memristor arrays (Hewlett-Packard-style), phase-change memory
  (IBM), or crossbar RRAM.
- **Why**: pattern completion is literally "given a partial key, return the
  full stored vector". CAM is the exact hardware primitive.

### Basal-ganglia WTA core

- **Substrate**: a small analog winner-take-all core (Mead-style
  neuromorphic analog). Alternatively a small digital tournament tree.
- **Why**: WTA is continuous, low-dimensional, and latency-critical.

### Cerebellar systolic array

- **Substrate**: a dense MAC systolic array with an FMA offload to BLAS/
  LAPACK — precisely what `neurorust::ffi::julia` already targets.
- **Why**: cerebellar computations reduce to large linear regressions and
  tridiagonal cable-equation solves.

### Neuromod broadcast fabric

- **Substrate**: an optical or high-fan-out electrical broadcast bus with a
  small (≤ 8 bits per neuromodulator) payload. No routing table — every
  cortex tile just listens.
- **Why**: neuromodulators are global scalars, not point-to-point signals.

### Brainstem safety MCU

- **Substrate**: a *separate*, hard-real-time microcontroller running a
  hardware watchdog. Independent power domain.
- **Why**: exactly like a real brainstem, this must never miss a deadline,
  so it does not share the substrate that runs cognition.

## Power / clock budget (illustrative)

| Region       | Est. dynamic power fraction | Notes                          |
|--------------|-----------------------------|--------------------------------|
| Cortex       | 60 %                        | Bulk of the machine            |
| Cerebellum   | 20 %                        | Dense linear algebra is hungry |
| Hippocampus  | 8 %                         | Bursty at theta troughs        |
| Thalamus     | 5 %                         | Small crossbar                 |
| BG + neuromod| 4 %                         | Low-dim, always-on             |
| Brainstem    | 3 %                         | Always-on, but tiny            |

This distribution mirrors metabolic PET/fMRI data for humans (Sokoloff et
al., 1977; Raichle & Mintun, 2006) — cortex dominates, cerebellum runs
warm, everything else is small.
