# Hyperion — a bio-inspired research substrate

> **Note on the previous README.** The version of `README.md` that shipped
> in the archive described Hyperion as *"the 100-trillion-parameter
> biological spiking neural network"* and claimed the code
> *"achieves organic language, reasoning and logic"*. That is not
> what the code does. This README is a truthful replacement; the
> original marketing text has been preserved as
> [`README_ORIGINAL.md`](README_ORIGINAL.md).
> See [`AGI_ASSESSMENT.md`](AGI_ASSESSMENT.md) for the full analysis.

## What this is

Hyperion is a Python package of biologically-themed research components
glued together. Concretely, it contains:

- A **predictive-coding cortex stack** with per-block precision and
  Hebbian updates (`embodied_agent/cortex/predictive_block.py`).
- A **Leaky-Integrate-and-Fire memristor layer with STDP** — one
  numeric kernel, available as either a small Rust `.so` or a
  pure-numpy port (`embodied_agent/cortex/snn_engine.py`).
- An **HTM sequence memory** for temporal prediction
  (`embodied_agent/praxis/htm_predictor.py`).
- A **global workspace + default-mode network** with a thalamic
  salience router and metabolic budget
  (`embodied_agent/subcortical/`).
- A **cerebellum forward/inverse model**, a **hippocampus** with
  episode replay, a **basal ganglia** action selector, and an
  **active-inference** policy selector.
- A **Hebbian embedding cortex** + **Predictive Coding language model** +
  **SQLite episodic memory** for a very small chat loop
  (`nlp_engine/layers/cain_brain.py`, `main.py`).
- A **Markov-blanket + Expected-Free-Energy tree-search planner**
  (`nlp_engine/`, `dual_planner.py`).

## What this isn't

Hyperion is **not** an AGI. It cannot pass a general reasoning
benchmark, does not hold long-horizon coherent conversation, and does
not transfer skills across tasks. It is a substrate — the components
are legitimate and worth iterating on, but the wiring, the training
data, and the parameter scale are all far below what general capability
would require. See [`AGI_ASSESSMENT.md`](AGI_ASSESSMENT.md) for a
detailed evaluation and the changes I made to get the archive
running.

## Quickstart

```bash
# Only hard dependency is numpy.
python3 -m pip install numpy

# 1) See what actually runs in this checkout.
python3 cli.py inspect

# 2) Run the Right/Left brain thought demo (dual_planner.py, ~0.5 s).
python3 cli.py dual-brain --proposals 5000

# 3) Run the embodied_agent embodied loop for 50 steps.
python3 cli.py aurora --steps 50

# 4) Boot the CainBrain chat loop (needs ./data/ from the archive).
python3 cli.py chat
```

`cli.py` is a thin dispatcher — the underlying scripts
(`dual_planner.py`, `main.py`, `embodied_agent/*`) still work directly if
you prefer.

### If you have a Rust toolchain

The `embodied_agent` numeric kernels have a Rust fast path. If your
current machine's architecture isn't compatible, you can rebuild them:

```bash
cd rust_backend && cargo build --release
cd ../agi-bundle/neurorust && cargo build --release
```

Otherwise the pure-numpy fallback kicks in automatically. You can force
the fallback for benchmarking:

```bash
HYPERION_FORCE_PY=1 python3 cli.py aurora --steps 100
```

## Repository layout (high level)

```
hyperion/
├── cli.py                    # ONE entry point for all modes.
├── embodied_agent/           # The most complete subsystem. Embodied loop.
├── nlp_engine/               # Cortex modules (CainBrain).
├── cognitive_plugins/        # Amygdala / claustrum / VTA / basal ganglia / ...
├── rust_backend/             # Legacy Rust numeric kernels (ctypes).
├── agi-bundle/neurorust/     # v33 Advanced Rust kernels (WTA, SIMD, STDP).
├── agi-kernel/               # C++/CUDA runtime execution kernels.
├── docs/                     # Assessment, changelogs, architecture guides.
├── examples/                 # Extracted legacy scripts (main, planner, gridworld).
├── rust_backend_shim.py      # Pure-Python shim for the missing PyO3 module.
└── README.md                 # This file.
```

## Provenance and honesty

This tree started life as `hyperion_agi_lightweight.tar.gz`. As shipped
it was **not runnable on x86_64** — the top-level demo crashed with
`ImportError`, the `embodied_agent` package could not import at all, and
`CainBrain` failed on a missing PyO3 module. Those issues have been
fixed additively; see [`CHANGES.md`](CHANGES.md) for the full diff of
what I touched and why.

If you're evaluating whether this is worth building on: yes, as a
research substrate for bio-inspired components. No, as a shortcut to
general intelligence.
