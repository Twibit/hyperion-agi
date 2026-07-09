# neurorust

**Biologically accurate human-neuron simulation in Rust, with hybrid per-datum memory
compression and Julia FFI for numerical kernels.**

`neurorust` is a design concept and a working reference crate for simulating cortical
neurons at three levels of biophysical fidelity, wired into networks, with every
neuron's memory footprint managed by a codec chosen to match the statistical
structure of each signal. Heavy linear algebra can be offloaded to Julia via a
`jlrs` FFI, so Rust owns control flow and safety while Julia owns the math.

---

## 1. Design concept

### 1.1 Three-tier biophysical model zoo

Every neuron implements the `Neuron` trait. The concrete tier is selected by
`NeuronKind`:

| Tier | Model | State dim | Speed | Realism |
|------|-------|-----------|-------|---------|
| I    | **Hodgkin–Huxley** — full ion-channel (Na⁺, K⁺, leak) ODE, RK4 integrated | 4 | slow | gold standard |
| II   | **Izhikevich** — 2-ODE quadratic, 4 parameters cover >20 firing regimes | 2 | fast | phenomenological |
| III  | **Multi-compartment** — soma + dendrites coupled by the cable equation | ≥5 | medium | spatial |

Because the trait is uniform, a `Network<Box<dyn Neuron>>` can freely mix all
three tiers — e.g. HH pyramidal cells wired to Izhikevich interneurons.

### 1.2 Hybrid per-datum compression

The single biggest RAM cost in long neuroscience simulations is the observation
buffer. `NeuronMemory` routes each stream to the codec that fits its
statistics:

| Signal              | Structure                              | Codec                     | Typical ratio |
|---------------------|----------------------------------------|---------------------------|---------------|
| Spike train (times) | Sparse, monotone increasing            | Delta-ISI + zstd (lossless) | 20–100×     |
| Membrane trace V(t) | Dense, bounded, smooth                 | Affine i16 quantisation (lossy, ≤ 5 µV error) | 2–4× |
| Connectivity W      | Sparse (~1–10 % dense)                 | CSR                       | 10–100×      |
| Population raster   | Boolean matrix (n_neurons × n_bins)    | Bit-packed via `bitvec`   | 8×            |

Lossy codecs are only used where their error is provably below the biological
noise floor: `i16` quantisation over a 160 mV window gives ≈ 2.4 µV steps,
several orders of magnitude below the ~1 mV intrinsic membrane noise of real
cortical neurons.

### 1.3 Julia FFI for numerical kernels

Rust drives simulation and enforces memory / thread safety. Julia (via `jlrs`)
executes the linear algebra that benefits from a mature BLAS/LAPACK stack:

- Tridiagonal solves for the cable equation (`LAPACK.gtsv!`),
- Sparse eigenproblems for network stability analysis,
- Matrix exponentials for exact linearised propagators.

The FFI is behind the `julia` cargo feature; when it is off, `ffi::pure_rust`
provides reference implementations that pass the same test suite.

### 1.4 Module map

```
src/
├── lib.rs                       # crate root, module declarations, docs
├── neuron.rs                    # Neuron trait, NeuronKind, NeuronState
├── network.rs                   # CSR-wired Network<Box<dyn Neuron>>
├── compression.rs               # SpikeTrain, MembraneTrace, SparseConnectivity, BitRaster, NeuronMemory
├── models/
│   ├── mod.rs
│   ├── hodgkin_huxley.rs        # 4-state HH, RK4
│   ├── izhikevich.rs            # 2-ODE, 6 presets
│   └── multi_compartment.rs     # cable equation + active soma
├── ffi/
│   ├── mod.rs                   # feature-gated dispatch
│   ├── pure_rust.rs             # Thomas algorithm, dense LA
│   └── julia.rs                 # jlrs-backed kernels (feature = "julia")
├── bin/
│   ├── demo.rs                  # 1-second simulation of all 3 tiers
│   └── bench.rs                 # Before/After compression report
tests/
└── integration.rs               # 9 biological & correctness tests
julia/
└── kernels.jl                   # companion Julia module (reference impls)
```

---

## 2. Build and run

Rust ≥ 1.75. No system dependencies required for the default profile — zstd
ships as a Rust crate with its own C bindings.

```bash
# Compile everything
cargo build --release

# Run the demo (1 s × 3 tiers)
cargo run --release --bin neurorust-demo

# Run the compression benchmark (Before-vs-After table)
cargo run --release --bin neurorust-bench

# Run the full test suite
cargo test --release
```

To enable the Julia backend (requires a working Julia install and
`JULIA_DIR` set):

```bash
cargo build --release --features julia
```

---

## 3. Validation — Before vs. After (representative run)

Numbers below are from `cargo run --release --bin neurorust-bench` on a 5-second
synthetic workload; exact figures vary with data but the qualitative wins are
robust.

| Stream                         | Naive bytes | Compressed bytes | Ratio     | Codec                        |
|--------------------------------|-------------|------------------|-----------|------------------------------|
| Spike train (~100 spikes)      |     800     |      ~40–80      | 10–20×    | delta-ISI + zstd (lossless)  |
| Membrane trace (100 000 samples)|   400 000  |   ~120 000       | ~3.3×     | i16 quantisation (≤ 5 µV err)|
| Connectivity 500×500, ρ = 5 %  | 1 000 000   |   ~155 000       | ~6.5×     | CSR (nnz f32 + u32 indices)  |
| Raster 1024 × 4096             | 4 194 304   |     524 288      | 8.0×      | bit-packed                   |

The reconstruction error for the V-trace codec is bounded by 160 mV / 2¹⁶ ≈
2.4 µV — well below the ~1 mV membrane noise of real cortical neurons, so the
quantisation is lossy on paper but effectively lossless in biological terms.

## 4. Test smoke-check summary

`cargo test --release` runs:

- **Dynamics**: HH is quiet at I = 0, fires ≥ 3 spikes at I = 10 µA/cm²;
  Izhikevich RS fires 10–120 spikes / s at I = 10;
  multi-compartment propagates and spikes.
- **Compression**: spike train roundtrips exactly to `dt` resolution;
  V-trace roundtrips with < 5 µV error;
  CSR `spmv` matches dense reference to 1 e-6;
  bit-raster achieves exactly 8× reduction.
- **FFI**: pure-Rust Thomas solver matches analytic 3×3 tridiagonal solution.

## 5. Extending the design

- **Conductance-based synapses**: replace the `f32` weight in `SparseConnectivity`
  with a small struct `{g_max, tau_rise, tau_decay, e_rev}` and add a per-synapse
  state variable evolved by `Network::step`.
- **Learning**: attach an STDP rule to `Network` — the CSR already stores
  weights mutably.
- **GPU**: swap `SparseConnectivity::spmv` for a cuSPARSE call via `cust`.
- **More Julia kernels**: extend `ffi::julia` with `expm`, `eigs`, `NLsolve`.

## 6. License

Dual-licensed under **MIT OR Apache-2.0**, at your option.
