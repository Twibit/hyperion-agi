# v33 migration guide

This document explains **what changed at the API level** between v32
and v33, so existing code can adopt the fixed learning layer or stay on
the old path deliberately.

## Nothing is required to migrate

v33 is strictly additive at the Python level. All v32 imports continue
to work — `MemristorLayer` still exists, still has the same signature,
and still has the same (broken-without-teacher) behaviour. If you
prefer to keep the old learning failure mode for baseline comparison,
change nothing.

## To adopt the fix

### Drop-in replacement: `WTAMemristorLayer`

```python
# v32
from embodied_agent.cortex.snn_engine import MemristorLayer
layer = MemristorLayer(n_inputs=32, n_outputs=4, threshold=1.0, leak=0.9)

# v33 — same constructor signature, plus optional knobs
from embodied_agent.cortex.snn_engine_v2 import WTAMemristorLayer
layer = WTAMemristorLayer(
    n_inputs=32, n_outputs=4,
    threshold=1.0, leak=0.9,
    # new v33 knobs (all optional, defaults are sane):
    init_noise=8.0,          # Gaussian σ around 128/255 for symmetry breaking
    seed=0,                  # deterministic init; pass None for true random
    homeostatic_lr=0.02,     # rate at which per-neuron thresholds adapt
    target_rate=0.25,        # desired win-share per neuron (1/n_outputs)
)
```

`layer.forward(spikes_in)` returns `spikes_out` exactly as before.
`layer.weights`, `layer.potentials` are the same fields, same dtypes.

New public fields on the v33 layer:
- `layer.thresholds` — per-neuron adaptive threshold (starts at
  `threshold`, drifts to make the win-rate match `target_rate`).
- `layer.win_ema` — running win-rate estimate per neuron.

### Sparse-firing cortex: `KWtaMemristorLayer`

For pools larger than a handful of outputs, hard WTA (winner=1) is a
poor match for biology — real cortex fires ~2–5 % of pyramidal neurons
per gamma cycle. Use k-WTA:

```python
from embodied_agent.cortex.snn_engine_v2 import KWtaMemristorLayer

# 256 output units, sparse code with top-5% (=13 winners)
layer = KWtaMemristorLayer(n_inputs=1024, n_outputs=256,
                           threshold=1.0, leak=0.9, k=None)  # k=None → ceil(0.05*n)
```

### Reward-modulated learning: `ThreeFactorMemristorLayer`

For any task where the environment gives you a scalar reward *after* an
action:

```python
from embodied_agent.cortex.snn_engine_v2 import ThreeFactorMemristorLayer

layer = ThreeFactorMemristorLayer(
    n_inputs=32, n_outputs=4,
    threshold=1.0, leak=0.9,
    tau_e_ticks=20.0,   # eligibility trace time constant, in ticks
    eta_ltp=0.02,       # per-coincidence eligibility bump (LTP side)
    eta_ltd=0.005,      # per-coincidence eligibility bump (LTD side)
    lr=0.5,             # weight-update rate on reward(...)
)

# training loop
for pattern, target_class in dataset:
    spikes = layer.forward(pattern)          # regular Hebbian forward
    predicted = int(np.argmax(spikes))
    r = +1.0 if predicted == target_class else -0.3
    layer.reward(r)                          # commits eligibility → weights
```

The eligibility trace `layer.eligibility` (shape `(n_outputs, n_inputs)`,
`float32`) is the causal-attribution vector — it remembers which
synapses were coactive up to `tau_e_ticks` ago. Reward gates whether
that memory gets committed to weights.

## Rust FFI (optional)

The `three_factor_stdp` Rust module exposes a C-ABI:

```rust
extern "C" {
    fn stdp_new(n_pre: usize, n_post: usize, tau_e: f32) -> *mut StdpHandle;
    fn stdp_free(h: *mut StdpHandle);
    fn stdp_tick(
        h: *mut StdpHandle,
        pre_spikes: *const u8, n_pre: usize,
        post_spikes: *const u8, n_post: usize,
        weights: *mut u8,
        modulator: f32,
    );
}
```

This isn't wired into the shipped `snn_engine_v2.py` fallback path
(pure-numpy is fast enough for the shipped test — 10k forward-passes/sec)
but the FFI is available for anyone building a larger simulation. Build
with:

```sh
cd agi-bundle/neurorust
RUSTFLAGS="-C target-cpu=native" cargo build --release
# → target/release/libneurorust.so
```

## Runtime kernel dispatch

For downstream code that wants the fastest path automatically:

```rust
use neurorust::asm_kernels_avx512::{axpy_f64_best, saxpy_f32_best};

// dispatches AVX-512 → AVX2 → scalar based on `is_x86_feature_detected!`
axpy_f64_best(0.5, &x, &mut y);
saxpy_f32_best(0.5, &xf, &mut yf);
```

On aarch64 the NEON path is used automatically when compiled for that
target — no dispatch call needed.

## What tests to run

```sh
# Rust unit tests (21 tests, incl. 12 new v33 ones)
cd agi-bundle/neurorust
RUSTFLAGS="-C target-cpu=native" cargo test --release --lib

# Rust benchmark (25-iter median timings for all v33 kernels)
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin neurorust-bench-v33

# Python learning-capability test (v33, no external teacher)
cd ../..
python3 tests/test_learning_capability_v33.py

# Original v32 learning test (still passes with external teacher, no regression)
python3 tests/test_learning_capability.py
```
