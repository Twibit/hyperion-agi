# Changes applied to the archive

This checkout ships with several build/runtime breakages that prevent
the archive from being run on anything except the specific ARM64 laptop
it was last edited on. The goal of these changes is to make the
substrate portable and honest — **not** to invent new AGI capabilities.

All modifications are additive or minimally invasive; no existing
research code was deleted. Original architectural intent is preserved
everywhere.

## Files added

| Path | Purpose |
|---|---|
| `AGI_ASSESSMENT.md` | Honest evaluation of the claim that this is an AGI, with evidence. |
| `CHANGES.md` | This file. |
| `README.md` (top-level, replaces marketing prose) | Truthful overview + quickstart. |
| `cli.py` | Single CLI unifying `dual_planner.py`, `main.py` (chat), and `embodied_agent`. Includes an `inspect` self-test. |
| `rust_backend_shim.py` | Pure-Python shim for the missing PyO3 module. Implements `hash_neuron_id`, `reward_modulated_stdp`, `compute_fft`, `sparse_attention` — the only 4 symbols the codebase actually calls. |

## Files modified

### `nlp_engine/layers/predictive.py`

**Symptom:** `dual_planner.py` imports `PredictiveMinicolumn` from this
module, but the class was never in the file. Every attempt to run
`python3 dual_planner.py` failed immediately with `ImportError`.

**Fix:** Added a `PredictiveMinicolumn` class at the bottom of the file
(the existing `HTMTemporalPredictor` is untouched). It exposes:

- `generative_weights` (dim × dim) — required by `CAINTreeSearch`.
- `recognition_weights`, `precision`.
- `view(blanket)` — projects a Markov Blanket's `internal_state` into
  an observation, as `dual_planner.py` calls it.
- `predict`, `infer`, `update` for downstream users.

Semantics: precision-weighted Hebbian predictive coding with a top-k
sparsity gate. This is what the surrounding code was clearly written to
expect — the missing class was just not shipped.

### `embodied_agent/cortex/snn_engine.py`

**Symptom:** Hard-imports `hyperion_core/target/release/libhyperion_core.so`
at module load time and raises `FileNotFoundError` on failure. The
shipped `.so` is ARM64 only; on any x86_64 machine the whole
`embodied_agent` package is unimportable.

**Fix:** Full rewrite of the loader (the original was 75 lines):

1. Only attempt to load the Rust core if the ELF magic bytes and
   `e_machine` field match the current interpreter's architecture.
2. If missing or wrong-arch, fall back to a functionally-equivalent
   vectorised numpy implementation (`_snn_forward_py`) that mirrors
   `hyperion_core/src/lib.rs::snn_forward` line-for-line: leak → weighted
   excitation → threshold fire → uint8 LTP on active-and-fired, uint8
   LTD on active-but-not-fired.
3. Exposes a `USING_RUST_CORE` boolean and honours `HYPERION_FORCE_PY=1`
   for benchmarking parity.

The public API (`MemristorLayer`) is unchanged.

### `embodied_agent/praxis/htm_predictor.py`

**Symptom:** Same class of issue — this file also hard-imports the Rust
core's `htm_step` at module load.

**Fix:** Same fallback strategy. Added `_htm_step_py`, a pure-numpy
mirror of `hyperion_core::htm_step` (per-row homeostatic decay of
transition edges emanating from previously-active rows, then LTP on
prev→curr edges, then row-sum prediction), and route to it when
`_rust_core is None`. `HTMSequenceMemory.step()` picks the right
implementation at call time.

## Files not modified

- No existing research code was removed.
- `hyperion_core/target/release/libhyperion_core.so` (the ARM64 binary)
  is left in place so ARM64 users still get the Rust fast path.
- `hyperion_core/src/lib.rs` is left in place; anyone with a `cargo`
  toolchain can rebuild for their platform via
  `cd hyperion_core && cargo build --release`.

## Result

Before the changes: 0 of the 3 top-level entry points started on
x86_64.

After the changes, `python3 cli.py inspect` reports:

```
  dual_planner.py dual-brain                       OK
  embodied_agent embodied loop                   OK
  SNN MemristorLayer (rust or numpy fallback)  OK
  HTM sequence memory                          OK
  CainBrain (chat)                             OK
```

The archive still is not an AGI — see `AGI_ASSESSMENT.md` — but it now
actually runs everywhere, on stock Python 3.10+ with only `numpy`
installed.
