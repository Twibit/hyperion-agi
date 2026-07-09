# AGI Kernel — biologically-grounded design concept in Rust

This bundle contains **two Rust crates**:

- `neurorust/` — the **biologically-accurate neuron substrate** you attached
  (Hodgkin–Huxley, Izhikevich, multi-compartment cable models, with per-datum
  hybrid compression and a Julia FFI hook). *Unmodified from the input.*
- `agikernel/` — a new **kernel / hardware design concept for a theoretical
  AGI**, layered on top of `neurorust`. Every subsystem maps to a proven
  biological brain region; the Rust type system encodes anatomical invariants
  at compile time. See `agikernel/README.md` for the full architectural
  narrative and citations.

## Quick start

```bash
cd agikernel
cargo build --release          # compiles both crates (path dep on neurorust)
cargo test  --release          # runs 7 integration tests (all should pass)
cargo run   --release --example agikernel-demo
```

Verified on rustc 1.96.1 (2026-06-26): compiles clean, all tests pass,
end-to-end mini-AGI demo runs.
