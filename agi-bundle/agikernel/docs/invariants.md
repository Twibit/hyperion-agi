# Type-level biological invariants

`agikernel` uses Rust's type system to make biologically implausible programs
**refuse to compile**. This is the same design philosophy that lets Rust
enforce data-race freedom — we extend it from memory safety to
*neuro-anatomical safety*.

## Invariant 1 — Thalamic gating precedes cortical arrival

```rust
pub struct Percept<S = Raw> { … _stage: PhantomData<S> }

impl Percept<Raw>   { pub    fn raw(…) -> Self }
impl Percept<Gated> { pub(crate) fn from_raw(…) -> Self }
```

`Percept<Gated>::from_raw` is `pub(crate)` and only invoked inside
`thalamus::Thalamus::gate`. Cortex requires `Percept<Gated>`. There is no
safe way for userspace to hand a raw percept to cortex.

## Invariant 2 — Consolidation requires replay

```rust
impl Cortex {
    pub fn consolidate(&mut self, batch: ReplayBatch, tag: &DopamineTag) { … }
}
```

`ReplayBatch` has a private constructor (`inner`). The only way to obtain
one is by calling `Hippocampus::replay`. `Cortex::consolidate` refuses
anything else — mirroring the two-stage complementary-learning-systems
theory.

## Invariant 3 — Motor output requires BG selection

```rust
pub struct ActionCommand<S = Proposed> { … }
impl ActionCommand<Selected> {
    pub(crate) fn from_proposed(p: ActionCommand<Proposed>) -> Self
}
```

The `Motor` bus channel is typed as `Sender<ActionCommand<Selected>>`. Only
`basal_ganglia::select` can produce that type. Cortex proposals cannot reach
the effectors without passing through BG.

## Invariant 4 — Plasticity is dopamine-gated

```rust
impl CorticalColumn {
    pub fn learn(&mut self, _tag: &DopamineTag, lr: f32) { … }
}
```

`DopamineTag` has a `pub(crate)` constructor produced only inside
`neuromod::NeuromodulatorySystem::broadcast` and `::vta_burst`. A caller who
tries to update cortical weights without holding one of these tags gets a
`missing arguments` compile error.

## Invariant 5 — Homeostasis has hard priority

```rust
impl Syscall {
    pub fn tick(&mut self) -> (f32, Option<u32>) {
        let alerts_before = self.brainstem.tick(&self.bus);
        let throttle      = self.brainstem.throttle_cortex;
        // …cortex is skipped when `throttle == true`…
    }
}
```

This one is a runtime invariant, not a type invariant — Rust does not have a
"phantom priority" mechanism. But the enforcement is centralised in exactly
one place (the master loop) and cannot be bypassed by any subsystem.
