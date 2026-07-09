# agikernel

**A biologically-grounded hardware/kernel design concept for a theoretical AGI,
written in Rust and built on top of the [`neurorust`](../neurorust) neuron
substrate.**

> _"The best model of a cat is a cat, preferably the same cat."_ — Norbert Wiener
>
> _"Anything a biological brain can do, a machine that faithfully implements
> the same computational architecture can do."_ — the working hypothesis of
> this crate.

`agikernel` is **not** a claim to have built AGI. It is a **design document
that compiles**: a concrete, testable Rust scaffold that maps the proven
functional architecture of the mammalian brain onto the subsystems of an
operating-system-like kernel, so that alternative AGI hypotheses can be
prototyped in a shared, biologically faithful runtime instead of a
philosophical vacuum.

Every subsystem in this crate corresponds to a brain structure whose function
is empirically well established in systems neuroscience. The Rust type system
is used to enforce the invariants those structures rely on — spike-time
ordering, thalamic gating before cortical routing, hippocampal replay before
cortical consolidation, dopaminergic gating of plasticity, and so on.

---

## 1. Design thesis

Modern deep-learning AGI proposals collapse the brain into "a big
differentiable function". Neuroscience, in contrast, tells us the brain is
several **co-operating specialised computers** with well-characterised
protocols between them. `agikernel` takes the second view seriously and treats
those computers as **kernel subsystems** with typed interfaces:

| Kernel subsystem            | Biological homologue                       | Established function                              |
|-----------------------------|--------------------------------------------|---------------------------------------------------|
| `cortex`                    | Neocortex (6-layer canonical microcircuit) | Predictive world model, hierarchical abstraction  |
| `thalamus`                  | Thalamic nuclei (LGN, MGN, pulvinar, TRN)  | Sensory gating, attention, cortico-cortical relay |
| `hippocampus`               | Hippocampal formation (CA3, CA1, DG)       | Episodic binding, pattern separation & completion |
| `basal_ganglia`             | Striatum, GPi, STN, SNc                    | Action selection, RL policy gating                |
| `cerebellum`                | Cerebellar cortex + deep nuclei            | Forward models, fine timing, error correction     |
| `brainstem`                 | Brainstem + hypothalamus                   | Homeostasis, drives, arousal, safety              |
| `neuromod`                  | VTA, LC, Raphé, NBM (DA, NE, 5-HT, ACh)    | Global gain / plasticity modulation               |
| `bus`                       | White-matter long-range fibers             | Typed inter-region message passing                |
| `scheduler`                 | Thalamo-cortical rhythms (α, β, γ, θ)      | Global time base, phase-locked processing         |
| `syscall`                   | Prefrontal cognitive control               | External API — how "userspace" talks to the kernel |

The neurons filling every subsystem are `neurorust::Neuron` trait objects —
Hodgkin–Huxley, Izhikevich, or multi-compartment cells — so the kernel is
biophysically grounded rather than a metaphor.

---

## 2. Architecture at a glance

```
        ┌─────────────────────────────────────────────────────────────┐
        │                       syscall (PFC API)                     │
        └──────────────▲───────────────────────────────▲──────────────┘
                       │                               │
                       │ cognitive control             │ report / percept
                       │                               │
        ┌──────────────┴───────────────────────────────┴──────────────┐
        │                                                             │
        │   ┌──────────┐   gate    ┌──────────────────┐  predict/error│
        │   │ thalamus │◀──────────│      cortex      │◀──────────┐   │
        │   │   TRN    │  attention│ 6-layer columns  │           │   │
        │   └────▲─────┘           │  hierarchies     │           │   │
        │        │                 └──▲───────▲───────┘           │   │
        │  senses│                    │       │                   │   │
        │        │             action │       │ episode           │   │
        │  ┌─────┴────┐        select │       │ index    ┌────────┴───┴──┐
        │  │brainstem │        ┌──────┴──┐    │          │  cerebellum   │
        │  │homeostat.│        │ basal   │    │          │ forward model │
        │  │  drives  │        │ ganglia │    │          │  fine timing  │
        │  └─────▲────┘        └────▲────┘    │          └────────▲──────┘
        │        │                  │         │                   │
        │        │       ┌──────────┴─────────┴─────────┐         │
        │        │       │        hippocampus           │         │
        │        │       │   pattern-sep / completion   │─────────┘
        │        │       │  episodic replay → cortex    │
        │        │       └──────────────┬───────────────┘
        │        │                      │
        │        │             ┌────────┴────────┐
        │        └─────────────│    neuromod     │  DA / NE / 5-HT / ACh
        │                      │ global gain +   │  broadcast on the bus
        │                      │  plasticity     │
        │                      └─────────────────┘
        │                                                             │
        │      All arrows are typed messages on `bus::NeuralBus`.      │
        │      Global time base provided by `scheduler` (θ,α,β,γ).     │
        └─────────────────────────────────────────────────────────────┘
```

---

## 3. Biological invariants enforced by the type system

Rust's borrow checker is used to enforce properties that computational
neuroscience says the brain _must_ maintain. If the compiler accepts your
`agikernel` program, the following biological invariants are already
guaranteed:

1. **Thalamic gating precedes cortical arrival.** Every sensory
   [`Percept`](src/bus/mod.rs) must be minted by the `thalamus` module — the
   only place that owns the `PerceptToken` constructor. Cortex accepts
   `Percept<Gated>` only.

2. **Hippocampal replay is required before cortical consolidation.** The
   `cortex::consolidate` API takes a `hippocampus::ReplayBatch`, which can
   only be produced by `hippocampus::replay()`. No shortcut is possible.

3. **Basal-ganglia disinhibition is the only way to emit an action.** The
   `Motor` bus channel accepts only `ActionCommand<Selected>`, which is
   constructed exclusively in `basal_ganglia::select()`.

4. **Plasticity is dopamine-gated.** `cortex::CorticalColumn::learn` requires
   a `neuromod::DopamineTag`, minted only by `neuromod::vta_burst()`. STDP
   without a neuromodulator token is a compile error.

5. **Homeostasis has priority.** The scheduler wakes `brainstem` before any
   cortical column each cycle; if a homeostatic set-point is violated, cortex
   ticks are throttled — mirroring how a sleep-deprived or hungry brain
   loses cognitive bandwidth first.

These are not runtime asserts; they are **type-level guarantees** that make
whole classes of biologically implausible bugs unrepresentable. See
[`docs/invariants.md`](docs/invariants.md).

---

## 4. Kernel cycle

One "kernel tick" corresponds to a **gamma cycle (~25 ms)**, nested inside
alpha (~100 ms) and theta (~250 ms) rhythms, mirroring cortical oscillations:

```
loop {                                         // each tick ≈ 25 ms wallclock
    brainstem.tick();                          // 1. homeostasis first
    let percepts = thalamus.gate(senses);      // 2. attentional gate
    let (predict, err) = cortex.tick(&percepts);// 3. predictive coding
    let candidate = cortex.propose_action();   // 4. motor plans
    let action = basal_ganglia.select(candidate,
                                      neuromod.dopamine());
    cerebellum.refine(&mut action, &predict);  // 5. fine motor / timing
    hippocampus.bind(&percepts, &action);      // 6. episode binding
    if scheduler.is_theta_trough() {           // 7. offline replay
        let replay = hippocampus.replay();
        cortex.consolidate(replay);
    }
    neuromod.broadcast(err, brainstem.drives());// 8. global gain / RPE
    bus.flush();                               // 9. deliver messages
}
```

Every one of those steps has a peer-reviewed biological analogue; see
[`docs/mapping.md`](docs/mapping.md) for citations to the primary literature.

---

## 5. Hardware profile (theoretical)

`agikernel` is a **software kernel**, but its structure implies a matching
**neuromorphic hardware profile**. The `docs/hardware.md` file spells out how
each subsystem would be realised in silicon:

| Region        | Substrate                     | Rationale                              |
|---------------|-------------------------------|----------------------------------------|
| Cortex        | Massively parallel spiking cores (Loihi-2 / SpiNNaker-2 class) | 10⁴–10⁶ columns run identical code |
| Thalamus      | Low-latency crossbar with per-channel gain | Attention needs single-cycle gating |
| Hippocampus   | High-bandwidth associative memory (memristor / phase-change) | Content-addressable, huge fan-in |
| Basal ganglia | Winner-take-all analog core   | Continuous action-value comparison     |
| Cerebellum    | Systolic MAC array + BLAS/LAPACK offload | Regular dense linear algebra   |
| Brainstem     | Small hard-real-time microcontroller | Safety-critical, must never miss deadline |
| Neuromod bus  | Broadcast wire / optical fabric | Global fan-out, low bit-width         |

The **only tightly coupled ODE integration** (the neuron dynamics themselves)
lives in `neurorust`, which is already portable across CPU, and can be moved
to GPU or neuromorphic silicon behind the `Neuron` trait without touching the
kernel.

---

## 6. Building and running

```bash
# Build the library and the demo
cargo build --release

# Run a 1-second "mini-AGI" cycle with all subsystems wired up
cargo run --release --bin agikernel-demo

# Run the invariant / integration tests
cargo test --release
```

Enabling the Julia backend for cerebellar kernels:

```bash
cargo build --release --features julia
```

---

## 7. Scope, honesty, and non-goals

- **This is a design concept, not a claim of intelligence.** No single Rust
  crate can be AGI. What this crate offers is a _biologically faithful,
  compilable substrate_ in which specific AGI hypotheses (global workspace,
  predictive coding, active inference, world models, RL-in-the-loop) can be
  swapped in and out at the cortex/basal-ganglia layer without rebuilding
  the plumbing every time.
- **Numerical figures are illustrative.** Neuron counts, tick rates and
  compression ratios in the code default to values that make the demo
  finish in seconds on a laptop, not to human-scale figures (~86 × 10⁹
  neurons).
- **Ethics and safety** are first-class concerns. The `brainstem` subsystem
  is deliberately given priority over cortex, mirroring how homeostatic and
  safety circuits in a real brain outrank cognition. Any deployment on
  neuromorphic hardware should retain this priority ordering.

---

## 8. License

Dual-licensed under **MIT OR Apache-2.0**, at your option.
