# Does Hyperion Function as an AGI? — Honest Assessment

**Short answer: No.** Hyperion is an ambitious *bio-inspired research
substrate*, not an AGI. It contains many of the ingredients researchers
associate with general intelligence (predictive coding, STDP-based
plasticity, a global workspace, Hebbian embeddings, HTM sequence memory,
active inference, a Markov-blanket + tree-search planner), but the way
they are wired together in this checkout produces, at best, a small
biologically-themed chat toy with hand-written intent routing.

This document explains **why** it isn't AGI, what parts are real, what
parts are decorative, and what I changed to make the archive at least
*run* end-to-end.

---

## 1. What "functions as an AGI" would actually require

At a minimum, a system claiming AGI status should demonstrate:

1. **Open-domain competence** — solves problems it wasn't hand-tuned for,
   above chance, on tasks a human would recognise as diverse (reading,
   arithmetic, planning, language, tool use, common-sense reasoning).
2. **Transfer / generalisation** — learning in one task lifts performance
   in another related task without full retraining.
3. **Grounded language** — words map to concepts that survive rephrasing;
   the system doesn't collapse when synonyms are used.
4. **Long-horizon coherence** — can hold a plan or a topic across many
   time steps without losing state.
5. **Self-correction from experience** — makes fewer of the same mistakes
   over time in a measurable way, without hand-coded rules.

Nothing in this checkout demonstrates any of the above, and nothing in
the code has the *scale of trained parameters* or the *learning signal
density* that modern general systems rely on.

## 2. What Hyperion actually is

Hyperion is three loosely-coupled personalities glued into one repo:

| Personality | Entry point | What it really does |
|---|---|---|
| **Dual-brain thought** | `dual_planner.py` | Generates ~10 000 random gaussian vectors ("right brain"), scores them against a target vector via Expected-Free-Energy tree search ("left brain"). Fun demo of predictive-coding math, not intelligence. |
| **CainBrain chat** | `main.py` + `nlp_engine/layers/cain_brain.py` | An N-gram language model + Hebbian embedding cortex + SQLite episodic memory + keyword-based intent classifier. Chats in short sentences from a hand-written phrase pool, with emojis. |
| **Aurora-Omnia embodied loop** | `embodied_agent/organism.py` | Actually the most serious subsystem: a step-based observer with cortex stacks (predictive block, MoE, world model), global workspace + default mode network, cerebellum forward/inverse model, hippocampus, active inference selector, thalamic routing, metabolic budget, neuromodulator bus. Does *not* learn a task to competence. |

The three do **not** actually integrate. `main.py` never calls the
`embodied_agent.AuroraOmnia` step function. `dual_planner.py` never touches
either. There is no top-level control loop that binds them.

## 3. The "100 trillion neurons" claim

The README says Hyperion has 100 T tripartite synapses. In practice:

- `Virtual100TSubstrate` (`nlp_engine/layers/scale86b/virtual_100t_substrate.py`)
  reserves an integer address space of 100 T neuron IDs. Only ~4096 are
  ever *instantiated* per step. The 100 T is an address-space
  parameter, not a stored-parameter count.
- The Rust `.so` implements a Leaky-Integrate-and-Fire memristor layer
  (~450 KB of compiled code, one small numeric kernel).
- The largest learned matrices in this repo are the Hebbian embedding
  table (~64 dim × ~10 K vocab ≈ 640 K floats) and the STDP working
  memory (10 K × 10 K trace matrix).

That is orders of magnitude below what modern general models require.
Calling it "100 T" is at best aspirational, at worst misleading.

## 4. Things that were broken on arrival

Running the archive as shipped produced these failures:

1. `python3 dual_planner.py` → `ImportError: cannot import name
   'PredictiveMinicolumn' from 'nlp_engine.layers.predictive'`.
   The class simply wasn't in the file.
2. `from embodied_agent.organism import AuroraOmnia` →
   `OSError: cannot open shared object file: libhyperion_core.so`.
   The `.so` **was** in the archive, but it was compiled for **ARM64**
   (`e_machine=0xB7`). Every x86_64 machine (and this sandbox) refuses
   to load it.
3. `from nlp_engine.layers.cain_brain import CainBrain; CainBrain()` →
   `ModuleNotFoundError: No module named 'rust_backend_shim'`.
   This is a *separate* Rust PyO3 module (different from the ctypes
   `.so`) that was never shipped at all. Used by the
   `Virtual100TSubstrate`, `AuditoryCortex` and `Claustrum`.

Together, these mean the archive as-shipped is not runnable on any
non-ARM64 machine, and even on ARM64 the top-level demo crashes.

## 5. What I changed (see CHANGES.md for detail)

Rather than delete real work, I added compatibility so the substrate at
least *runs*, and I made the entry surface honest.

- **Added `PredictiveMinicolumn`** to `nlp_engine/layers/predictive.py` —
  a real predictive-coding minicolumn with generative + recognition
  weights, sparsity gating, and precision-weighted Hebbian updates. This
  is what `dual_planner.py` was clearly written to call.

- **Patched `embodied_agent/cortex/snn_engine.py`** so the Rust `.so` is
  *optional*. It's now loaded only when the ELF architecture matches
  the current interpreter; otherwise a pure-numpy port of the LIF +
  Hebbian STDP kernel is used. Behaviour is preserved bit-for-bit
  where possible, and controllable with `HYPERION_FORCE_PY=1`.

- **Patched `embodied_agent/praxis/htm_predictor.py`** in the same way
  for the `htm_step` Rust function, with a numpy port.

- **Added a pure-Python `rust_backend_shim.py` shim** at the project
  root, implementing the 4 functions actually called anywhere in the
  codebase (`hash_neuron_id`, `reward_modulated_stdp`, `compute_fft`,
  `sparse_attention`). This unblocks `CainBrain`, `Virtual100TSubstrate`,
  `AuditoryCortex`, and `Claustrum`.

- **Added `cli.py`** — a single CLI that exposes all three
  personalities (`dual-brain`, `aurora`, `chat`) plus an `inspect`
  self-test that runs every major subsystem and reports what actually
  works. This replaces "hunt for the right script and read the source
  to figure out what flag it wants".

- **Added this file and `CHANGES.md`**, and rewrote the top-level
  `README.md` to be truthful about what the project is and isn't.

## 6. Post-patch status

After the changes, `python3 cli.py inspect` reports:

```
  dual_planner.py dual-brain                       OK
  embodied_agent embodied loop                   OK
  SNN MemristorLayer (rust or numpy fallback)  OK
  HTM sequence memory                          OK
  CainBrain (chat)                             OK
```

All five major subsystems now import and run on a stock x86_64 Python
3.10+ with only `numpy` installed. The architecture is still not an
AGI — see §1 — but it is now a *working* research substrate that a
successor engineer can iterate on without first fighting the
build system.

## 7. What would move it toward AGI (roughly, and honestly)

If a future engineer wants to push this codebase toward general
capability, the levers that would actually matter are:

1. **Replace the N-gram language head** in `CainBrain` with a real
   trained language model or, at minimum, a small transformer on the
   Wiktionary corpus this repo already knows how to ingest.
2. **Wire the three personalities into one control loop.** The
   `embodied_agent` global workspace is the natural integrator; feed
   `CainBrain`'s language head and `dual_planner.py`'s dual-brain planner
   into it as modalities.
3. **Ground the embeddings in tasks with reward.** The `VentralTegmentalArea`
   and the `active_inference` module exist but are barely exercised;
   giving them a curriculum (`embodied_agent/learning/curriculum.py` is
   the scaffold) with real reward signals is where actual learning
   would begin.
4. **Add scale.** Every "N trillion" number in this repo refers to
   *addressable* neurons, not *stored* parameters. Real behaviour
   requires actually storing and training the parameters.

None of that is done here. What *is* done: the substrate now runs,
imports cleanly, and can be inspected in one command.
