# LLMs vs biological brains vs what Hyperion actually does

This project's stance is that transformer-based LLMs are not the path to
general intelligence. That's a defensible research position. It is also
important, if you take that position, to be precise about what LLMs
and brains *actually* differ on, so the substrate you build instead
addresses real gaps rather than imagined ones. This doc is that
precision pass.

## 1. Representation

**LLM (GPT-style transformer):**
- Continuous vectors of ~4 k–15 k dimensions per token, stored as
  bf16/fp16/fp8.
- Dense: every dimension is nonzero every step.
- Same representation is used for input, computation, and output — one
  vector-space for everything.

**Human cortex:**
- Spiking activity: neurons fire discrete events at 0–200 Hz.
- Extremely sparse: ~1–5 % of neurons active at any instant in most
  cortical areas (compare: 100 % of an LLM's neurons are "active" in
  the sense of participating in matmul every forward pass).
- Population coding: information is in *which* small subset spikes and
  the relative timing, not in a dense magnitude vector.

**Hyperion:**
- `neurorust::Neuron` (Hodgkin-Huxley, Izhikevich, multi-compartment):
  models real membrane potential and ion-channel gating. This is on the
  brain side of the divide.
- `nlp_engine/layers/cain_brain.py`: uses dense Hebbian embeddings, not
  spikes. This is on the LLM side of the divide.
- The two substrates do not talk to each other in the shipped wiring.

## 2. Learning rule

**LLM:**
- Backpropagation through the entire network via the chain rule.
- Requires storing all forward activations to compute gradients.
- Global: a loss at the output modifies weights everywhere.
- Off-line: the model is frozen at deployment; new information does
  not update weights.

**Human cortex:**
- Local rules: STDP (weight change depends only on pre- and
  post-synaptic timing), homeostatic scaling, heterosynaptic
  competition, neuromodulator gating (dopamine, ACh, NE).
- No known biological mechanism implements backprop's exact gradient
  flow. There are candidate approximations (Feedback Alignment,
  predictive-coding-as-backprop, target propagation) but none has been
  demonstrated in vivo.
- Continuous / online: the brain never stops updating.

**Hyperion:**
- `MemristorLayer` uses Hebbian STDP with uint8 memristive saturation
  — local, no backprop, exactly the brain-side of this divide.
- Predictive-coding blocks in `embodied_agent/cortex/predictive_block.py`
  implement the Rao & Ballard 1999 update, which is a plausible
  local-Hebbian approximation to backprop through a hierarchy of
  predictions. Also brain-side.
- **What is missing:** neuromodulator gating of the STDP rule. Without
  dopamine-modulated STDP, the substrate cannot do reward-based
  learning. The empirical learning-test result (see
  `LEARNING_TEST_RESULTS.md`) illustrates this: STDP alone gives
  uniform LTP; supervised discrimination requires an *external* WTA
  supervisor. A real cortical column solves the same problem
  internally via GABAergic interneurons + neuromodulator gating.

## 3. Scale

**LLM (GPT-4-class):**
- ~1.8 T parameters (estimated), all stored, all touched per forward
  pass.
- Trained on ~10¹³ tokens.
- Energy: ~50 kJ per forward pass on datacenter hardware.

**Human cortex:**
- ~86 B neurons, ~10¹⁴–10¹⁵ synapses.
- Trained on ~10 s of continuous multimodal input for ~2·10⁹ s
  (a human lifetime). Estimated ~10¹⁷ synaptic events over a life.
- Energy: ~20 W total, ~3 W for cortex specifically.

**Hyperion:**
- Actual instantiated state: ~4096 neurons at any given step in
  `Virtual100TSubstrate`, ~1 M–10 M synaptic weights in the STDP layer.
- Trained on: none, out of the box. `assembly_book.txt` is a ~1 KB
  toy corpus.
- Energy per step: microseconds of a single core.

The "100 trillion" number in the original README was an address-space
parameter, not a stored-parameter count. The archive's own rewritten
README says this. It bears repeating because the gap here between
Hyperion (10⁶ real synapses) and either an LLM (10¹² parameters) or a
brain (10¹⁴ synapses) is 6–8 orders of magnitude. No architectural
insight can close a 6-OoM gap; it has to come from scale.

## 4. Temporal dynamics

**LLM:**
- Discrete time = token index. No milliseconds. No absolute time.
- Attention is O(n²) in context length, all pairs computed jointly.
- No refractory period, no leak, no membrane time constant.

**Human cortex:**
- Real time in milliseconds. Neurons have ~10 ms membrane time
  constants, ~1–2 ms refractory periods.
- Long-range communication uses axons with 0.5–120 m/s conduction
  velocity — inter-hemispheric delays are 20–30 ms.
- Rhythms: theta (4–8 Hz), gamma (30–80 Hz), ripples (150–250 Hz)
  organise computation in time.

**Hyperion:**
- The neurorust HH and Izhikevich models integrate at dt = 0.01 ms
  with real membrane dynamics. Correctly biology-side.
- No inter-region conduction delays are modelled — everything happens
  in the same tick. That's a real limitation if you want to reproduce
  gamma/theta coupling.

## 5. What Hyperion's approach implicitly bets on

If you accept the project's premise — transformers are not the path —
then the bet the codebase is making is:

1. **Sparse spike-based representation** will scale better than dense
   vectors when you get to embodied real-time tasks. There is genuine
   evidence for this in neuromorphic hardware (Loihi, SpiNNaker, Akida).
2. **Local learning rules (STDP, predictive coding)** will eventually
   match or beat backprop when combined with the right lateral-inhibition
   and neuromodulator scaffolding. The evidence here is much weaker —
   at present no local-rule system has matched backprop on any hard task.
3. **Biological fidelity buys generalisation.** The claim is that
   because Hodgkin-Huxley + STDP + predictive coding is what the brain
   does, and the brain generalises, a system built the same way will
   generalise. This is not a mainstream ML view. It is a defensible
   research bet, but it is a *bet*, not a proof.

## 6. Where the v32 low-level work fits

The AVX2 kernels added in this revision (`asm_kernels_avx2.rs`) make
the neurorust simulator faster:

- AXPY (used by cable-equation integration): **~2.6–3.1× speedup** for
  L2-resident vectors.
- CSR spmv (used by connectome synaptic-current propagation):
  **~1.7–2.0× speedup** across realistic connectome sizes.

These are honest micro-optimisations to the simulation substrate. They
do not close any of the six-orders-of-magnitude gap discussed in §3.
They make it possible to run bigger simulations in the same wall-clock
budget, which is genuinely useful, and they close a real gap between
what LLVM was auto-vectorising and what the hardware can do.

They do not make the substrate think. Nothing in this repo does.
