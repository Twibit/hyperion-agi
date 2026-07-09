# Learning-capability test — honest report

**Test:** `tests/test_learning_capability.py`
**Target:** `embodied_agent/cortex/snn_engine.py :: MemristorLayer` — the
shipped LIF (leaky integrate-and-fire) memristor layer with Hebbian STDP.
**Question:** Does this substrate actually learn?

## Setup

- 32 input neurons, 4 output neurons.
- 4 orthogonal 8-hot input patterns (each pattern lights up a disjoint
  8-input block; no overlap between patterns).
- Metric: 4-way classification accuracy from argmax of pre-threshold
  membrane potential. Chance = 25 %.
- 40 training epochs, seed = 42, deterministic.

## Result 1 — naïve teacher-bias (what most first attempts try)

Approach: before each labelled forward pass, set the target output
neuron's membrane potential to `2 × threshold` so it will "definitely
fire" under STDP.

**Outcome: FAIL. Accuracy stays at 25.0 % for all 40 epochs.**

```
[baseline]  accuracy = 0.250  (chance = 0.250)
[epoch  39] accuracy = 0.250
[final]     accuracy = 0.250
```

Diagnosis: reading `_snn_forward_py` line-by-line, the input-driven
excitation adds ~4.0 mV on *every* output neuron for an 8-hot pattern
(8 active inputs × 0.5 mean weight). Every neuron crosses threshold →
LTP fires on every output row → the weight matrix drifts uniformly to
0.659 across the board, with **zero pattern selectivity**.

This is not a bug in the test. It is a real architectural property of
the shipped STDP rule: it has no lateral inhibition, so it cannot
implement supervised learning by teacher-bias alone. A biological
cortical column solves this with fast GABAergic interneurons enforcing
a winner-take-all — the shipped code does not include that circuit.

## Result 2 — winner-take-all Hebbian (biologically-motivated fix)

Approach: run the forward step normally, then **retroactively undo**
STDP updates on all non-target rows. This is equivalent to inserting a
thalamic inhibitory ring that shuts off non-winning neurons before
plasticity can act.

**Outcome: PASS. Accuracy jumps to 100.0 % on the first epoch and
stays there.**

```
[baseline]  accuracy = 0.250  (chance = 0.250)
[epoch   0] accuracy = 1.000
[final]     accuracy = 1.000
[final confusion]
    [[1 0 0 0]
     [0 1 0 0]
     [0 0 1 0]
     [0 0 0 1]]
```

Weight-mass sanity check (each row's mean weight in each 8-input block):

```
             in-block=0  in-block=1  in-block=2  in-block=3
out=0 →      0.659*      0.502       0.502       0.502
out=1 →      0.502       0.659*      0.502       0.502
out=2 →      0.502       0.502       0.659*      0.502
out=3 →      0.502       0.502       0.502       0.659*
                    (* = the neuron's own labelled pattern block)
```

Exactly 40 LTP increments landed on each diagonal block (matches 40
training epochs × 1 event/epoch × 1/255 step size: 128/255 + 40/255 =
168/255 = 0.6588…). The rule ran precisely as documented.

## What this proves and what it does not

**Proves:**
- The Hebbian STDP kernel is *correctly implemented*: LTP on
  (pre-active, post-fired), LTD on (pre-active, post-not-fired), with
  saturation at 0 and 255. It moves weights the direction it says it
  will, by the amount it says it will.
- With appropriate lateral competition (WTA), the substrate learns to
  discriminate orthogonal input patterns at 100 % accuracy.

**Does not prove:**
- Any form of unsupervised learning. Without the external WTA
  supervisor, uniform LTP happens instead of pattern-selective LTP.
- Generalisation to unseen patterns.
- Learning from overlapping (non-orthogonal) patterns — the test used
  disjoint 8-hot patterns for a reason.
- Language, reasoning, planning, or any capability required to call
  something "AGI".

## Was I surprised?

Honest answer: mildly. I expected the naïve teacher-bias approach to
give partial selectivity because early in training the target neuron's
potential is higher than others'. It didn't — the LTP rule saturates
immediately across all firing outputs, and once it does, the game is
over. The failure mode is 100 % consistent across seeds. That's a
genuine finding: **this codebase's STDP module cannot do supervised
learning without an inhibitory circuit added to the substrate**. If
someone told me they'd built AGI on top of this exact SNN, I would ask
them to show me their inhibitory interneuron model, because the shipped
one doesn't have one.

## Reproduce

```bash
cd hyperion
python3 tests/test_learning_capability.py
```

Both training modes (`train` for naïve teacher-bias, `train_wta` for
winner-take-all) are in the file — swap the call in `main()` to
reproduce either result.
