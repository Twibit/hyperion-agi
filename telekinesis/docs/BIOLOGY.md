# Biological & Theoretical Grounding

This document explains, honestly, what parts of the `telekinesis` design are borrowed from **settled neuroscience** and what parts are borrowed from **contested parapsychology**. We use the vocabulary of the latter because it is evocative, but every load-bearing piece of the software architecture is grounded in the former.

---

## 1. The sensorimotor loop — settled science

### 1.1 Motor cortex population coding (Georgopoulos, 1986)
Apostolos Georgopoulos and colleagues recorded from primate primary motor cortex (M1) while monkeys reached in 8 directions. No individual neuron encoded direction, but a weighted **population vector** across ~200 cells predicted movement direction with high fidelity.

> Georgopoulos AP, Schwartz AB, Kettner RE. *Neuronal Population Coding of Movement Direction.* **Science** 233 (4771): 1416–1419 (1986). doi:10.1126/science.3749885

**Software mapping.** In `src/cortex/planner.rs`, an `Intent` is decomposed into a *bundle* of small `MotorCommand`s. No single command carries the whole plan; the plan lives in the aggregate — exactly as in M1.

### 1.2 Efference copy / *Reafferenzprinzip* (von Holst & Mittelstaedt, 1950)
When the brain issues a motor command, it also sends an internal copy (**Efferenzkopie**) to sensory areas. The predicted sensory consequences ("reafference") are subtracted from actual input, so the organism can tell "did I move, or did the world move?"

> von Holst E, Mittelstaedt H. *Das Reafferenzprinzip.* **Naturwissenschaften** 37: 464–476 (1950).

**Software mapping.** Every command executed by a `Limb` first produces an `EfferenceCopy` — a snapshot of the world state it *expects* to change (file contents, mtime, size). If actual and predicted diverge, the `Proprioceptor` raises a `ReafferenceMismatch` and rollback is triggered. This is bounded by an LRU so we cannot leak memory on high-throughput agents.

### 1.3 Proprioception (Sherrington, 1906)
Charles Sherrington named the "sixth sense" — muscle spindles and Golgi tendon organs continuously report limb position and force, closing the motor loop within ~30 ms.

> Sherrington CS. *The Integrative Action of the Nervous System.* Yale University Press (1906).

**Software mapping.** After every write the `Verifier` re-reads the affected inode's metadata (+ optional content hash) and reports the *actual* state back to the cortex. Without this, an agent is a hand with no feeling — it can move but cannot know if it succeeded.

### 1.4 Mirror neurons (Rizzolatti et al., 1990s)
Neurons in area F5 of macaque premotor cortex fire *both* when the animal performs an action and when it observes another performing the same action. This is the leading neural correlate of action understanding and imitation.

> Rizzolatti G, Craighero L. *The Mirror-Neuron System.* **Annual Review of Neuroscience** 27: 169–192 (2004).

**Software mapping.** `mirror::Observer` emits structured tracing events for every command *and* every observed reafference, so a supervisor process can "watch" a limb and reconstruct its intent. This is our telemetry substrate.

### 1.5 Libet's readiness potential (1983)
Benjamin Libet found a *Bereitschaftspotential* rising ~350–550 ms before conscious awareness of the intent to move — motor preparation precedes conscious decision.

> Libet B, Gleason CA, Wright EW, Pearl DK. *Time of Conscious Intention to Act in Relation to Onset of Cerebral Activity.* **Brain** 106: 623–642 (1983).

**Software mapping.** `cortex::Readiness` interposes a configurable *pre-motor gate* between intent-received and command-dispatched. Its default value is not 350 ms — that would be absurd for software — but it exists so that supervisory policies (rate-limits, sandbox checks, human-in-the-loop confirmations) can intervene during the "pre-conscious" window. This is how we get *safe agentic tool use* for free.

---

## 2. Psychokinesis — contested, disclosed honestly

The word "telekinesis" implies moving matter with the mind. In humans this is not established science. We include it here for architectural inspiration and because the failure modes of parapsychology are instructive — they tell us exactly what a rigorous engineering framework must avoid.

### 2.1 The PEAR laboratory (Jahn & Dunne, 1979–2007)
Princeton Engineering Anomalies Research ran ~28 years of intention-vs-RNG studies. They reported small but statistically significant deviations from chance, with cumulative effect sizes on the order of 10⁻⁴ per bit.

> Jahn RG, Dunne BJ. *The PEAR Proposition.* **Journal of Scientific Exploration** 19(2): 195–245 (2005). PMID 17560342.

### 2.2 Radin & Nelson meta-analysis (1989, updated 2006)
Aggregated 597 RNG-PK studies and reported an overall effect at many standard deviations from chance — but with substantial file-drawer, quality, and selection concerns.

> Radin DI, Nelson RD. *Evidence for Consciousness-Related Anomalies in Random Physical Systems.* **Foundations of Physics** 19(12): 1499–1514 (1989).

### 2.3 Counter-analyses
Kugel (2011) and Maier et al. (2018, Bayesian re-analysis) argue the effects vanish under stricter quality control and preregistration.

> Kugel W. *A Faulty PK Meta-Analysis.* **Journal of Parapsychology** (2011).
> Maier MA et al. *Intentional Observer Effects on Quantum Randomness: A Bayesian Analysis Reveals Evidence Against Micro-PK.* **Frontiers in Psychology** 9: 379 (2018). doi:10.3389/fpsyg.2018.00379

### 2.4 Bem (2011) precognition controversy
Daryl Bem's *"Feeling the Future"* (JPSP 100:407-425) reported precognitive effects; failed replications and the resulting methodological reforms (preregistration, higher statistical thresholds) reshaped psychology's replication practices — a *methodological win regardless of the underlying phenomenon*.

### 2.5 Honest stance
This crate uses "telekinesis" as a **naming convention**, not a scientific claim. The code makes no assumption that minds can move matter; it does assume that a well-designed cognitive architecture *for planning and safely executing distal effects* is worth building. The parapsychology literature is cited to (a) acknowledge where the naming comes from and (b) illustrate why we deliberately over-engineer the verification path.

---

## 3. What this buys us

| Real-world failure of AI tool use            | Fixed by biological analogue                          |
|----------------------------------------------|--------------------------------------------------------|
| Silent partial writes ("hallucinated success") | Efference copy + proprioceptive verification         |
| Race conditions across concurrent tool calls | Actor mesh + message ordering per limb                |
| Unauditable actions                          | Mirror-neuron observability layer                     |
| No safe-abort window before an irreversible op | Libet-style `Readiness` pre-motor gate               |
| Rigid, hand-coded tool dispatch              | Population-vector-style intent decomposition          |

The takeaway: **evolution has already spent 500 million years debugging the plan-act-verify loop.** Software agents that ignore that work re-invent every bug.
