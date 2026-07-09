# Biological → Kernel mapping (design rationale + citations)

This document justifies every subsystem in `agikernel` against the primary
neuroscience literature. Nothing here is speculative — the *architecture* is
consensus systems neuroscience; only the claim that this architecture, when
faithfully implemented, is *sufficient* for AGI is a hypothesis.

## Cortex — the canonical microcircuit

- **Anatomical basis**: Mountcastle, V. B. (1978). *An organizing principle
  for cerebral function: the unit module and the distributed system*.
- **Predictive coding**: Bastos, A. M. et al. (2012). Canonical microcircuits
  for predictive coding. *Neuron* 76.
- **Layer-specific role of L5 in output**: Larkum, M. E. (2013). A cellular
  mechanism for cortical associations. *Trends in Neurosciences* 36.
- **Wiring statistics (~5 % density)**: Braitenberg, V. & Schüz, A. (1998).
  *Cortex: statistics and geometry of neuronal connectivity.*

## Thalamus — attentional gate

- **Sensory relay**: Sherman, S. M. & Guillery, R. W. (2006). *Exploring the
  Thalamus and its Role in Cortical Function*.
- **TRN attentional gating**: Halassa, M. M. & Kastner, S. (2017). Thalamic
  functions in distributed cognitive control. *Nature Neuroscience* 20.
- **L6 → thalamus feedback**: Olsen, S. R. et al. (2012). Gain control by
  layer 6 in cortical circuits of vision. *Nature* 483.

## Hippocampus — episodic memory

- **Pattern separation (DG) & completion (CA3)**: Marr, D. (1971). Simple
  memory: a theory for archicortex.
- **Modern DG evidence**: Yassa, M. A. & Stark, C. E. L. (2011). Pattern
  separation in the hippocampus. *Trends in Neurosciences* 34.
- **Sharp-wave-ripple replay**: Buzsáki, G. (2015). Hippocampal sharp
  wave-ripple: A cognitive biomarker. *Hippocampus* 25.
- **Complementary Learning Systems**: McClelland, McNaughton & O'Reilly
  (1995). *Psychological Review* 102.

## Basal ganglia — action selection

- **Selection-by-disinhibition**: Redgrave, P., Prescott, T. J. & Gurney, K.
  (1999). The basal ganglia: a vertebrate solution to the selection problem.
  *Neuroscience* 89.
- **D1/D2 model**: Frank, M. J. (2005). Dynamic dopamine modulation in the
  basal ganglia: A neurocomputational account. *J. Cognitive Neuroscience* 17.

## Cerebellum — forward model / fine timing

- **Marr–Albus theory**: Marr, D. (1969); Albus, J. S. (1971).
- **Climbing-fiber error signal**: Ito, M. (1984). *The Cerebellum and Neural
  Control*.

## Brainstem / hypothalamus — homeostasis

- **Priority of homeostasis**: Damasio, A. (1994). *Descartes' Error*.
- **Sleep pressure as a drive**: Borbély, A. A. (1982). A two process model
  of sleep regulation. *Human Neurobiology* 1.

## Neuromodulators — global gain and plasticity

- **Dopamine = RPE**: Schultz, W., Dayan, P. & Montague, P. R. (1997). A
  neural substrate of prediction and reward. *Science* 275.
- **Noradrenaline = uncertainty / gain**: Aston-Jones, G. & Cohen, J. D.
  (2005). *Annual Review of Neuroscience* 28. Yu, A. J. & Dayan, P. (2005).
  *Neuron* 46.
- **Serotonin = patience / aversion**: Doya, K. (2002). Metalearning and
  neuromodulation. *Neural Networks* 15.
- **Acetylcholine = attention / SNR**: Sarter, M., Hasselmo, M. E., Bruno,
  J. P. & Givens, B. (2005). *Brain Research Reviews* 48.

## Global oscillations — the scheduler

- **Nested gamma/theta rhythms**: Buzsáki, G. (2006). *Rhythms of the Brain*.
- **Global-Workspace ignition**: Dehaene, S. (2011). *Consciousness and the
  Brain*.
