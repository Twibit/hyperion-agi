# Hyperion — Changes Log (this session)

This document records every modification made to the Hyperion codebase in the
working session. It complements the repo's own `README.md`, `AGI_ASSESSMENT.md`
and `README_AGI_REWRITE.md`.

> **Honest status:** Hyperion is a bio-inspired *research substrate*, not an AGI.
> The changes below add real cognitive scaffolding (deliberation, relaxed limits,
> a training hook) and remove a fake/vestigial N-gram, but the underlying model is
> still small and largely untrained — see "Limitations" at the end.

---

## 1. Repo relocation
- Moved the project from
  `bio-ai-lab/dual_planner` into the workspace at
  `/root/Documents/Codex/2026-07-07/dual_planner`.

## 2. N-gram purged (repo-wide)
The earlier docs claimed the N-gram "was replaced," but the code still carried a
dead `DomainCortexBank`/`self.banks` N-gram substrate. It was never actually
defined (the imports already failed and were swallowed by `try/except`), so the
live chat path already used `PredictiveDecoder`. The purge makes that official
and removes the vestiges.

### Changes inside `nlp_engine/layers/cain_brain.py`
- Removed all `self.banks` / `DomainCortexBank` usage.
- `full_status()`: replaced the `ngram_banks` report with a `predictive_decoder` report.
- `learn_image()` / `listen()`: removed the vision/audio N-gram CMN banks; spikes
  are now stored directly into the Hippocampus.
- `speak()` EXECUTIVE and DMN branches: replaced the broken
  `super(CainBrain, self).speak(...)` N-gram fallbacks with the new
  `_generate_semantic()` helper (PredictiveDecoder + Entorhinal + Claustrum).
- Extracted `CainBrain._generate_semantic()` — the single generation path.
- **Fixed a pre-existing crash:** `self.hippocampus` was never initialized, so chat
  was already broken (`AttributeError` on `for mem in self.hippocampus`). Added
  `self.hippocampus = []`.
- Docstrings / comments updated (no more "N-Gram Probability Banks").

### Deleted standalone N-gram files
- `phoenix/train_phoenix.py`, `phoenix/train_phoenix_v7.py`
- `phoenix/train_gold_baseline.py`, `gold/train_gold.py`, `gold/train_gold_baseline.py`
- `core_scripts/universal_grammar.py`, `core_scripts/brocas_imprinting.py`,
  `core_scripts/hyperion_grammar_school.py`
- `simulate_main.py`, `chat_improved.py`
- `scripts/benchmark_learning.py`, `tests/test_cmn.py`
- `nlp_engine/experiments/polyglot_training.py`, `fluency_test.py`, `cyber_scenario.py`,
  `ai_war_games.py`, `advanced_cyber_scenario.py`, `neural_trader_v2.py`
- `cli.py` docstring updated ("driven by the PredictiveDecoder (no N-gram)").

## 3. Inner monologue — System-2 deliberation ("thinking")
New module `nlp_engine/layers/inner_monologue.py` implementing a chain-of-thought loop:
1. Retrieves related long-term facts from the knowledge graph.
2. Generates a candidate sub-thought via the generative decoder.
3. **Metacognitively critiques** it (relevance to context + novelty vs history).
4. **Self-corrects** by re-sampling weak steps with higher temperature.
5. Chains accepted sub-thoughts with a **sliding window** (bounded context, no loop).
6. A final synthesis pass returns the answer.

Wiring:
- `CainBrain.__init__` builds `self.monologue = InnerMonologue(self)`.
- New `CainBrain.think(prompt, max_steps=4, return_trace=False)` → returns the
  synthesized answer, or `(answer, trace)` when `return_trace=True`.
- New `CainBrain.bootstrap(sentences=..., corpus_path=..., limit=2000)` → ingests a
  real text corpus (Wiktionary / books) so the model can "understand fast".
- `main.py` chat loop now runs `brain.think(...)` as the System-2 draft and prints
  the visible reasoning trace.

## 4. Removed hardcoded limits that blocked learning
- `nlp_engine/layers/cellular_mechanics.py`: the "ATP DEBT CRASH — comatose" branch that
  **zeroed dopamine and halted all learning** is gone. Fatigue is now a *mild*
  dopamine throttle (`0.6` floor) that never stops learning.
- `cain_brain.learn()`: removed the hard coma path; learning continues under load.
- `embodied_agent/config.py`:
  - `metabolic_budget_frac`: `0.02 → 0.20` (the old 2% ceiling froze ~98% of
    regions at near-zero learning).
  - `metabolic_rest_gain`: `0.05 → 0.30` (resting regions now still learn).

## 5. Better generation
- `nlp_engine/layers/predictive_decoder.py` `decode_word()`: replaced greedy nearest
  neighbour with **temperature-scaled sampling** over cosine similarity, so the
  monologue explores instead of collapsing into an exact loop.

---

## How to run
```bash
cd dual_planner
python3 cli.py inspect      # self-test: 5 subsystems should report OK
python3 cli.py dual-brain   # right/left brain demo
python3 cli.py aurora --steps 50
python3 main.py                           # chat CLI (shows inner-monologue trace)
```
Optional — give it real comprehension:
```python
from nlp_engine.layers.cain_brain import CainBrain
b = CainBrain()
b.bootstrap(corpus_path="data/wiktionary_defs.txt")  # point at any line-delimited corpus
ans, trace = b.think("how does intelligence work", return_trace=True)
print(trace)
```

## Limitations (read before believing the hype)
- The substrate is still a tiny (64-dim), essentially **untrained** Hebbian net.
  The monologue is coherent *in structure* but output remains weak/jumbled until
  it is trained on real data via `bootstrap()` (or backed by a trained LM).
- The "100-trillion neuron" substrate (`nlp_engine/layers/scale86b`) is an **address
  space**, not stored parameters — only ~4096 neurons are ever instantiated.
- `README.md` / `AGI_ASSESSMENT.md` still describe the *old* N-gram state and the
  "not an AGI" verdict; they were left as-is. This file is the accurate record of
  what changed.

## Files changed (this session)
- `nlp_engine/layers/cain_brain.py` (N-gram removal, `_generate_semantic`, `think`,
  `bootstrap`, hippocampus init, softened fatigue)
- `nlp_engine/layers/inner_monologue.py` (NEW)
- `nlp_engine/layers/predictive_decoder.py` (sampled decode)
- `nlp_engine/layers/cellular_mechanics.py` (no more coma)
- `embodied_agent/config.py` (metabolic ceilings raised)
- `main.py` (inner-monologue draft + trace)
- `cli.py` (docstring)
- `README_CHANGES.md` (this file)
- Deleted: see section 2 list.
