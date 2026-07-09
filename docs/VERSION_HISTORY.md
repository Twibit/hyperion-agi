# Hyperion version history

A one-page summary of what each iteration actually contributed. Keeps
the archive traceable when someone (including future-me) opens it cold.
Recent releases first.

| Version | Codename                       | Headline contribution                                                                                                      | Learning-test status                                                            |
|:-------:|:-------------------------------|:---------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|
| **v33** | `hyperion_agi_v33_cortex_wta`  | **Biological learning fix:** in-model WTA + homeostatic plasticity + three-factor STDP. AVX-512 & ARM NEON kernel paths.  | **PASS 100 %** on 4-class pattern separation **without external teacher**       |
| v32     | `hyperion_agi_v32_avx2_honest` | Packed AVX2 kernels (`vfmadd231pd`, `vpgatherdps`), scalar+prefetch CSR SPMV. **Honest rollback** of prior AGI marketing. | PASS only with external teacher WTA hack; naive teacher-bias fails at 25 %     |
| v31     | (implicit, packed into v32)    | Predictive-coding cortex stack, HTM sequence memory, cerebellum + hippocampus + basal ganglia sketches, chat loop.       | Untested for learning-capability at the substrate level.                        |
| ≤ v30   | earlier drafts                 | Claimed "100-trillion-parameter" AGI. Retracted in v32's README (see `README_ORIGINAL.md` for the old marketing text).   | No published test.                                                              |

## What changed *between* v32 and v33 in one paragraph

v32 was an *honest performance pass* — it made the numeric kernels
faster and documented that the naive learning rule fails at chance. It
did not fix the learning rule. v33 fixes it, by putting the biology
that was externalised into the test harness (winner-take-all
suppression of non-target rows) back inside the substrate where a real
cortical column has it. Three additions carry the fix: (1) hard/soft
WTA lateral inhibition, (2) homeostatic per-neuron threshold
adaptation, (3) heterogeneous weight init for symmetry breaking. With
those three the shipped `MemristorLayer` subclass learns 4 orthogonal
8-hot patterns at 100 % from unsupervised forward passes alone, and
does the same at 100 % with reward gating via three-factor STDP.

## What v33 explicitly did *not* try to do

- Make the system "AGI". WTA+STDP+homeostasis is well-known biology
  (Rumelhart & Zipser 1985; Turrigiano 2011; Frémaux & Gerstner 2016)
  and works at the scale of one small cortical column. General
  capability needs orders of magnitude more scale, curriculum,
  embodiment, and multi-region coordination — none of which v33
  attempts.
- Reintroduce "0 ms" or "no limits" framing. v32's honest labelling
  policy is preserved verbatim.
- Rewrite v32 files. All v32 code and docs are unchanged; v33
  additions live in new files (`snn_engine_v2.py`, new Rust modules,
  new test).
