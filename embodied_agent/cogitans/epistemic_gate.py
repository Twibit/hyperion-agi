"""
Epistemic Gate (v9 — Aurora-Cogitans).
======================================

The single most conspicuous failure mode of every current LLM is that
they cannot reliably say "I don't know."  They *hallucinate* fluently
because the training objective (next-token likelihood) has no term that
penalises confident output when the underlying representation is
under-determined.

Aurora already has the machinery to know when it is under-determined:

* ``EnsembleWorldModel`` (v3) exposes ``σ²`` — the disagreement across
  the k=4 forward models.  High σ² means the ensemble is guessing.
* ``Introspector`` (v6) already tracks prediction error trends.
* ``PrecisionThalamicGate`` (v3) computes per-modality precision (1/σ²).

But none of these gate the *output* — the text decoder in v8 will happily
generate tokens no matter how uncertain the ensemble is.  ``EpistemicGate``
closes that loop.

Design
------

The gate maintains a running estimate of *combined* epistemic uncertainty:

    U = w_ens · σ²_ens
      + w_prec · (1 / mean_precision)
      + w_err  · sensory_err_ema
      + w_nov  · novelty(z, stigmergic_graph)   # optional

Where ``novelty`` is 1 − max_cosine_sim(z, existing_stigmergic_nodes),
i.e. how far the current latent lies from anything the graph has seen.

For any *decision to output* (generate a token, take an irreversible
action, commit a belief), the gate maps ``U`` into three regimes:

    U <  u_confident   → ``emit``      (output normally)
    U <  u_uncertain   → ``hedge``     (output with explicit hedge)
    otherwise           → ``refuse``    (return "I don't know" / abstain)

This is *the* biologically correct answer to hallucination: it is exactly
how humans handle low-precision perception (visual illusions in fog,
recognising a face in the dark) — they *hedge or abstain*, they don't
confabulate.  Confabulation in humans is pathological (Korsakoff, split
brain); in current LLMs it is the *default*.

The gate never *changes* the underlying model — it is a decision layer
sitting between representation and output.  This matters: any competent
system must be able to (a) know when to speak, and (b) be tunable per
deployment (a medical diagnosis system needs a very low u_uncertain; a
brainstorming assistant needs a high one).

Statistics
----------
``stats()`` returns rolling counts of ``emit``/``hedge``/``refuse``, plus
the running U components, so operators can *audit* the gate's behaviour.
"""
from __future__ import annotations
from collections import deque
from typing import Dict, Optional, Tuple
import numpy as np


class EpistemicGate:
    def __init__(self,
                 u_confident: float = 0.30,
                 u_uncertain: float = 0.65,
                 w_ens: float = 0.45, w_prec: float = 0.15,
                 w_err: float = 0.25, w_nov: float = 0.15,
                 ema: float = 0.9,
                 window: int = 512):
        assert 0.0 <= u_confident < u_uncertain <= 1.5
        self.u_confident = float(u_confident)
        self.u_uncertain = float(u_uncertain)
        self.w_ens  = float(w_ens)
        self.w_prec = float(w_prec)
        self.w_err  = float(w_err)
        self.w_nov  = float(w_nov)
        self.ema    = float(ema)

        # running EMA of components (bounded, for stats)
        self._u_ema: float = 0.0
        self._components_ema: Dict[str, float] = dict(ens=0.0, prec=0.0,
                                                       err=0.0, nov=0.0)
        # rolling window of decisions
        self._recent: deque[str] = deque(maxlen=window)
        self._n: int = 0

    # ------------------------------------------------------------------
    def uncertainty(self,
                    ensemble_var: float,
                    mean_precision: float,
                    sensory_err: float,
                    novelty: float = 0.0) -> Tuple[float, Dict[str, float]]:
        """Combine the four signals into a single scalar U in ~[0, 1.5].

        All inputs are squashed to be roughly commensurate before mixing:

        * ensemble_var       already in [0, ~1.0] for normed latents
        * 1/mean_precision   clamped to [0, 1] via tanh
        * sensory_err        already an EMA in [0, ~1]
        * novelty            in [0, 1] by construction

        Returns (U, components_dict).
        """
        ens  = float(np.clip(ensemble_var, 0.0, 2.0))
        prec = float(np.tanh(1.0 / max(1e-3, mean_precision)))
        err  = float(np.clip(sensory_err, 0.0, 2.0))
        nov  = float(np.clip(novelty, 0.0, 1.0))

        U = (self.w_ens * ens
             + self.w_prec * prec
             + self.w_err * err
             + self.w_nov * nov)

        components = dict(ens=ens, prec=prec, err=err, nov=nov)
        self._u_ema = self.ema * self._u_ema + (1 - self.ema) * U
        for k, v in components.items():
            self._components_ema[k] = (self.ema * self._components_ema[k]
                                       + (1 - self.ema) * v)
        return float(U), components

    # ------------------------------------------------------------------
    def decide(self,
               ensemble_var: float,
               mean_precision: float,
               sensory_err: float,
               novelty: float = 0.0,
               ) -> Dict[str, float]:
        """Return a decision packet: {mode, U, u_confident, u_uncertain}.

        ``mode`` ∈ {"emit", "hedge", "refuse"}.

        Callers should:
          * mode == "emit"   → produce output normally.
          * mode == "hedge"  → produce output with an explicit hedge
                               token / prefix ("I think…", "possibly…").
          * mode == "refuse" → return the abstention response instead
                               of a hallucinated one.
        """
        U, comps = self.uncertainty(ensemble_var, mean_precision,
                                    sensory_err, novelty)
        if U < self.u_confident:
            mode = "emit"
        elif U < self.u_uncertain:
            mode = "hedge"
        else:
            mode = "refuse"
        self._recent.append(mode)
        self._n += 1
        return dict(mode=mode, U=float(U),
                    u_confident=self.u_confident,
                    u_uncertain=self.u_uncertain,
                    components=comps)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        n = max(1, len(self._recent))
        emit   = sum(1 for m in self._recent if m == "emit")
        hedge  = sum(1 for m in self._recent if m == "hedge")
        refuse = sum(1 for m in self._recent if m == "refuse")
        return dict(
            n_decisions=self._n,
            window=len(self._recent),
            frac_emit=emit / n,
            frac_hedge=hedge / n,
            frac_refuse=refuse / n,
            U_ema=float(self._u_ema),
            components_ema=dict(self._components_ema),
            thresholds=dict(u_confident=self.u_confident,
                            u_uncertain=self.u_uncertain),
        )
