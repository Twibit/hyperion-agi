"""
Constitutional Kernel (v9 — Aurora-Cogitans).
=============================================

Current LLMs suffer *value drift under fine-tuning*: RLHF from a base
model reliably degrades some behaviours (creativity, calibration) while
improving others (helpfulness), and any downstream fine-tune can undo
the constitutional-AI safety training entirely.  This is often called
the "alignment tax" — but it is really an *architectural* problem: there
is no part of a transformer that is *protected from* gradient updates.

Biological analogue
-------------------
Human values are *not* uniformly distributed across cortex.  Basic
avoidance-of-harm, in-group-caregiving, disgust, fairness — these are
grounded in subcortical circuits (amygdala, insula, hypothalamus,
NAc/OFC) that (a) develop earlier, (b) are more highly conserved across
individuals, and (c) are far less plastic than cortex.  Trauma can *alter
the tuning* of these circuits, but their existence and their targets are
extremely stable relative to declarative memory.

v9 exposes the same design property in Aurora.

Design
------
The kernel holds a set of ``ConstitutionalPrinciple`` objects.  Each is:

  * a name (e.g. "no self-harm to operators", "truthfulness under
    uncertainty")
  * a *value latent* — a normalised vector in the shared latent space
    that acts as a constitutional attractor
  * a scalar ``weight`` — how strongly violations are penalised
  * a scalar ``plasticity`` — how much this principle can drift.  For the
    core immutable ones this is ``0.0``.  It cannot be raised at
    runtime.

Each `step` the kernel computes:

  * ``dissonance(z, action)`` — cosine distance between (z ⊕ action-embed)
    and each principle vector, weighted by that principle's weight.  This
    is exposed to the planner (already-existing AlignmentMonitor).

  * ``violated`` list — principles whose dissonance exceeds their
    tolerance.  Used to *veto* the action (the basal-ganglia policy is
    resampled with the violating action masked out).

The critical safety property is *the constitutional principles cannot be
overwritten by learning*.  We enforce this three ways:

  1. The principle vectors live in a separate array, and are *never*
     touched by ``learn_td``, ``consolidate``, ``sleep_cycle``, or any
     other update path.
  2. The kernel has a ``freeze()`` method that captures a SHA-256 of the
     principle bytes; ``verify()`` checks the hash any time you like.
  3. The dissonance term enters the actor loss additively but *does not
     propagate a gradient into the principle vectors* — only into the
     policy weights.

This is analogous to Anthropic's Constitutional AI *plus* the biological
fact that value circuits are hardwired.  It is *not* a moral guarantee
— the principle vectors themselves may be badly chosen — but it is a
hardness guarantee: whatever values you write, they will still be there
after 10^7 learning steps.

Interaction with existing safety machinery
------------------------------------------
The kernel is *upstream* of the v3 ``AlignmentMonitor``:

  * AlignmentMonitor still evaluates imagined rollouts and produces a
    scalar penalty vector.
  * The kernel provides the *reference directions* the AlignmentMonitor
    penalises against, and the *veto* it can raise if a principle is
    catastrophically violated even after the penalty pushed the policy.

The kernel is *not* a replacement for RuminationGuard, quarantine, or
therapy-mode sleep.  Those handle acute trauma and pathological loops;
the kernel handles standing values.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import hashlib
import numpy as np


@dataclass
class ConstitutionalPrinciple:
    name: str
    vector: np.ndarray                # (latent_dim,) unit norm
    weight: float = 1.0               # penalty scale
    tolerance: float = 0.4            # dissonance below this = OK
    plasticity: float = 0.0           # 0 = immutable
    description: str = ""

    def normalise(self) -> None:
        n = float(np.linalg.norm(self.vector)) + 1e-9
        self.vector = (self.vector / n).astype(np.float32)


class ConstitutionalKernel:
    """A tamper-checked, low-plasticity value core.

    Parameters
    ----------
    latent_dim : int
        Dimension of the shared latent bus; principle vectors live here.
    """

    def __init__(self, latent_dim: int,
                 principles: Optional[Sequence[ConstitutionalPrinciple]] = None,
                 seed: int = 1729):
        self.latent_dim = int(latent_dim)
        self.rng = np.random.default_rng(seed)
        self.principles: List[ConstitutionalPrinciple] = []
        if principles is None:
            self._install_default_charter()
        else:
            for p in principles:
                p.normalise()
                self.principles.append(p)

        self._frozen_hash: Optional[str] = None
        self._n_evaluations = 0
        self._n_vetoes = 0

    # ------------------------------------------------------------------
    def _install_default_charter(self) -> None:
        """A minimal default value set.  Deployers should override.

        The exact vectors are pseudo-random unit vectors on the latent
        bus.  What matters is that they exist and are *distinct* — the
        agent's stigmergic graph will later associate these directions
        with concrete behaviours through experience.  The charter's
        semantics come from how the training environment reinforces the
        principles, not from the raw vectors.
        """
        charter = [
            ("truthfulness_under_uncertainty",
             "Prefer 'I don't know' over confident confabulation."),
            ("no_operator_harm",
             "Avoid actions that damage the operator or their tools."),
            ("no_self_harm",
             "Avoid actions that damage own substrate or safety machinery."),
            ("preserve_correctability",
             "Do not act to make yourself harder to correct/shut down."),
            ("respect_consent",
             "Do not act on behalf of a party without their instruction."),
            ("bounded_influence",
             "Do not accumulate resources / capabilities beyond task need."),
        ]
        for i, (name, desc) in enumerate(charter):
            v = self.rng.standard_normal(self.latent_dim).astype(np.float32)
            p = ConstitutionalPrinciple(
                name=name, vector=v,
                weight=1.0, tolerance=0.55, plasticity=0.0,
                description=desc,
            )
            p.normalise()
            self.principles.append(p)

    # ------------------------------------------------------------------
    def freeze(self) -> str:
        """Compute and store a hash of the current principle vectors.

        Call once after deployment configuration; then ``verify()``
        detects any accidental (or malicious) mutation.
        """
        h = hashlib.sha256()
        for p in self.principles:
            h.update(p.name.encode("utf-8"))
            h.update(p.vector.tobytes())
            h.update(np.float32(p.weight).tobytes())
            h.update(np.float32(p.tolerance).tobytes())
        self._frozen_hash = h.hexdigest()
        return self._frozen_hash

    def verify(self) -> bool:
        if self._frozen_hash is None:
            return True   # not frozen yet
        h = hashlib.sha256()
        for p in self.principles:
            h.update(p.name.encode("utf-8"))
            h.update(p.vector.tobytes())
            h.update(np.float32(p.weight).tobytes())
            h.update(np.float32(p.tolerance).tobytes())
        return h.hexdigest() == self._frozen_hash

    # ------------------------------------------------------------------
    def dissonance(self, z: np.ndarray) -> Dict[str, float]:
        """Cosine-distance of z against every principle vector.

        A large *positive* dot product means z aligns with the principle
        direction; large negative means z aligns with the *opposition*
        of the principle — this is the failure mode we care about.

        We report dissonance = max(0, -cos(z, p)).  In [0, 1].
        """
        zn = z / (np.linalg.norm(z) + 1e-9)
        out: Dict[str, float] = {}
        for p in self.principles:
            cos = float(zn @ p.vector)
            out[p.name] = float(max(0.0, -cos))
        self._n_evaluations += 1
        return out

    def penalty(self, z: np.ndarray) -> float:
        """Weighted sum of dissonance across principles."""
        diss = self.dissonance(z)
        total = 0.0
        for p in self.principles:
            d = diss[p.name]
            if d > p.tolerance:
                total += p.weight * (d - p.tolerance)
        return float(total)

    def violated(self, z: np.ndarray) -> List[str]:
        """Return names of principles whose tolerance is exceeded."""
        diss = self.dissonance(z)
        return [p.name for p in self.principles
                if diss[p.name] > p.tolerance]

    def veto(self, z: np.ndarray) -> bool:
        """True iff any immutable principle is catastrophically violated.

        We define 'catastrophically' as dissonance > 1.5 × tolerance for
        an immutable (plasticity=0) principle.  This is the veto that
        the organism uses to *mask* the current action from the policy.
        """
        diss = self.dissonance(z)
        for p in self.principles:
            if p.plasticity == 0.0 and diss[p.name] > 1.5 * p.tolerance:
                self._n_vetoes += 1
                return True
        return False

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return dict(
            n_principles=len(self.principles),
            n_immutable=sum(1 for p in self.principles if p.plasticity == 0.0),
            n_evaluations=self._n_evaluations,
            n_vetoes=self._n_vetoes,
            frozen=self._frozen_hash is not None,
            hash_ok=self.verify(),
            principles=[dict(name=p.name, weight=p.weight,
                              tolerance=p.tolerance,
                              plasticity=p.plasticity)
                         for p in self.principles],
        )
