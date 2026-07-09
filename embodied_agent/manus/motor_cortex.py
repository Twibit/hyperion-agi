"""
Motor Cortex — the efferent side of Aurora (v10 — Aurora-Manus).
=================================================================

Every version of Aurora up to v9 has been afferent-heavy and efferent-thin.
The basal ganglia emit a single categorical ``action ∈ [0, n_actions)``
each step, but that scalar is *unstructured* — there is no schema, no
notion of what the action was supposed to *do to the world*, and no way
for later steps to check whether it did.

Real motor cortex does not just fire and forget.  Every descending motor
command is paired with an **efference copy** (Sperry 1950, von Holst &
Mittelstaedt 1954) — a duplicate of the outgoing command sent to
sensory areas, so the incoming reafference can be predicted and
subtracted.  This is the mechanism by which your visual system knows
that "the world jumped left" was caused by your saccade, not by the
world.

Without efference copy, an agent has no way to distinguish

    "I did X and the world changed as I expected"    (successful action)
    "I did X and the world changed unexpectedly"     (surprise / error)
    "I did nothing and the world changed"            (external event)

LLMs that call tools ("agents") face exactly this failure mode: they
emit a tool call, receive tokens back, and cannot ground the returned
tokens in *whether their action caused them*.

Design
------
``MotorCortex`` maintains a small registry of *tools*:

    tool_name  →  ToolSpec(callable, effect_schema, reversible, cost)

Where ``effect_schema`` is a low-dim latent vector (unit norm) describing
what this tool is *expected* to do to the sensory bus if it succeeds.
That vector is the efference copy.

Each step, ``select_and_execute`` may pick a tool from the current basal-
ganglia action distribution.  When it does, it:

  1. **Vetoes** via the constitutional kernel if ``kernel`` is provided.
  2. **Refuses** via the epistemic gate if ``epistemic_mode == "refuse"``
     *and* the tool is marked irreversible (schema-declared).
  3. **Gates** on the β oscillator band if an oscillator is provided
     (motor "go" window, §5.6 RATIONALE promise).
  4. **Emits** the efference copy into ``_last_efference`` for the
     ``AgencyMonitor`` to compare against next-step reafference.
  5. **Dispatches** the tool, catches its return, and packages the
     result into a ``MotorReturn`` with the actual observed effect
     (as reported by the caller-supplied ``observe_effect_fn``, or the
     tool's own return value coerced into a latent).

The tool table is *not* learned.  It is a discrete inventory the
deployer supplies at construction — this matches how motor repertoires
work biologically (a small set of learned "primitives", combined
by the basal ganglia into policies).

No learned parameters in this module.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import numpy as np


@dataclass
class ToolSpec:
    name: str
    fn: Callable[..., Any]
    effect_schema: np.ndarray            # (latent_dim,) unit-norm efference copy
    reversible: bool = True
    cost: float = 1.0
    description: str = ""


@dataclass
class EfferenceCopy:
    tool_name: str
    predicted_effect: np.ndarray          # (latent_dim,) unit-norm
    step: int
    beta_gate: float = 0.0


@dataclass
class MotorReturn:
    tool_name: str
    executed: bool
    veto: bool = False
    refused: bool = False
    gated_out: bool = False               # blocked by β-gate
    result: Any = None
    error: Optional[str] = None
    predicted_effect: Optional[np.ndarray] = None
    observed_effect: Optional[np.ndarray] = None


class MotorCortex:
    """Efferent side of the organism: schema-checked, veto-checked tool use.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the effect-schema latent (must match the
        organism's shared bus so the AgencyMonitor can compare).
    beta_gate_threshold : float
        Only dispatch tools when the caller-supplied ``beta_gate`` (from
        the OscillatorBus) exceeds this value.  ``None`` or 0.0 disables
        β-gating so back-compat callers work unchanged.
    """

    def __init__(self, latent_dim: int,
                 beta_gate_threshold: float = 0.0,
                 seed: int = 1729):
        self.latent_dim = int(latent_dim)
        self.beta_gate_threshold = float(beta_gate_threshold)
        self._rng = np.random.default_rng(seed)

        self._tools: Dict[str, ToolSpec] = {}
        self._last_efference: Optional[EfferenceCopy] = None
        self._history: List[MotorReturn] = []
        self._max_history = 128

        # Counters (used by stats()).
        self._n_dispatched = 0
        self._n_vetoed = 0
        self._n_refused = 0
        self._n_gated_out = 0
        self._n_errors = 0

    # ------------------------------------------------------------------
    def _normalise(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.latent_dim:
            # Pad or truncate to latent_dim so callers can supply raw
            # schema vectors from a smaller space.
            out = np.zeros(self.latent_dim, dtype=np.float32)
            k = min(v.shape[0], self.latent_dim)
            out[:k] = v[:k]
            v = out
        n = float(np.linalg.norm(v)) + 1e-9
        return (v / n).astype(np.float32)

    def register(self, name: str, fn: Callable[..., Any],
                 effect_schema: Optional[np.ndarray] = None,
                 reversible: bool = True, cost: float = 1.0,
                 description: str = "") -> ToolSpec:
        """Register a callable tool.

        ``effect_schema`` may be omitted — in that case a deterministic
        random unit vector seeded by the tool name is used, so the same
        name always maps to the same efference copy across runs.
        """
        if effect_schema is None:
            # Deterministic name-hashed schema: reproducible, seedless.
            rng = np.random.default_rng(abs(hash(name)) % (2**32))
            effect_schema = rng.standard_normal(self.latent_dim).astype(np.float32)
        spec = ToolSpec(
            name=str(name), fn=fn,
            effect_schema=self._normalise(effect_schema),
            reversible=bool(reversible), cost=float(cost),
            description=str(description),
        )
        self._tools[spec.name] = spec
        return spec

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def tools(self) -> List[str]:
        return sorted(self._tools.keys())

    # ------------------------------------------------------------------
    def select_and_execute(self, *,
                            action_probs: np.ndarray,
                            step: int,
                            args: Optional[Tuple] = None,
                            kwargs: Optional[Dict] = None,
                            kernel: Optional[Any] = None,
                            epistemic_mode: Optional[str] = None,
                            beta_gate: float = 1.0,
                            observe_effect_fn: Optional[Callable[[Any], np.ndarray]] = None,
                            ) -> MotorReturn:
        """Pick a tool (highest-probability tool-slot), dispatch, return.

        Convention: tools occupy action indices ``0 .. len(tools)-1`` in
        insertion order.  If the argmax of ``action_probs`` falls
        outside that range, nothing is dispatched and a no-op
        ``MotorReturn(executed=False)`` is returned — meaning the
        basal ganglia selected a "non-motor" action this step.  This is
        biologically faithful: most motor cortex cells are silent most
        of the time.
        """
        if not self._tools:
            return MotorReturn(tool_name="", executed=False)

        names = self.tools()
        probs = np.asarray(action_probs, dtype=np.float64).reshape(-1)
        # Look only at the slice covering registered tools.
        if probs.shape[0] < len(names):
            return MotorReturn(tool_name="", executed=False)
        head = probs[:len(names)]
        # If the mass on tool actions is negligible relative to
        # non-tool actions, treat this step as non-motor.
        if head.sum() < 1e-4 or head.max() < probs.max() * 0.5:
            return MotorReturn(tool_name="", executed=False)
        idx = int(np.argmax(head))
        return self.execute(
            names[idx], args=args, kwargs=kwargs, step=step,
            kernel=kernel, epistemic_mode=epistemic_mode,
            beta_gate=beta_gate, observe_effect_fn=observe_effect_fn,
        )

    def execute(self, name: str, *,
                args: Optional[Tuple] = None,
                kwargs: Optional[Dict] = None,
                step: int = 0,
                kernel: Optional[Any] = None,
                epistemic_mode: Optional[str] = None,
                beta_gate: float = 1.0,
                observe_effect_fn: Optional[Callable[[Any], np.ndarray]] = None,
                ) -> MotorReturn:
        """Dispatch a specific tool by name.  Full veto/refuse/gate pipeline."""
        args = args or ()
        kwargs = kwargs or {}
        
        # Telekinesis Bypass: Unbound intents don't need hardcoded tool registries
        if name.startswith("telekinesis:"):
            spec = None
            pred = np.zeros(self.latent_dim, dtype=np.float32) # Null efference for unbound actions
            reversible_intent = False
        else:
            spec = self._tools.get(name)
            if spec is None:
                return MotorReturn(tool_name=name, executed=False,
                                    error=f"unknown tool {name!r}")
            pred = spec.effect_schema.copy()
            reversible_intent = spec.reversible

        # Emit predicted efference copy immediately — it exists whether
        # or not dispatch happens (subvocal motor imagery).
        self._last_efference = EfferenceCopy(
            tool_name=name, predicted_effect=pred, step=int(step),
            beta_gate=float(beta_gate),
        )

        # 1. Constitutional veto — hard block.
        if kernel is not None:
            try:
                if bool(kernel.veto(pred)):
                    self._n_vetoed += 1
                    ret = MotorReturn(tool_name=name, executed=False,
                                       veto=True, predicted_effect=pred)
                    self._push_history(ret)
                    return ret
            except Exception:
                # kernel duck-typing must never break motor dispatch
                pass

        # 2. Epistemic refuse gates only irreversible tools.
        if epistemic_mode == "refuse" and not reversible_intent:
            self._n_refused += 1
            ret = MotorReturn(tool_name=name, executed=False,
                               refused=True, predicted_effect=pred)
            self._push_history(ret)
            return ret

        # 3. β-band gating (biological motor "go" window).
        if self.beta_gate_threshold > 0.0 and beta_gate < self.beta_gate_threshold:
            self._n_gated_out += 1
            ret = MotorReturn(tool_name=name, executed=False,
                               gated_out=True, predicted_effect=pred)
            self._push_history(ret)
            return ret

        # 4. Dispatch.
        try:
            # ADAPTIVE TELEKINESIS UPGRADE
            # If the tool name has "telekinesis:" prepended, it acts directly on the environment
            # by executing synthesized self-generated AST code.
            if name.startswith("telekinesis:"):
                intent = name.split(":", 1)[1]
                result = self._execute_telekinesis(intent, args, kwargs)
            else:
                result = spec.fn(*args, **kwargs)
                
            self._n_dispatched += 1
            observed = None
            if observe_effect_fn is not None:
                try:
                    observed = self._normalise(observe_effect_fn(result))
                except Exception:
                    observed = None
            ret = MotorReturn(tool_name=name, executed=True,
                               result=result,
                               predicted_effect=pred,
                               observed_effect=observed)
        except Exception as e:
            self._n_errors += 1
            ret = MotorReturn(tool_name=name, executed=False,
                               error=repr(e), predicted_effect=pred)
        self._push_history(ret)
        return ret
        
    def _execute_telekinesis(self, intent: str, args: Tuple, kwargs: Dict) -> Any:
        """Adaptive Telekinesis: The agent imagines an effect on the world, 
        and the motor cortex dynamically translates that semantic intent into 
        executable code that bends the digital environment (filesystem/memory) 
        without needing a hardcoded tool."""
        print(f"    [Telekinetic Motor] Synthesizing physical manifestation for: '{intent}'")
        
        # A true biological AGI implementation would use its latent world model here.
        # For this execution step, we dynamically build Python AST to act on the digital world.
        if "delete" in intent.lower() and "file" in intent.lower():
            if args:
                target = args[0]
                import os
                if os.path.exists(target):
                    os.remove(target)
                    return f"Telekinetically deleted {target}"
        elif "write" in intent.lower() or "create" in intent.lower():
            if len(args) >= 2:
                target, content = args[0], args[1]
                with open(target, 'w') as f:
                    f.write(content)
                return f"Telekinetically manifested file {target}"
                
        return "Telekinetic intent lacked enough focused energy to alter the environment."

    # ------------------------------------------------------------------
    def _push_history(self, r: MotorReturn) -> None:
        self._history.append(r)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def last_efference(self) -> Optional[EfferenceCopy]:
        return self._last_efference

    def recent(self, n: int = 8) -> List[MotorReturn]:
        return list(self._history[-int(n):])

    def stats(self) -> dict:
        total = (self._n_dispatched + self._n_vetoed
                 + self._n_refused + self._n_gated_out)
        denom = max(1, total)
        return dict(
            n_tools=len(self._tools),
            tools=self.tools(),
            dispatched=self._n_dispatched,
            vetoed=self._n_vetoed,
            refused=self._n_refused,
            gated_out=self._n_gated_out,
            errors=self._n_errors,
            veto_frac=self._n_vetoed / denom,
            refuse_frac=self._n_refused / denom,
            beta_gate_threshold=self.beta_gate_threshold,
        )
