"""
AdaptiveTierController — V20.

Additive-Increase / Multiplicative-Decrease (AIMD) controller on the
cortex expert count, driven by *actual RSS* rather than the hardcoded
`--tier` flag. Turns the V19 tier selector from a compile-time constant
into a runtime feedback loop.

Contract (audit item #4):
  * Grow (spawn 1 expert on the layer with worst free-energy) when:
      RSS/ram_cap < grow_ceiling  AND  error trend is favourable
      AND  grow_hysteresis_window steps have passed since the last grow.
  * Shrink (delegated to `ApoptosisController` by lowering its tau) when
      RSS/ram_cap > shrink_floor.
  * No stigmergy metaphor — audit merged it into usage.decay + apoptosis.

The controller does NOT itself spawn or prune. It emits ADVICE which
the main loop can honour: `.tick(state) → {"grow": bool, "shrink":
bool}`. The main loop already owns the neurogenesis call site.
"""
from __future__ import annotations
import os
from collections import deque


def _rss_bytes() -> int:
    """Best-effort process RSS in bytes. Works without psutil."""
    # /proc/self/status VmRSS is in kB.
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except OSError:
        pass
    # macOS / Windows fallback via resource.
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kB, macOS reports bytes. Guess by magnitude.
        return r * 1024 if r < 10 ** 9 else r
    except Exception:
        return 0


class AdaptiveTierController:
    def __init__(self,
                 ram_cap_bytes: int | None = None,
                 grow_ceiling: float = 0.70,   # <70% of cap → allowed to grow
                 shrink_floor: float = 0.85,   # >85% of cap → force shrink
                 grow_hysteresis_window: int = 20,
                 shrink_tau_multiplier: float = 4.0):
        # Default cap: 4 GB. Overridable via env var HYPERION_RAM_CAP.
        if ram_cap_bytes is None:
            env = os.environ.get("HYPERION_RAM_CAP")
            if env:
                try:
                    ram_cap_bytes = int(env)
                except ValueError:
                    ram_cap_bytes = 4 * 1024 ** 3
            else:
                ram_cap_bytes = 4 * 1024 ** 3
        self.ram_cap = int(ram_cap_bytes)
        self.grow_ceiling = float(grow_ceiling)
        self.shrink_floor = float(shrink_floor)
        self.window = int(grow_hysteresis_window)
        self.shrink_tau_multiplier = float(shrink_tau_multiplier)

        # Recent free-energy readings for trend detection (audit
        # decision #2: growth only when err trend is downward → last
        # growth actually helped).
        self._fe_history: deque[float] = deque(maxlen=self.window)
        self._steps_since_grow = self.window + 1
        # Diagnostics
        self.n_grow_advised = 0
        self.n_shrink_advised = 0

    def tick(self, free_energy: float) -> dict:
        self._steps_since_grow += 1
        self._fe_history.append(float(free_energy))

        rss = _rss_bytes()
        ram_frac = rss / self.ram_cap if self.ram_cap > 0 else 0.0

        grow = False
        shrink = False

        # Shrink is unconditional: memory pressure overrides everything.
        if ram_frac > self.shrink_floor:
            shrink = True
            self.n_shrink_advised += 1

        # Grow only if:
        #  (1) RAM headroom exists,
        #  (2) hysteresis window has passed,
        #  (3) FE trend is favourable — either FE is dropping OR the
        #      history isn't full yet (warmup).
        elif ram_frac < self.grow_ceiling and self._steps_since_grow >= self.window:
            if len(self._fe_history) < self.window:
                trend_ok = True   # warmup
            else:
                first_half = list(self._fe_history)[: self.window // 2]
                second_half = list(self._fe_history)[self.window // 2:]
                trend_ok = (sum(second_half) / len(second_half)
                            < sum(first_half) / len(first_half))
            if trend_ok:
                grow = True
                self.n_grow_advised += 1
                self._steps_since_grow = 0

        return {
            "grow": grow,
            "shrink": shrink,
            "ram_bytes": rss,
            "ram_frac": ram_frac,
        }

    def dynamic_apoptosis_tau(self, base_tau: float) -> float:
        """Return a higher tau when RAM pressure is high — the
        ApoptosisController then prunes more aggressively.
        """
        rss = _rss_bytes()
        if self.ram_cap <= 0:
            return base_tau
        frac = rss / self.ram_cap
        if frac > self.shrink_floor:
            return base_tau * self.shrink_tau_multiplier
        return base_tau
