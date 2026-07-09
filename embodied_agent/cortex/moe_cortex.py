"""
Sparse Mixture-of-Experts cortex (the "scale" answer)
=====================================================

The user observed:  "86 B neurons ≈ 100 T parameter model".

Naively that's true — a dense 100 T model would require ~400 TB just for
weights at fp32.  But the brain is **not** dense: at any moment only
~1-2 % of neurons fire, and the connectome is itself sparse (~10⁴
synapses per neuron, so 1.5e14 synapses total, not 7.4e21).

This file implements the v3 cortex as a routed MoE:

  * `n_experts` cortical columns per layer  (the "reservoir")
  * `experts_active` top-k columns activated per token (the "firing rate")

`config.estimated_params()` reports both the total reservoir capacity and
the *active* compute per step — the latter is what dominates FLOPs and
the former is what dominates representational reach.  At the "brain"
tier:

       capacity ≈ 1.5 × 10¹⁴  (synapses)
       active   ≈ 1 × 10¹²    (~1% firing)
       ratio    ≈ 100×        (matches biology)

so a 100T-parameter Aurora costs roughly the same per-token compute as a
1T dense model.  That's how you get brain-equivalent capacity with
trainable compute budgets.

The reference implementation here is numpy and runs single-process.  The
production runtime would shard experts across devices; the routing
arithmetic is identical.
"""
from __future__ import annotations
import numpy as np
from .predictive_block import PredictiveBlock
from .synaptic_pager import SynapticPager


class _Router:
    """Top-k gating with load-balancing aux loss tracking."""
    USAGE_DECAY_ALPHA = 0.02

    def __init__(self, dim: int, n_experts: int, k: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((n_experts, dim)).astype(np.float32) * np.float32(1.0 / np.sqrt(dim))
        self.k = k
        self.n_experts = n_experts
        self.usage = np.zeros(n_experts, dtype=np.float32)
        self.lifetime_hits = np.zeros(n_experts, dtype=np.int64)

    def route(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = self.W @ z
        logits = np.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        idx = np.argpartition(logits, -self.k)[-self.k:]
        # softmax over the chosen k only (sparse routing).
        s = logits[idx]
        s = np.clip(s, -20, 20)
        s = s - s.max()
        p = np.exp(s)
        p_sum = p.sum()
        if p_sum == 0 or np.isnan(p_sum):
            p = np.ones_like(p) / len(p)
        else:
            p = p / p_sum
        # Track usage with homeostatic EMA decay.
        a = self.USAGE_DECAY_ALPHA
        self.usage *= (1.0 - a)
        self.usage[idx] += a * p
        self.lifetime_hits[idx] += 1
        return idx.astype(np.int64), p.astype(np.float32)

    def load_balance_penalty(self) -> float:
        """Coefficient-of-variation of usage; lower = better balance."""
        m = self.usage.mean() + 1e-8
        return float(self.usage.std() / m)

    def underused_indices(self, tau: float) -> np.ndarray:
        """Return indices of experts whose usage falls below threshold *tau*."""
        return np.where(self.usage < tau)[0]

    def drop_expert(self, idx: int):
        """Remove expert *idx* from the router (apoptosis)."""
        self.W = np.delete(self.W, idx, axis=0)
        self.usage = np.delete(self.usage, idx)
        self.lifetime_hits = np.delete(self.lifetime_hits, idx)
        self.n_experts -= 1


class MoEPredictiveLayer:
    def __init__(self, dim: int, n_experts: int, k: int, seed: int = 0):
        self.router = _Router(dim, n_experts, k, seed=seed)
        self.experts = [PredictiveBlock(dim, dim, seed=seed + i + 1)
                        for i in range(n_experts)]
        # Assign IDs for the pager
        for i, exp in enumerate(self.experts):
            exp.id = f"expert_{seed}_{i}"
            
        self.k = k
        self.pager = None # Will be set by stack
        self._last_idx = None
        self._last_p = None
        self._last_mix = None
        self._last_err = 0.0

    @property
    def W_gen(self) -> np.ndarray:
        """Diagnostic aggregate matching `PredictiveBlock.W_gen`.

        A MoE layer owns many expert matrices, not one matrix. Older
        tests and monitors inspect `layer.W_gen` to verify that extreme
        rewards/pain do not blow up cortex weights. Returning the mean
        awake expert matrix preserves that diagnostic contract without
        changing the routed expert path.
        """
        awake = [e.W_gen for e in self.experts if e.W_gen is not None]
        if not awake:
            return np.zeros_like(self.experts[0].state[:, None] @ self.experts[0].state[None, :])
        return np.mean(np.stack([w.astype(np.float32, copy=False) for w in awake]), axis=0)

    def step(self, bottom: np.ndarray, top=None, lr: float = 3e-4,
             reuse_routing: bool = False) -> tuple[np.ndarray, float]:
        if reuse_routing and self._last_idx is not None:
            idx, p = self._last_idx, self._last_p
        else:
            idx, p = self.router.route(bottom)
            self._last_idx = idx
            self._last_p = p

        out = np.zeros_like(bottom)
        err_total = 0.0
        
        # 1. Wake up selected experts if they were asleep
        if not reuse_routing and self.pager is not None:
            self.pager.tick()
            for j in idx:
                self.pager.touch(self.experts[j].id)
                self.experts[j].wake(self.pager)

        # 2. Execute active experts
        for j, w in zip(idx, p):
            s, e = self.experts[j].step(bottom, top=top, lr=lr)
            out += w * s
            err_total += w * e
            
        # 3. Hibernate inactive experts (if unused for > 50 steps)
        if not reuse_routing and self.pager is not None:
            for j, exp in enumerate(self.experts):
                if j not in idx:
                    if self.pager.should_hibernate(exp.id, max_idle=50):
                        exp.hibernate(self.pager)

        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.isnan(err_total) or np.isinf(err_total):
            err_total = 1.0

        if not reuse_routing:
            self._last_mix = out.copy()
        self._last_err = float(err_total)

        return out, float(err_total)

    def precision(self) -> float:
        return float(np.mean([e.precision for e in self.experts]))

    def most_confidently_wrong_expert(self, err_threshold: float) -> int | None:
        """Return index of the expert with highest precision * last_err above *err_threshold*."""
        worst_idx = None
        worst_score = err_threshold
        for i, exp in enumerate(self.experts):
            score = exp.precision * self._last_err
            if score > worst_score:
                worst_score = score
                worst_idx = i
        return worst_idx

    def free_energy(self) -> float:
        """Combined free energy: mean expert free energy + load-balance penalty."""
        expert_fe = float(np.mean([e.free_energy() for e in self.experts]))
        return expert_fe + 0.1 * self.router.load_balance_penalty()


class MoECortexStack:
    def __init__(self, latent_dim: int, n_layers: int, n_experts: int, k: int,
                 seed: int = 0):
        self.layers = [MoEPredictiveLayer(latent_dim, n_experts, k, seed=seed + 100 * i)
                       for i in range(n_layers)]
        
        self.pager = SynapticPager()
        for layer in self.layers:
            layer.pager = self.pager

    def forward(self, sensory_z: np.ndarray, lr: float = 3e-4) -> tuple[np.ndarray, float]:
        x = sensory_z
        total_err = 0.0
        for layer in self.layers:
            x, err = layer.step(x, top=None, lr=lr)
            total_err += err
        # top-down refinement
        for i in range(len(self.layers) - 2, -1, -1):
            top = self.layers[i + 1]
            # cheap top-down: predict from top's first expert state.
            top_state = top.experts[0].state
            self.layers[i].step(self.layers[i].experts[0].state.copy(),
                                top=top_state, lr=lr * 0.5,
                                reuse_routing=True)
        return x, total_err / max(1, len(self.layers))

    def imagine(self, z_top: np.ndarray) -> np.ndarray:
        x = z_top
        for layer in reversed(self.layers):
            x = layer.experts[0].predict(x)
        return x

    def precision_profile(self) -> list[float]:
        return [l.precision() for l in self.layers]

    def load_balance(self) -> list[float]:
        return [l.router.load_balance_penalty() for l in self.layers]

    def free_energy(self) -> float:
        """Mean free energy across all layers."""
        return float(np.mean([l.free_energy() for l in self.layers]))
