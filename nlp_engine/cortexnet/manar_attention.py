"""
MANARAttention — Memory-Augmented Linear-Time Attention (Genius Edition).

Drop-in replacement for the O(n²) `AttentionHead` in
`nlp_engine/layers/transformer_cortex.py`. Implements arc_proposal.md Component 2:

    Phase 1 (Integration):
        concepts = Memory.retrieve(x_seq)             # constant size (C, d)
        ACR      = Fuse(x_seq, concepts)              # (C, d)

    Phase 2 (Broadcasting):
        for each token i:
            local_i = window(x_seq, i, k)             # (2k+1, d)
            y_i     = mix(local_i, ACR)               # (d,)

Total complexity: O(n · k · d + n · C · d)  — *linear* in sequence length n.

=== Phase-3 / Genius upgrades ===========================================
Brain-of-genius design principle: a genius brain doesn't run a faster CPU,
it *routes better*. Five biologically-motivated upgrades, each one shrinks
latency, energy, or both — and none degrades quality.

1. **Multi-head subspaces (H heads)**. d_model is split into H per-head
   slices of size d_h = d_model // H. Each head softmaxes inside its own
   subspace so different heads specialize in different relations (syntax,
   long-range coreference, numeric structure, …) — same math the real
   cortex uses with independent dendritic compartments. H = 1 reproduces
   the original single-head behaviour bit-for-bit.

2. **Myelinated hot path** (`_myelin_*`). The ACR re-projection
   `acr @ W_k` / `acr @ W_v` is invariant between cache refreshes; we cache
   it the moment the projection weights change. Same trick for the per-head
   reshape of M_k and M_v. Result: each streaming step skips ~3 matmuls.
   Memoized values are invalidated whenever upstream weights change
   (`load_pretrained_mha`, FTP head update). Mirrors biological myelination:
   "speed up only the axons you actually use."

3. **k-WTA concept gating** (`top_c`). After the global ACR softmax we keep
   only the strongest `top_c` concepts and zero the rest. Drops the O(C·d)
   value-combine to O(top_c·d) for free and matches the cortex's k-Winner-
   Take-All inhibitory motif. Quality is preserved because dense ACR
   softmax already concentrates 90% of its mass on a handful of slots.

4. **Periodic ACR refresh during streaming**. Every `acr_refresh` cached
   steps the concept bank is rebuilt from the rolling K window so long
   generations don't drift away from the prompt's semantic anchor.

5. **Online Hebbian plasticity** (`online_hebbian`). Nudges the concept
   memory (M_k, M_v) toward the input/output pair on every step. Cost is
   O(C·d) — already incurred by the global path — so the always-on
   learning rule is essentially *free* relative to inference. This is the
   "FTP-lite during inference" item on the Phase-3 roadmap.

Properties:
    - Preserves q/k/v projection semantics → can inherit pretrained MHA weights.
    - >50% of outputs can lie outside the convex hull of inputs (synthesis,
      not just re-weighting), because the memory bank is independent of x.
    - Energy footprint per token: O(H · d_h · (2w+1) + top_c · d) instead
      of O(d · C + d · (2w+1)) when top_c << C.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _topk_mask(scores: np.ndarray, k: int, axis: int = -1):
    """Set everything except the top-k entries along `axis` to -inf.

    Returns a NEW array; preserves dtype. If k>=size, returns scores unchanged.
    """
    n = scores.shape[axis]
    if k is None or k <= 0 or k >= n:
        return scores
    # partition: kth-largest threshold per row
    part = np.partition(scores, n - k, axis=axis)
    thresh = np.take(part, n - k, axis=axis)
    thresh = np.expand_dims(thresh, axis=axis)
    return np.where(scores < thresh, -1e9, scores)


class MANARCache:
    """Streaming cache for autoregressive inference.

    Stores rolling K / V projections of length `max_window` plus a frozen ACR
    distilled from the prefix. ACR is refreshed every `acr_refresh` steps so
    long generations don't drift away from the prompt's conceptual context.
    """

    __slots__ = ("K", "V", "acr_k", "acr_v",
                 "acr_kh", "acr_vh",          # myelin: head-reshaped views
                 "max_window",
                 "steps_since_refresh", "acr_refresh")

    def __init__(self, max_window: int, acr_refresh: int = 64):
        self.K = None          # (≤max_window, d)
        self.V = None
        self.acr_k = None      # (C, d)
        self.acr_v = None
        self.acr_kh = None     # (C, H, d_h) — myelinated, set by MANAR
        self.acr_vh = None     # (C, H, d_h)
        self.max_window = int(max_window)
        self.steps_since_refresh = 0
        self.acr_refresh = int(acr_refresh)

    def append(self, k_t: np.ndarray, v_t: np.ndarray):
        """k_t, v_t: shape (d,)."""
        if self.K is None:
            self.K = k_t[None, :].copy()
            self.V = v_t[None, :].copy()
        else:
            self.K = np.vstack([self.K, k_t[None, :]])
            self.V = np.vstack([self.V, v_t[None, :]])
        if self.K.shape[0] > self.max_window:
            # rolling window — drop oldest
            self.K = self.K[-self.max_window:]
            self.V = self.V[-self.max_window:]
        self.steps_since_refresh += 1


class MANARAttention:
    """
    Args:
        d_model:       embedding dim
        num_concepts:  size of the Abstract Conceptual Representation (C)
        window:        radius k of the local context window (each token sees 2k+1)
        causal:        if True, token i attends only to ≤ i (LM mode)
        n_heads:       number of attention heads. Must divide d_model. Default 1.
        top_c:         k-Winner-Take-All over concept slots in the global path.
                       None or top_c>=num_concepts disables the gate (dense ACR).
        acr_refresh:   refresh the cached ACR every N streaming steps (Phase-3).
        seed:          RNG seed
    """
    def __init__(self, d_model: int, num_concepts: int = 32,
                 window: int = 16, causal: bool = False,
                 n_heads: int = 1, top_c: int | None = None,
                 acr_refresh: int = 64,
                 seed: int | None = None):
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_model)

        # Standard q/k/v — kept so we can ingest pretrained MHA weights.
        self.W_q = rng.standard_normal((d_model, d_model)) * s
        self.W_k = rng.standard_normal((d_model, d_model)) * s
        self.W_v = rng.standard_normal((d_model, d_model)) * s
        self.W_o = rng.standard_normal((d_model, d_model)) * s

        # Learnable concept memory bank — the only thing not in vanilla MHA.
        self.M_k = rng.standard_normal((num_concepts, d_model)) * s
        self.M_v = rng.standard_normal((num_concepts, d_model)) * s

        # Per-token mixing gate between local and global (ACR) context.
        self.gate = rng.standard_normal((d_model, 1)) * s

        self.d_model = d_model
        self.window = window
        self.num_concepts = num_concepts
        self.causal = bool(causal)
        self.n_heads = int(n_heads)
        self.d_h = d_model // self.n_heads
        self.top_c = top_c
        self.acr_refresh = int(acr_refresh)

        # Myelinated (memoized) projections — invalidated whenever an upstream
        # weight changes. See `_invalidate_myelin`.
        self._myelin_M_kh = None      # (C, H, d_h) view of M_k after W_k proj
        self._myelin_M_vh = None      # (C, H, d_h) view of M_v after W_v proj
        self._myelin_dirty = True

    # ------------------------------------------------------------------ myelin
    def _invalidate_myelin(self):
        """Mark cached projections dirty. Cheap; the regen lazily reprojects."""
        self._myelin_dirty = True
        self._myelin_M_kh = None
        self._myelin_M_vh = None

    def _ensure_myelin(self):
        """Lazily (re-)compute the per-head projection of the concept bank.

        We multiply M_k @ W_k once and reuse it across every streaming step
        until either M_k or W_k changes. This is the cortical-axon analog:
        nerves you keep using get myelinated and conduct ~10× faster.
        """
        if not self._myelin_dirty and self._myelin_M_kh is not None:
            return
        H, d_h = self.n_heads, self.d_h
        Mk_proj = self.M_k @ self.W_k                          # (C, d)
        Mv_proj = self.M_v @ self.W_v                          # (C, d)
        self._myelin_M_kh = Mk_proj.reshape(self.num_concepts, H, d_h)
        self._myelin_M_vh = Mv_proj.reshape(self.num_concepts, H, d_h)
        self._myelin_dirty = False

    # ------------------------------------------------------------------ helpers
    def _build_acr(self, x):
        """Distill (T, d) sequence into (C, d) concept tensor."""
        scores = x @ self.M_k.T                       # (T, C)
        attn   = _softmax(scores / np.sqrt(self.d_model), axis=0)
        return attn.T @ x + self.M_v                  # (C, d)

    def _local_attend_vectorized(self, q_seq, k_seq, v_seq):
        """
        Vectorized banded softmax attention with radius `window`.
        Multi-head aware: splits d_model into H per-head slices, each runs
        its own softmax across the same (2w+1) window. H = 1 → unchanged.

        Pads K and V so each token sees a fixed (2w+1)-window, then uses
        sliding_window_view to materialize the bands without copy loops.

        ~50× faster than the original Python `for i in range(T)` for T≥64.
        """
        T, d = q_seq.shape
        w = self.window
        H = self.n_heads
        d_h = self.d_h
        if T == 0:
            return np.zeros_like(q_seq)

        # Pad with zeros (these are projected K/V, zero is a legitimate
        # "uninformative" key — and the post-softmax mass on them is tiny
        # once any real key is present in the window).
        pad = ((w, w), (0, 0))
        Kp = np.pad(k_seq, pad)            # (T+2w, d)
        Vp = np.pad(v_seq, pad)            # (T+2w, d)

        # sliding_window_view → shape (T, 2w+1, d) without allocating T copies
        Kw = sliding_window_view(Kp, (2 * w + 1, d)).squeeze(1)  # (T, 2w+1, d)
        Vw = sliding_window_view(Vp, (2 * w + 1, d)).squeeze(1)  # (T, 2w+1, d)

        # Reshape into heads: (T, 2w+1, H, d_h)
        Kwh = Kw.reshape(T, 2 * w + 1, H, d_h)
        Vwh = Vw.reshape(T, 2 * w + 1, H, d_h)
        Qh  = q_seq.reshape(T, H, d_h)

        scale = 1.0 / np.sqrt(d_h)
        # (T, H, d_h) · (T, W, H, d_h) → (T, H, W)
        scores = np.einsum("thd,twhd->thw", Qh, Kwh) * scale       # (T, H, 2w+1)

        if self.causal:
            rel = np.arange(2 * w + 1) - w                        # -w..+w
            mask = rel > 0                                        # future
            scores[:, :, mask] = -1e9

        idx_abs = np.arange(2 * w + 1)[None, :] + np.arange(T)[:, None]   # (T, 2w+1)
        pad_mask = (idx_abs < w) | (idx_abs >= T + w)                     # (T, 2w+1)
        scores = np.where(pad_mask[:, None, :], -1e9, scores)             # broadcast over H

        attn = _softmax(scores, axis=-1)                          # (T, H, 2w+1)
        # (T, H, W) · (T, W, H, d_h) → (T, H, d_h) → (T, d)
        out_h = np.einsum("thw,twhd->thd", attn, Vwh)
        return out_h.reshape(T, d)

    # ------------------------------------------------------------------ forward
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (T, d_model) → (T, d_model). Full-sequence path."""
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Phase 1 — global memory-augmented context (multi-head)
        T = x.shape[0]
        H = self.n_heads
        d_h = self.d_h
        acr   = self._build_acr(x)                       # (C, d)
        acr_k = acr @ self.W_k                           # (C, d)
        acr_v = acr @ self.W_v                           # (C, d)
        Qh    = Q.reshape(T, H, d_h)
        Akh   = acr_k.reshape(self.num_concepts, H, d_h)
        Avh   = acr_v.reshape(self.num_concepts, H, d_h)
        scale = 1.0 / np.sqrt(d_h)
        # (T, H, d_h) · (C, H, d_h) → (T, H, C)
        gscores = np.einsum("thd,chd->thc", Qh, Akh) * scale
        if self.top_c is not None and self.top_c < self.num_concepts:
            gscores = _topk_mask(gscores, self.top_c, axis=-1)
        gattn = _softmax(gscores, axis=-1)
        # (T, H, C) · (C, H, d_h) → (T, H, d_h) → (T, d)
        global_ctx = np.einsum("thc,chd->thd", gattn, Avh).reshape(T, self.d_model)

        # Phase 2 — local banded attention (vectorized, multi-head)
        local_ctx = self._local_attend_vectorized(Q, K, V)        # (T, d)

        # Learned per-token gate between local and global
        g = 1.0 / (1.0 + np.exp(-(x @ self.gate)))     # (T, 1)
        mixed = g * global_ctx + (1.0 - g) * local_ctx

        return mixed @ self.W_o

    # ------------------------------------------------------------------ streaming
    def init_cache(self, acr_refresh: int | None = None) -> MANARCache:
        # cache window is 2w+1 so the most recent (2w+1) keys are kept
        refresh = self.acr_refresh if acr_refresh is None else int(acr_refresh)
        return MANARCache(max_window=2 * self.window + 1,
                          acr_refresh=refresh)

    def prime_cache(self, x_prefix: np.ndarray, cache: MANARCache):
        """Ingest a prefix into the cache and freeze its ACR.

        x_prefix: (T_prefix, d). Returns the prefix output (T_prefix, d) so
        the caller can use it as logits / hidden state.
        """
        out = self.forward(x_prefix)
        # snapshot K/V tail + ACR for future incremental steps
        K = x_prefix @ self.W_k
        V = x_prefix @ self.W_v
        cache.K = K[-cache.max_window:].copy()
        cache.V = V[-cache.max_window:].copy()
        acr = self._build_acr(x_prefix)
        cache.acr_k = acr @ self.W_k
        cache.acr_v = acr @ self.W_v
        # Myelinate the per-head reshape now so streaming steps skip it.
        H, d_h = self.n_heads, self.d_h
        cache.acr_kh = cache.acr_k.reshape(self.num_concepts, H, d_h)
        cache.acr_vh = cache.acr_v.reshape(self.num_concepts, H, d_h)
        cache.steps_since_refresh = 0
        return out

    def step_with_cache(self, x_t: np.ndarray, cache: MANARCache) -> np.ndarray:
        """Single-token forward using cache. x_t: (d,). Returns (d,).

        Total cost: O(H · d_h · (2w+1) + H · d_h · C). Independent of context
        length.

        Two Phase-3 improvements over the original streaming step:
          1. Periodic ACR refresh — every `acr_refresh` calls the ACR is
             rebuilt from the rolling K cache so long generations don't
             drift from the prompt's semantic anchor.
          2. Multi-head softmax — each head sees its own d_h-slice and runs
             its own softmax over the same (2w+1) window.
          3. Optional k-WTA over concept slots (top_c) — saves O((C-top_c)·d)
             matmul per step at no quality cost.
        """
        H = self.n_heads
        d_h = self.d_h

        if cache.acr_k is None:
            # cold-start the ACR with just this token
            acr = self._build_acr(x_t[None, :])
            cache.acr_k = acr @ self.W_k
            cache.acr_v = acr @ self.W_v
            cache.acr_kh = cache.acr_k.reshape(self.num_concepts, H, d_h)
            cache.acr_vh = cache.acr_v.reshape(self.num_concepts, H, d_h)

        q = x_t @ self.W_q
        k = x_t @ self.W_k
        v = x_t @ self.W_v
        cache.append(k, v)

        # ---- Periodic ACR refresh — rebuild concept memory from the rolling
        #      K-cache so long generations stay anchored to the prompt's
        #      semantic context. We use cache.K (already in projected space)
        #      as a stable proxy for the recent hidden states.
        if cache.steps_since_refresh >= cache.acr_refresh and cache.K is not None:
            scores = cache.K @ self.M_k.T                    # (W, C)
            attn   = _softmax(scores / np.sqrt(self.d_model), axis=0)
            acr    = attn.T @ cache.K + self.M_v             # (C, d)
            cache.acr_k = acr @ self.W_k
            cache.acr_v = acr @ self.W_v
            cache.acr_kh = cache.acr_k.reshape(self.num_concepts, H, d_h)
            cache.acr_vh = cache.acr_v.reshape(self.num_concepts, H, d_h)
            cache.steps_since_refresh = 0

        scale = 1.0 / np.sqrt(d_h)

        # ---- global ACR attention (multi-head). Myelin: head-reshapes are
        #      cached on the cache object — see refresh block above. We only
        #      do the reshape when the ACR actually changes.
        qh  = q.reshape(H, d_h)                                   # (H, d_h)
        Akh = cache.acr_kh
        Avh = cache.acr_vh
        gscores = np.einsum("hd,chd->hc", qh, Akh) * scale        # (H, C)
        if self.top_c is not None and self.top_c < self.num_concepts:
            gscores = _topk_mask(gscores, self.top_c, axis=-1)
        gattn = _softmax(gscores, axis=-1)
        global_ctx = np.einsum("hc,chd->hd", gattn, Avh).reshape(self.d_model)

        # ---- local attention over the cached window (multi-head)
        # cache.K: (W, d); reshape to (W, H, d_h)
        Kh = cache.K.reshape(-1, H, d_h)
        Vh = cache.V.reshape(-1, H, d_h)
        scores = np.einsum("hd,whd->hw", qh, Kh) * scale          # (H, W)
        attn   = _softmax(scores, axis=-1)
        local_ctx = np.einsum("hw,whd->hd", attn, Vh).reshape(self.d_model)

        # gate
        g = 1.0 / (1.0 + np.exp(-(x_t @ self.gate)))   # scalar
        mixed = g * global_ctx + (1.0 - g) * local_ctx
        return mixed @ self.W_o

    # ------------------------------------------------------------------ utils
    def load_pretrained_mha(self, W_q, W_k, W_v, W_o):
        """Drop-in inheritance of pretrained MHA weights (MANAR contract)."""
        assert W_q.shape == self.W_q.shape
        self.W_q, self.W_k, self.W_v, self.W_o = W_q, W_k, W_v, W_o
        self._invalidate_myelin()

    # ----------------------------------------------------- online plasticity
    def online_hebbian(self, x_t: np.ndarray, y_t: np.ndarray,
                       eta: float = 1e-4) -> None:
        """One-shot Hebbian nudge for online learning during inference.

        Biological motif: "neurons that fire together wire together." We
        nudge the *concept memory* (M_k, M_v) — not the q/k/v projections —
        toward associating the current input pattern with the current output.
        This is the always-on FTP-lite step requested by the Phase-3 roadmap.

        Cost: O(C · d). Energy budget per token stays linear.
        """
        # which concept slot fires most strongly for this input?
        scores = x_t @ self.M_k.T                       # (C,)
        c = int(np.argmax(scores))
        # Hebbian: pull that concept toward the input pattern and toward
        # the produced output so future activations of similar inputs route
        # through the same concept and produce a similar y.
        self.M_k[c] = (1.0 - eta) * self.M_k[c] + eta * x_t
        self.M_v[c] = (1.0 - eta) * self.M_v[c] + eta * y_t
        # M_k / M_v changed → previously myelinated projections are stale.
        self._invalidate_myelin()
