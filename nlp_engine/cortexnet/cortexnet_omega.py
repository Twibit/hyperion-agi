"""
CortexNetOmega — Phase-2 top-level model.

This is the upgrade target for the README's "Known Problems":

    KV cache               → MANARAttention.step_with_cache    [DONE]
    Sleep consolidation    → HippocampalReplay.consolidate     [DONE]
    Neuromodulatory split  → NeuromodulatoryState (4 channels) [DONE]
    Hardware migration     → still NumPy on purpose (one repo, no Rust dep);
                              but every loop the user can hit is now vectorized
                              and the per-token streaming cost is O(k·d + C·d).

It also adds the things the spec didn't mention but a modern AI needs:

    - Universal prompt frontend (byte-level + role + modality + language tags)
    - Deliberative `think_then_speak()` API (chain-of-thought, internalised)
    - `generate()` with temperature / top-k / repetition penalty
    - Reservoir for hippocampal replay + a `sleep()` convenience method
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from .cortical_column import CorticalColumn
from .complement_pruner import ComplementPruner
from .ftp_trainer import FTPTrainer
from .universal_tokenizer import (
    UniversalTokenizer, PromptBlock, VOCAB_SIZE,
    TOK_PAD, TOK_BOS, TOK_EOS, TOK_ROLE_ASSISTANT, TOK_END_BLOCK,
)
from .hippocampal_replay import HippocampalReplay
from .thinking_loop import PrefrontalCortex, ThinkingLoop


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class CortexNetOmega:
    """Phase-2 cortical model.

    Args:
        vocab_size:    defaults to UniversalTokenizer.vocab_size = 320.
        d_model:       embedding dim.
        n_columns:     number of stacked CorticalColumns.
        num_concepts:  ACR memory bank size per column.
        window:        local-attention radius per column.
        causal:        treat as a language model (recommended for generation).
        seed:          deterministic init.
    """

    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 d_model: int = 128,
                 n_columns: int = 4,
                 num_concepts: int = 32,
                 window: int = 16,
                 causal: bool = True,
                 n_heads: int = 1,
                 top_c: int | None = None,
                 acr_refresh: int = 64,
                 online_hebbian: bool = False,
                 hebbian_eta: float = 1e-4,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.E = rng.standard_normal((vocab_size, d_model)) * (1.0 / np.sqrt(d_model))
        self.columns: List[CorticalColumn] = [
            CorticalColumn(d_model, num_concepts, window,
                           causal=causal, n_heads=n_heads, top_c=top_c,
                           acr_refresh=acr_refresh, seed=seed + 10 * i)
            for i in range(n_columns)
        ]
        self.online_hebbian = bool(online_hebbian)
        self.hebbian_eta = float(hebbian_eta)
        self.W_out = rng.standard_normal((d_model, vocab_size)) * (1.0 / np.sqrt(d_model))

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pruner = ComplementPruner()
        self.tokenizer = UniversalTokenizer()
        self.hippocampus = HippocampalReplay(capacity=4096)
        self.ftp_head = FTPTrainer([d_model, d_model, vocab_size],
                                   activations=["tanh", "id"], seed=seed)

    # ------------------------------------------------------------------ embed
    def embed(self, ids: np.ndarray) -> np.ndarray:
        return self.E[ids]

    # ------------------------------------------------------------------ forward
    def forward(self, ids: np.ndarray) -> np.ndarray:
        """Full-sequence forward (training / scoring path)."""
        h = self.embed(ids)
        for col in self.columns:
            h = col.forward(h) + h
        return h @ self.W_out

    def predict(self, ids: np.ndarray) -> np.ndarray:
        return _softmax(self.forward(ids), axis=-1)

    # ------------------------------------------------------------------ encode
    def encode_prompt(self,
                      prompt: Union[str, Sequence[PromptBlock]]) -> np.ndarray:
        """Universal prompt encoder.

        Accepts:
            - A plain string. Modality is auto-detected.
            - A list of PromptBlock(text, role, modality, lang).

        Returns a 1-D int64 array of token ids ready for `generate()`.
        """
        return self.tokenizer.encode(prompt, add_bos=True, add_eos=False)

    # ------------------------------------------------------------------ generate
    def generate(self,
                 prompt: Union[str, np.ndarray, Sequence[PromptBlock]],
                 *,
                 max_new_tokens: int = 64,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 repetition_penalty: float = 1.1,
                 stop_on_eos: bool = True,
                 seed: Optional[int] = None) -> str:
        """Universal autoregressive generation with streaming KV cache.

        Cost per token: O((2w+1)·d + C·d)  — independent of context length.
        """
        rng = np.random.default_rng(seed)
        if isinstance(prompt, np.ndarray):
            prompt_ids = prompt.astype(np.int64)
        else:
            prompt_ids = self.encode_prompt(prompt)

        # Mark the start of the assistant turn so the model knows it's
        # supposed to speak now.
        prompt_ids = np.concatenate([prompt_ids,
                                     np.array([TOK_ROLE_ASSISTANT], dtype=np.int64)])

        # Phase 1: prime each column's MANAR cache from the prompt
        caches = [col.init_cache() for col in self.columns]
        h = self.embed(prompt_ids)
        for ci, col in enumerate(self.columns):
            h = col.prime_cache(h, caches[ci]) + h

        # Phase 2: stream tokens
        out_ids: List[int] = []
        recently_seen: List[int] = list(prompt_ids[-32:].tolist())
        last_id = int(prompt_ids[-1])

        for _ in range(max_new_tokens):
            x_t = self.E[last_id]
            # Per-column streaming step. If online Hebbian is enabled we also
            # nudge each column's concept memory toward the (input → output)
            # association at this token — always-on plasticity, no extra pass.
            for ci, col in enumerate(self.columns):
                pre = x_t
                x_t = col.step_with_cache(x_t, caches[ci]) + x_t
                if self.online_hebbian:
                    col.l23_attn.online_hebbian(pre, x_t, eta=self.hebbian_eta)
            logits = x_t @ self.W_out

            # Repetition penalty
            if repetition_penalty != 1.0 and recently_seen:
                penalize = np.unique(np.array(recently_seen, dtype=np.int64))
                logits[penalize] /= repetition_penalty

            # Forbid PAD; allow EOS/control naturally
            logits[TOK_PAD] = -1e9
            logits[TOK_BOS] = -1e9

            next_id = self._sample(logits, temperature=temperature,
                                   top_k=top_k, rng=rng)

            if stop_on_eos and (next_id == TOK_EOS or next_id == TOK_END_BLOCK):
                break

            out_ids.append(next_id)
            recently_seen.append(next_id)
            if len(recently_seen) > 64:
                recently_seen.pop(0)
            last_id = next_id

        return self.tokenizer.decode(out_ids)

    # ------------------------------------------------------------------ thinking
    def think_then_speak(self,
                         prompt: Union[str, np.ndarray, Sequence[PromptBlock]],
                         *,
                         max_thoughts: int = 24,
                         max_new_tokens: int = 64,
                         temperature: float = 0.8,
                         top_k: int = 50,
                         seed: Optional[int] = None,
                         return_trace: bool = False):
        """Deliberation-then-answer pipeline (chain-of-thought, internalised).

        1. Encode prompt.
        2. Run the deliberative thinking loop — high ACh, low DA — to grow
           the concept memory of each column without leaking scratch tokens
           to the user.
        3. Generate the final answer at normal temperature.
        """
        rng_seed = seed if seed is not None else int(np.random.SeedSequence().entropy & 0xFFFFFFFF)
        rng = np.random.default_rng(rng_seed)

        if isinstance(prompt, np.ndarray):
            prompt_ids = prompt.astype(np.int64)
        else:
            prompt_ids = self.encode_prompt(prompt)

        trace = deliberative_think(self, self.tokenizer, prompt_ids,
                                   max_thoughts=max_thoughts,
                                   temperature=max(0.5, temperature),
                                   rng=rng)

        answer = self.generate(prompt_ids,
                               max_new_tokens=max_new_tokens,
                               temperature=temperature,
                               top_k=top_k,
                               seed=rng_seed + 1)

        if return_trace:
            return {
                "answer": answer,
                "thought": self.tokenizer.decode(trace.thought_ids),
                "n_thought_steps": trace.n_steps,
            }
        return answer

    # ------------------------------------------------------------------ sampling
    @staticmethod
    def _sample(logits: np.ndarray, *,
                temperature: float, top_k: int,
                rng: np.random.Generator) -> int:
        if temperature <= 1e-6:
            return int(np.argmax(logits))
        x = logits / temperature
        if top_k and top_k < x.shape[0]:
            kth = np.partition(x, -top_k)[-top_k]
            x = np.where(x < kth, -1e9, x)
        p = _softmax(x, axis=-1)
        return int(rng.choice(p.shape[0], p=p))

    # ------------------------------------------------------------------ training
    def train_step(self, ids: np.ndarray,
                   target_ids: Optional[np.ndarray] = None,
                   *, lr: float = 5e-3,
                   record_for_replay: bool = True,
                   salience: float = 1.0) -> float:
        """One FTP training step.

        `ids` is the input id sequence. `target_ids` (next-token by default)
        becomes a one-hot target for FTP. Returns the pre-update loss.
        """
        if target_ids is None:
            target_ids = np.roll(ids, -1)
            target_ids[-1] = ids[-1]
        h = self.embed(ids)
        Y = np.zeros((ids.shape[0], self.vocab_size), dtype=np.float32)
        Y[np.arange(ids.shape[0]), target_ids] = 1.0
        loss = self.ftp_head.train_step(h, Y, lr=lr, eta=1.0)
        # Also update the readout head from the FTP output layer so the
        # generative path benefits. We project the last FTP layer's W back
        # into W_out by EMA — cheap, no extra gradients.
        last_W = self.ftp_head.layers[-1].W
        # last_W: (d_model, vocab_size) — same shape as self.W_out
        self.W_out = 0.9 * self.W_out + 0.1 * last_W
        if record_for_replay:
            self.hippocampus.record(ids, targets=target_ids, salience=salience)
        return loss

    # ------------------------------------------------------------------ sleep
    def sleep(self, *, n_replays: int = 32, lr: float = 5e-3,
              prune: bool = True, seed: Optional[int] = None) -> dict:
        """One sleep cycle: prioritized hippocampal replay → cortical
        weight update → (optionally) one round of complement pruning."""
        rng = np.random.default_rng(seed)
        stats = self.hippocampus.consolidate(self, self.ftp_head,
                                              n_replays=n_replays,
                                              lr=lr, rng=rng)
        if prune:
            self.prune_step()
            stats["pruned"] = True
        # again EMA the readout from the FTP last-layer weights
        last_W = self.ftp_head.layers[-1].W
        self.W_out = 0.9 * self.W_out + 0.1 * last_W
        return stats

    # ------------------------------------------------------------------ prune
    def prune_step(self):
        """One C1q/C3 cycle across all dense weights in every column."""
        for col in self.columns:
            for name in ("W_q", "W_k", "W_v", "W_o"):
                W = getattr(col.l23_attn, name)
                self.pruner.observe(W)
                new_W, _ = self.pruner.step(W)
                setattr(col.l23_attn, name, new_W)
        self.pruner.observe(self.W_out)
        self.W_out, _ = self.pruner.step(self.W_out)

    # ------------------------------------------------------------------ info
    def info(self):
        col_params = 0
        for col in self.columns:
            for name in ("W_q", "W_k", "W_v", "W_o", "M_k", "M_v"):
                col_params += getattr(col.l23_attn, name).size
        attn0 = self.columns[0].l23_attn
        return {
            "vocab": self.vocab_size,
            "d_model": self.d_model,
            "columns": len(self.columns),
            "approx_params": int(self.E.size + self.W_out.size + col_params),
            "tokenizer": "UniversalTokenizer (byte-level + role/modality/lang)",
            "neuromodulators": ["DA", "ACh", "NE", "5-HT"],
            "thinking_loop": True,
            "kv_cache": True,
            "hippocampal_replay": True,
            # Phase-3 / Genius Edition
            "n_heads": attn0.n_heads,
            "d_h": attn0.d_h,
            "top_c": attn0.top_c,
            "acr_refresh": attn0.acr_refresh,
            "myelinated_concept_bank": True,
            "online_hebbian": self.online_hebbian,
        }
