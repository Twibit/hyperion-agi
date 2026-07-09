"""
Language Grounding — Broca's & Wernicke's Areas — V11 (Aurora-Praxis).
======================================================================

Biological language processing is split across two cortical areas connected
by the Arcuate Fasciculus:

  * **Wernicke's Area (Posterior Superior Temporal Gyrus):**
    Language *comprehension*. Maps incoming acoustic/text tokens into
    semantic SDR representations. In Aurora, this is the bridge between
    the raw text encoder and the deep cortical latent space — it learns
    to map surface forms to meaning vectors.

  * **Broca's Area (Inferior Frontal Gyrus):**
    Language *production*. Takes an internal semantic representation and
    sequences it into grammatical motor output (speech/text). In Aurora,
    this is the inverse of Wernicke's — it decodes latent vectors back
    into token sequences, guided by a language-specific context vector.

  * **Arcuate Fasciculus (the white-matter bridge):**
    A bidirectional fibre tract that allows Wernicke's comprehension signal
    to inform Broca's production, and Broca's motor plan to feed back
    into Wernicke's prediction (internal monologue / subvocalization).

Together, these three components form a closed language loop that supports:
  - Comprehension → Semantics (Wernicke)
  - Semantics → Production (Broca)
  - Production → Self-monitoring (Arcuate feedback)
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, List


class WernickesArea:
    """
    Language comprehension: maps surface token embeddings to deep semantic SDRs.
    Learns an associative mapping from token-space to concept-space.
    """

    def __init__(self, latent_dim: int, vocab_size: int = 8192,
                 learning_rate: float = 0.01, seed: int = 1729):
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.lr = learning_rate
        self.rng = np.random.default_rng(seed)

        # Token → semantic mapping (associative matrix)
        self.W_comprehend = (
            self.rng.standard_normal((latent_dim, vocab_size)).astype(np.float32)
            / np.sqrt(vocab_size)
        )
        self._n_comprehensions = 0

    def comprehend(self, token_id: int, context_z: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Map a token ID to a semantic vector.
        If context_z is provided, the output is biased toward the current context
        (contextual word meaning, not just dictionary meaning).
        """
        token_id = int(token_id) % self.vocab_size
        semantic = self.W_comprehend[:, token_id].copy()

        if context_z is not None:
            ctx = np.asarray(context_z, dtype=np.float32)
            # Blend: 70% learned meaning + 30% contextual bias
            semantic = 0.7 * semantic + 0.3 * ctx

        norm = np.linalg.norm(semantic) + 1e-9
        self._n_comprehensions += 1
        return (semantic / norm).astype(np.float32)

    def learn(self, token_id: int, target_z: np.ndarray) -> float:
        """
        Hebbian update: strengthen the association between token_id and the
        target semantic vector (e.g., from cortical processing).
        Returns the error before update.
        """
        token_id = int(token_id) % self.vocab_size
        current = self.W_comprehend[:, token_id]
        target = np.asarray(target_z, dtype=np.float32)
        error = float(np.linalg.norm(current - target))

        # Hebbian: move the column toward the target
        self.W_comprehend[:, token_id] += self.lr * (target - current)
        return error

    def stats(self) -> Dict[str, Any]:
        return {"n_comprehensions": self._n_comprehensions}


class BrocasArea:
    """
    Language production: decodes a semantic SDR back into a sequence of token IDs.
    Uses a learned inverse mapping (semantic → token probabilities).
    """

    def __init__(self, latent_dim: int, vocab_size: int = 8192,
                 learning_rate: float = 0.01, seed: int = 1729):
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.lr = learning_rate
        self.rng = np.random.default_rng(seed)

        # Semantic → token mapping (generative matrix, transpose of Wernicke's)
        self.W_produce = (
            self.rng.standard_normal((vocab_size, latent_dim)).astype(np.float32)
            / np.sqrt(latent_dim)
        )
        self._n_productions = 0

    def produce(self, semantic_z: np.ndarray, temperature: float = 1.0,
                language_bias: Optional[np.ndarray] = None) -> int:
        """
        Given a semantic vector, produce the most likely token ID.
        language_bias: optional per-token bias vector (e.g., from a language context
        vector that favors English grammar vs. Spanish grammar).
        """
        z = np.asarray(semantic_z, dtype=np.float32)
        logits = self.W_produce @ z

        if language_bias is not None:
            logits += np.asarray(language_bias, dtype=np.float32)

        # Stable softmax with temperature
        logits = np.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
        logits = logits - logits.max()
        logits = np.clip(logits / max(temperature, 0.01), -20, 0)
        probs = np.exp(logits)
        psum = probs.sum()
        if psum == 0 or np.isnan(psum):
            probs = np.ones_like(probs) / self.vocab_size
        else:
            probs = probs / psum

        token_id = int(self.rng.choice(self.vocab_size, p=probs))
        self._n_productions += 1
        return token_id

    def produce_sequence(self, semantic_z: np.ndarray, n_tokens: int = 8,
                         temperature: float = 1.0,
                         language_bias: Optional[np.ndarray] = None) -> List[int]:
        """Autoregressively produce a sequence of token IDs."""
        tokens = []
        z = semantic_z.copy()
        for _ in range(n_tokens):
            tok = self.produce(z, temperature=temperature, language_bias=language_bias)
            tokens.append(tok)
            # Shift the semantic vector slightly based on the produced token
            # (autoregressive drift — the production changes the internal state)
            z = 0.9 * z + 0.1 * self.W_produce[tok, :]
            norm = np.linalg.norm(z) + 1e-9
            z = z / norm
        return tokens

    def learn(self, token_id: int, source_z: np.ndarray) -> float:
        """
        Hebbian update: strengthen the association between the semantic vector
        and the correct token_id.
        """
        token_id = int(token_id) % self.vocab_size
        current = self.W_produce[token_id, :]
        source = np.asarray(source_z, dtype=np.float32)
        error = float(np.linalg.norm(current - source))
        self.W_produce[token_id, :] += self.lr * (source - current)
        return error

    def stats(self) -> Dict[str, Any]:
        return {"n_productions": self._n_productions}


class ArcuateFasciculus:
    """
    Bidirectional white-matter tract connecting Wernicke's and Broca's areas.

    Supports:
    1. Forward pass: Wernicke comprehension → Broca production (normal speech)
    2. Backward pass: Broca motor plan → Wernicke prediction (internal monologue)
    """

    def __init__(self, wernicke: WernickesArea, broca: BrocasArea,
                 feedback_strength: float = 0.3):
        self.wernicke = wernicke
        self.broca = broca
        self.feedback_strength = float(feedback_strength)
        self._n_forward = 0
        self._n_backward = 0

    def speak(self, token_id: int, context_z: Optional[np.ndarray] = None,
              temperature: float = 1.0, n_tokens: int = 8) -> List[int]:
        """
        Full language loop: comprehend input → produce response.
        Wernicke processes the input token, then Broca generates a response sequence.
        """
        # Wernicke: comprehend
        semantic = self.wernicke.comprehend(token_id, context_z=context_z)

        # Broca: produce
        output_tokens = self.broca.produce_sequence(semantic, n_tokens=n_tokens,
                                                     temperature=temperature)
        self._n_forward += 1
        return output_tokens

    def inner_speech(self, semantic_z: np.ndarray, n_tokens: int = 4,
                     temperature: float = 0.5) -> np.ndarray:
        """
        Internal monologue: Broca produces tokens, then feeds them back
        through Wernicke to generate a refined semantic representation.
        This is the neural substrate of 'thinking before speaking'.
        """
        # Broca produces candidate tokens
        candidate_tokens = self.broca.produce_sequence(
            semantic_z, n_tokens=n_tokens, temperature=temperature
        )

        # Feed each token back through Wernicke and average
        refined_vectors = []
        ctx = semantic_z.copy()
        for tok in candidate_tokens:
            v = self.wernicke.comprehend(tok, context_z=ctx)
            refined_vectors.append(v)
            # Update context with each new comprehension
            ctx = 0.7 * ctx + 0.3 * v

        refined = np.mean(refined_vectors, axis=0).astype(np.float32)
        norm = np.linalg.norm(refined) + 1e-9
        self._n_backward += 1
        return (refined / norm).astype(np.float32)

    def stats(self) -> Dict[str, Any]:
        return {
            "n_forward_passes": self._n_forward,
            "n_backward_passes": self._n_backward,
            "wernicke": self.wernicke.stats(),
            "broca": self.broca.stats(),
        }
