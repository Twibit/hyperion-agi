"""
information_cortex.py — Information-Theoretic Reasoning Module for Hyperion

Provides Shannon entropy measurement, Bayesian inference, compression-based
complexity analysis, and knowledge quality evaluation. Together these give
Hyperion the ability to measure and reason about information quality.
"""

import math
import re
import zlib
from collections import Counter, defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# 1. EntropyMeter
# ---------------------------------------------------------------------------

class EntropyMeter:
    """Measures information content of text and probability distributions
    using Shannon Information Theory."""

    def shannon_entropy(self, distribution: dict[str, float]) -> float:
        """Compute Shannon entropy H = -Σ p(x) * log2(p(x)).

        Args:
            distribution: Mapping of outcomes to their probabilities.

        Returns:
            Entropy in bits.  Returns 0.0 for empty or degenerate
            distributions.  Zero-probability entries are skipped.
        """
        if not distribution:
            return 0.0

        entropy = 0.0
        for p in distribution.values():
            if p > 0.0:
                entropy -= p * math.log2(p)
        return entropy

    def text_entropy(self, text: str) -> float:
        """Compute character-level Shannon entropy of *text*.

        Args:
            text: Arbitrary Unicode string.

        Returns:
            Entropy in bits per character.  Returns 0.0 for empty strings.
        """
        if not text:
            return 0.0

        counts = Counter(text)
        total = len(text)
        distribution = {ch: count / total for ch, count in counts.items()}
        return self.shannon_entropy(distribution)

    def token_entropy(self, tokens: list[str]) -> float:
        """Compute token-level Shannon entropy.

        Args:
            tokens: List of string tokens (words, sub-words, etc.).

        Returns:
            Entropy in bits per token.  Returns 0.0 for empty lists.
        """
        if not tokens:
            return 0.0

        counts = Counter(tokens)
        total = len(tokens)
        distribution = {tok: count / total for tok, count in counts.items()}
        return self.shannon_entropy(distribution)

    def conditional_entropy(
        self,
        joint_counts: dict[tuple, float],
        marginal_counts: dict[str, float],
    ) -> float:
        """Compute conditional entropy H(Y|X) = H(X,Y) - H(X).

        Args:
            joint_counts: Mapping of (x, y) pairs to their counts/frequencies.
            marginal_counts: Mapping of x values to their counts/frequencies.

        Returns:
            Conditional entropy in bits.  Returns 0.0 when inputs are empty.
        """
        if not joint_counts or not marginal_counts:
            return 0.0

        # Normalise counts into probability distributions
        joint_total = sum(joint_counts.values())
        marginal_total = sum(marginal_counts.values())

        if joint_total == 0 or marginal_total == 0:
            return 0.0

        joint_dist = {k: v / joint_total for k, v in joint_counts.items()}
        marginal_dist = {k: v / marginal_total for k, v in marginal_counts.items()}

        h_joint = self.shannon_entropy(
            {str(k): v for k, v in joint_dist.items()}
        )
        h_marginal = self.shannon_entropy(marginal_dist)

        conditional = h_joint - h_marginal
        return max(0.0, conditional)  # guard against floating-point drift

    def mutual_information(
        self,
        joint_counts: dict[tuple[str, str], float],
    ) -> float:
        """Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).

        Marginal distributions are derived from the joint counts.

        Args:
            joint_counts: Mapping of (x, y) pairs to their counts.

        Returns:
            Mutual information in bits.  Returns 0.0 for empty input.
        """
        if not joint_counts:
            return 0.0

        total = sum(joint_counts.values())
        if total == 0:
            return 0.0

        # Build marginals
        x_counts: dict[str, float] = defaultdict(float)
        y_counts: dict[str, float] = defaultdict(float)
        for (x, y), count in joint_counts.items():
            x_counts[x] += count
            y_counts[y] += count

        joint_dist = {str(k): v / total for k, v in joint_counts.items()}
        x_dist = {k: v / total for k, v in x_counts.items()}
        y_dist = {k: v / total for k, v in y_counts.items()}

        h_x = self.shannon_entropy(x_dist)
        h_y = self.shannon_entropy(y_dist)
        h_joint = self.shannon_entropy(joint_dist)

        mi = h_x + h_y - h_joint
        return max(0.0, mi)  # guard against floating-point drift

    def information_gain(
        self,
        before: dict[str, float],
        after: dict[str, float],
    ) -> float:
        """Compute how much entropy was reduced (information gain).

        Args:
            before: Distribution before observation.
            after:  Distribution after observation.

        Returns:
            H(before) - H(after).  Can be negative if entropy increased.
        """
        return self.shannon_entropy(before) - self.shannon_entropy(after)


# ---------------------------------------------------------------------------
# 2. BayesianReasoner
# ---------------------------------------------------------------------------

class BayesianReasoner:
    """Maintains beliefs about hypotheses and updates them with evidence
    via Bayes' theorem."""

    def __init__(self) -> None:
        """Initialise the reasoner with an empty hypothesis store.

        ``self.hypotheses`` maps *domain* → *hypothesis* → probability
        (the current posterior, initialised as the prior).
        """
        self.hypotheses: dict[str, dict[str, float]] = defaultdict(dict)

    def add_hypothesis(
        self,
        domain: str,
        hypothesis: str,
        prior: float = 0.5,
    ) -> None:
        """Register a hypothesis with a prior probability.

        Args:
            domain: Knowledge domain (e.g. ``"weather"``).
            hypothesis: Human-readable hypothesis label.
            prior: Initial belief probability in [0, 1].
        """
        self.hypotheses[domain][hypothesis] = max(0.0, min(1.0, prior))

    def update(
        self,
        domain: str,
        hypothesis: str,
        evidence: str,
        likelihood: float,
        marginal_likelihood: float | None = None,
    ) -> None:
        """Perform a Bayesian update on *hypothesis* given *evidence*.

        P(H|E) = P(E|H) * P(H) / P(E)

        If *marginal_likelihood* is ``None`` it is estimated as:

            P(E) = P(E|H)*P(H) + P(E|¬H)*P(¬H)

        where P(E|¬H) = 1 - likelihood.

        Args:
            domain: Knowledge domain.
            hypothesis: Hypothesis to update.
            evidence: Description of the observed evidence (stored for
                traceability but not used in computation).
            likelihood: P(E|H) — probability of evidence given the
                hypothesis is true.
            marginal_likelihood: Optional P(E).  Computed automatically
                when ``None``.
        """
        if domain not in self.hypotheses or hypothesis not in self.hypotheses[domain]:
            return  # unknown hypothesis — nothing to update

        prior = self.hypotheses[domain][hypothesis]

        if marginal_likelihood is None:
            p_e_not_h = 1.0 - likelihood
            marginal_likelihood = likelihood * prior + p_e_not_h * (1.0 - prior)

        if marginal_likelihood == 0.0:
            return  # avoid division by zero

        posterior = (likelihood * prior) / marginal_likelihood
        self.hypotheses[domain][hypothesis] = max(0.0, min(1.0, posterior))

    def get_belief(self, domain: str, hypothesis: str) -> float:
        """Return the current posterior probability.

        Args:
            domain: Knowledge domain.
            hypothesis: Hypothesis label.

        Returns:
            Current posterior in [0, 1], or 0.0 if unknown.
        """
        return self.hypotheses.get(domain, {}).get(hypothesis, 0.0)

    def get_most_likely(self, domain: str) -> tuple[str, float]:
        """Return the hypothesis with the highest posterior in *domain*.

        Args:
            domain: Knowledge domain.

        Returns:
            ``(hypothesis, probability)`` tuple.  Returns ``("", 0.0)``
            when the domain is empty or unknown.
        """
        domain_hyps = self.hypotheses.get(domain, {})
        if not domain_hyps:
            return ("", 0.0)
        best = max(domain_hyps.items(), key=lambda item: item[1])
        return best

    def reset_domain(self, domain: str) -> None:
        """Reset all hypotheses in *domain* to uniform priors.

        Args:
            domain: Knowledge domain to reset.
        """
        domain_hyps = self.hypotheses.get(domain)
        if not domain_hyps:
            return
        n = len(domain_hyps)
        uniform = 1.0 / n if n > 0 else 0.0
        for hyp in domain_hyps:
            domain_hyps[hyp] = uniform


# ---------------------------------------------------------------------------
# 3. CompressionAnalyzer
# ---------------------------------------------------------------------------

class CompressionAnalyzer:
    """Uses compression ratio as a proxy for information content
    (Kolmogorov complexity approximation)."""

    def compression_ratio(self, text: str) -> float:
        """Measure how compressible *text* is via zlib.

        Args:
            text: Input string.

        Returns:
            ``len(compressed) / len(original_bytes)``.
            Lower values indicate more redundancy.
            Returns 1.0 for empty text.
        """
        if not text:
            return 1.0
        original = text.encode("utf-8")
        compressed = zlib.compress(original)
        return len(compressed) / len(original)

    def novelty_score(self, new_text: str, corpus: list[str]) -> float:
        """Compute average Normalised Compression Distance against a corpus.

        NCD(x, y) = (C(xy) − min(C(x), C(y))) / max(C(x), C(y))

        Higher values indicate more novel content.

        Args:
            new_text: Text to evaluate.
            corpus: List of existing texts to compare against.

        Returns:
            Average NCD in [0, 1].  Returns 1.0 when the corpus is empty
            (maximally novel by convention).
        """
        if not corpus:
            return 1.0
        if not new_text:
            return 0.0

        x_bytes = new_text.encode("utf-8")
        c_x = len(zlib.compress(x_bytes))

        ncd_values: list[float] = []
        for entry in corpus:
            if not entry:
                continue
            y_bytes = entry.encode("utf-8")
            c_y = len(zlib.compress(y_bytes))
            c_xy = len(zlib.compress(x_bytes + y_bytes))

            denominator = max(c_x, c_y)
            if denominator == 0:
                continue
            ncd = (c_xy - min(c_x, c_y)) / denominator
            ncd_values.append(ncd)

        if not ncd_values:
            return 1.0

        return sum(ncd_values) / len(ncd_values)

    def redundancy(self, text: str) -> float:
        """Compute the redundancy of *text*.

        Args:
            text: Input string.

        Returns:
            ``1.0 - compression_ratio``.  Higher values mean more
            redundant content.
        """
        return 1.0 - self.compression_ratio(text)


# ---------------------------------------------------------------------------
# 4. KnowledgeQualityMeter
# ---------------------------------------------------------------------------

class KnowledgeQualityMeter:
    """Evaluates the quality and coverage of learned knowledge."""

    def __init__(self) -> None:
        """Initialise with an empty learning history."""
        self.history: list[dict[str, Any]] = []

    def record_learning(
        self,
        source: str,
        domain: str,
        token_count: int,
        entropy: float,
        compression: float,
    ) -> None:
        """Log a learning event.

        Args:
            source: Where the knowledge came from (e.g. URL, filename).
            domain: Knowledge domain label.
            token_count: Number of tokens consumed.
            entropy: Measured entropy of the source material.
            compression: Compression ratio of the source material.
        """
        self.history.append(
            {
                "source": source,
                "domain": domain,
                "token_count": token_count,
                "entropy": entropy,
                "compression": compression,
            }
        )

    def coverage_score(
        self,
        vocab_size: int,
        target_vocab: int = 50_000,
    ) -> float:
        """Estimate vocabulary coverage as a fraction of a target.

        Args:
            vocab_size: Current vocabulary size.
            target_vocab: Target vocabulary size (default 50 000).

        Returns:
            ``min(1.0, vocab_size / target_vocab)``.
        """
        if target_vocab <= 0:
            return 1.0
        return min(1.0, vocab_size / target_vocab)

    def diversity_score(self) -> float:
        """Compute domain diversity as normalised Shannon entropy.

        Returns:
            Value in [0, 1].  1.0 means perfectly uniform domain
            distribution.  Returns 0.0 when the history is empty.
        """
        if not self.history:
            return 0.0

        domain_counts = Counter(event["domain"] for event in self.history)
        total = sum(domain_counts.values())
        if total == 0:
            return 0.0

        distribution = {d: c / total for d, c in domain_counts.items()}
        entropy = 0.0
        for p in distribution.values():
            if p > 0:
                entropy -= p * math.log2(p)

        n_domains = len(domain_counts)
        max_entropy = math.log2(n_domains) if n_domains > 1 else 1.0
        if max_entropy == 0:
            return 0.0

        return min(1.0, entropy / max_entropy)

    def diminishing_returns_detector(self) -> dict:
        """Analyse recent learning events for diminishing returns.

        Inspects the last 20 learning events.  If the average per-event
        entropy change (slope) is below 0.01 the domain is considered
        saturated and a domain switch is recommended.

        Returns:
            ``{'saturated': bool, 'recommendation': str (if saturated),
              'slope': float}``
        """
        window = self.history[-20:] if len(self.history) >= 2 else self.history

        if len(window) < 2:
            return {"saturated": False, "slope": 0.0}

        entropies = [e["entropy"] for e in window]
        # Simple linear slope: average change between consecutive events
        deltas = [
            abs(entropies[i + 1] - entropies[i])
            for i in range(len(entropies) - 1)
        ]
        slope = sum(deltas) / len(deltas) if deltas else 0.0

        if slope < 0.01:
            return {
                "saturated": True,
                "recommendation": "switch domain",
                "slope": slope,
            }
        return {"saturated": False, "slope": slope}

    def report(self) -> dict:
        """Generate a comprehensive knowledge quality report.

        Returns:
            Dict with keys: ``total_events``, ``coverage`` (None — caller
            should supply vocab size), ``diversity``, ``per_domain_stats``,
            ``diminishing_returns``.
        """
        per_domain: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"events": 0, "total_tokens": 0, "avg_entropy": 0.0, "avg_compression": 0.0}
        )

        for event in self.history:
            d = event["domain"]
            stats = per_domain[d]
            stats["events"] += 1
            stats["total_tokens"] += event["token_count"]
            stats["avg_entropy"] += event["entropy"]
            stats["avg_compression"] += event["compression"]

        # Normalise averages
        for d, stats in per_domain.items():
            n = stats["events"]
            if n > 0:
                stats["avg_entropy"] /= n
                stats["avg_compression"] /= n

        return {
            "total_events": len(self.history),
            "diversity": self.diversity_score(),
            "per_domain_stats": dict(per_domain),
            "diminishing_returns": self.diminishing_returns_detector(),
        }


# ---------------------------------------------------------------------------
# 5. InformationCortex
# ---------------------------------------------------------------------------

class InformationCortex:
    """Integrated information-theoretic reasoning module for Hyperion.

    Composes :class:`EntropyMeter`, :class:`BayesianReasoner`,
    :class:`CompressionAnalyzer`, and :class:`KnowledgeQualityMeter` into
    a single coherent interface.
    """

    def __init__(self) -> None:
        """Create all sub-modules."""
        self.entropy_meter = EntropyMeter()
        self.bayesian_reasoner = BayesianReasoner()
        self.compression_analyzer = CompressionAnalyzer()
        self.quality_meter = KnowledgeQualityMeter()

    def analyze_text(self, text: str) -> dict:
        """Run a full information-theoretic analysis on *text*.

        Args:
            text: Input string.

        Returns:
            Dict with keys:
            - ``char_entropy``: Character-level Shannon entropy.
            - ``compression_ratio``: zlib compression ratio.
            - ``redundancy``: 1 − compression_ratio.
            - ``novelty_estimate``: Compression ratio used as a rough proxy
              (higher ratio ≈ less compressible ≈ more novel).
            - ``information_density``: entropy / max_possible_entropy
              (max entropy for *n* unique chars = log2(n)).
        """
        char_entropy = self.entropy_meter.text_entropy(text)
        comp_ratio = self.compression_analyzer.compression_ratio(text)
        redundancy = self.compression_analyzer.redundancy(text)

        unique_chars = len(set(text)) if text else 1
        max_entropy = math.log2(unique_chars) if unique_chars > 1 else 1.0
        info_density = char_entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            "char_entropy": char_entropy,
            "compression_ratio": comp_ratio,
            "redundancy": redundancy,
            "novelty_estimate": comp_ratio,
            "information_density": info_density,
        }

    def evaluate_learning_value(
        self,
        text: str,
        existing_vocab: set,
    ) -> dict:
        """Estimate the learning value of *text* relative to existing knowledge.

        Tokenisation uses a simple ``re`` word-boundary split.

        Args:
            text: Candidate learning material.
            existing_vocab: Set of already-known tokens.

        Returns:
            Dict with keys:
            - ``new_tokens``: Count of tokens not in *existing_vocab*.
            - ``novelty_ratio``: new / total tokens (0.0 when empty).
            - ``estimated_value``: novelty_ratio × compression_ratio × 100.
        """
        tokens = re.findall(r"\b\w+\b", text.lower()) if text else []
        total = len(tokens)

        new_tokens = [t for t in tokens if t not in existing_vocab]
        new_count = len(new_tokens)
        novelty_ratio = new_count / total if total > 0 else 0.0
        comp_ratio = self.compression_analyzer.compression_ratio(text)
        estimated_value = novelty_ratio * comp_ratio * 100

        return {
            "new_tokens": new_count,
            "novelty_ratio": novelty_ratio,
            "estimated_value": estimated_value,
        }

    def bayesian_update(
        self,
        domain: str,
        hypothesis: str,
        evidence: str,
        likelihood: float,
    ) -> None:
        """Delegate a Bayesian update to :class:`BayesianReasoner`.

        Args:
            domain: Knowledge domain.
            hypothesis: Hypothesis to update.
            evidence: Evidence description.
            likelihood: P(E|H).
        """
        self.bayesian_reasoner.update(domain, hypothesis, evidence, likelihood)

    def status(self) -> dict:
        """Aggregate status from all sub-modules.

        Returns:
            Dict with keys ``entropy_meter``, ``bayesian_reasoner``,
            ``compression_analyzer``, ``knowledge_quality``.
        """
        return {
            "entropy_meter": {"type": "EntropyMeter", "status": "active"},
            "bayesian_reasoner": {
                "type": "BayesianReasoner",
                "domains": len(self.bayesian_reasoner.hypotheses),
                "total_hypotheses": sum(
                    len(h)
                    for h in self.bayesian_reasoner.hypotheses.values()
                ),
                "status": "active",
            },
            "compression_analyzer": {
                "type": "CompressionAnalyzer",
                "status": "active",
            },
            "knowledge_quality": self.quality_meter.report(),
        }
