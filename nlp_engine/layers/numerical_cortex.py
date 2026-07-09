"""
numerical_cortex.py — Numerical reasoning module for Hyperion.

Provides the ability to understand numbers, magnitudes, mathematical
relationships, and numerical patterns. Includes multi-scale number
encoding, statistical memory for numerical facts, sequence pattern
detection, and an integrated cortex that ties everything together.
"""

import numpy as np
import math
import re
from collections import defaultdict

def list_defaultdict():
    return defaultdict(list)


class NumberEncoder:
    """Converts raw numbers into multi-scale representations.

    Encodes numerical values into rich descriptors capturing magnitude,
    sign, normalization, integrality, and categorical bin. Also provides
    a logarithmic-ratio similarity metric between two numbers.
    """

    def encode(self, value: float) -> dict:
        """Encode a number into a multi-scale representation.

        Args:
            value: The numerical value to encode.

        Returns:
            A dictionary with keys:
                - 'magnitude': floor(log10(abs(value))), or 0 if value is 0.
                - 'sign': 1, -1, or 0.
                - 'normalized': value / 10^magnitude (in [1, 10) for nonzero).
                - 'is_integer': Whether the value is an integer.
                - 'bin': Categorical bin string.
        """
        # Sign
        if value > 0:
            sign = 1
        elif value < 0:
            sign = -1
        else:
            sign = 0

        # Magnitude
        if value != 0:
            magnitude = int(math.floor(math.log10(abs(value))))
        else:
            magnitude = 0

        # Normalized
        if value != 0:
            normalized = value / (10 ** magnitude)
        else:
            normalized = 0.0

        # Is integer
        is_integer = float(value).is_integer()

        # Categorical bin
        abs_val = abs(value)
        if abs_val == 0:
            bin_label = 'zero'
        elif abs_val < 1:
            bin_label = 'tiny'
        elif abs_val < 100:
            bin_label = 'small'
        elif abs_val < 10000:
            bin_label = 'medium'
        elif abs_val < 1_000_000:
            bin_label = 'large'
        else:
            bin_label = 'huge'

        return {
            'magnitude': magnitude,
            'sign': sign,
            'normalized': normalized,
            'is_integer': is_integer,
            'bin': bin_label,
        }

    def similarity(self, a: float, b: float) -> float:
        """Compute similarity between two numbers based on ratio proximity.

        Uses the formula: 1.0 / (1.0 + abs(log(a/b))) for same-sign
        nonzero values.

        Args:
            a: First numerical value.
            b: Second numerical value.

        Returns:
            A float in [0.0, 1.0] representing similarity.
            Returns 0.0 if the values have different signs or either is zero
            (unless both are zero, in which case returns 1.0).
        """
        if a == 0 and b == 0:
            return 1.0
        if a == 0 or b == 0:
            return 0.0
        if (a > 0) != (b > 0):
            return 0.0

        ratio = abs(a) / abs(b)
        return 1.0 / (1.0 + abs(math.log(ratio)))


class NumericalMemory:
    """Stores numerical facts as (subject, relation, value) triples."""

    def __init__(self):
        """Initialize the NumericalMemory store."""
        # maps subject -> relation -> list of observed values
        self.facts: dict[str, dict[str, list[float]]] = defaultdict(list_defaultdict)

    def store(self, subject: str, relation: str, value: float) -> None:
        """Store a numerical observation.

        Appends the value to the list for the given (subject, relation)
        pair, keeping only the last 100 observations.

        Args:
            subject: The entity the fact is about.
            relation: The type of numerical relationship.
            value: The observed numerical value.
        """
        self.facts[subject][relation].append(value)
        # Keep only last 100 observations
        if len(self.facts[subject][relation]) > 100:
            self.facts[subject][relation] = self.facts[subject][relation][-100:]

    def query(self, subject: str, relation: str) -> dict | None:
        """Query statistical summary for a (subject, relation) pair.

        Args:
            subject: The entity to query.
            relation: The relationship to query.

        Returns:
            A dictionary with keys 'mean', 'median', 'std', 'min', 'max',
            'count', and 'trend' (slope of last 10 values via simple
            linear regression). Returns None if no data exists.
        """
        if subject not in self.facts or relation not in self.facts[subject]:
            return None

        values = self.facts[subject][relation]
        if not values:
            return None

        arr = np.array(values, dtype=float)
        count = len(arr)

        # Trend via simple linear regression on last 10 values
        trend_values = arr[-10:]
        if len(trend_values) >= 2:
            x = np.arange(len(trend_values), dtype=float)
            x_mean = np.mean(x)
            y_mean = np.mean(trend_values)
            numerator = np.sum((x - x_mean) * (trend_values - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            if denominator != 0:
                trend = float(numerator / denominator)
            else:
                trend = 0.0
        else:
            trend = 0.0

        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'count': count,
            'trend': trend,
        }

    def compare(self, subj_a: str, subj_b: str, relation: str) -> dict | None:
        """Compare two subjects on a given relation.

        Args:
            subj_a: First subject.
            subj_b: Second subject.
            relation: The relation to compare on.

        Returns:
            A dictionary with keys 'ratio', 'difference', and 'which_larger'.
            Returns None if either subject lacks data for the relation.
        """
        stats_a = self.query(subj_a, relation)
        stats_b = self.query(subj_b, relation)

        if stats_a is None or stats_b is None:
            return None

        mean_a = stats_a['mean']
        mean_b = stats_b['mean']

        difference = mean_a - mean_b

        if mean_b != 0:
            ratio = mean_a / mean_b
        else:
            ratio = float('inf') if mean_a != 0 else 1.0

        if mean_a > mean_b:
            which_larger = subj_a
        elif mean_b > mean_a:
            which_larger = subj_b
        else:
            which_larger = 'equal'

        return {
            'ratio': ratio,
            'difference': difference,
            'which_larger': which_larger,
        }


class SequenceDetector:
    """Detects mathematical patterns in numerical sequences.

    Identifies arithmetic, geometric, Fibonacci-like, constant,
    or random patterns and predicts the next value.
    """

    def detect_pattern(self, sequence: list[float]) -> dict:
        """Analyze a numerical sequence and detect its pattern.

        Checks for constant, arithmetic, geometric, and Fibonacci-like
        patterns in order of specificity.

        Args:
            sequence: A list of numerical values to analyze.

        Returns:
            A dictionary with keys:
                - 'type': Pattern type string.
                - 'confidence': Float in [0.0, 1.0].
                - 'next_predicted': Predicted next value.
                - 'common_diff': (if arithmetic) The common difference.
                - 'common_ratio': (if geometric) The common ratio.
        """
        if len(sequence) < 2:
            return {
                'type': 'random',
                'confidence': 0.0,
                'next_predicted': sequence[-1] if sequence else 0.0,
            }

        arr = np.array(sequence, dtype=float)

        # Check constant
        result = self._check_constant(arr)
        if result is not None:
            return result

        # Check arithmetic
        result = self._check_arithmetic(arr)
        if result is not None:
            return result

        # Check geometric
        result = self._check_geometric(arr)
        if result is not None:
            return result

        # Check fibonacci-like (needs at least 3 elements)
        if len(arr) >= 3:
            result = self._check_fibonacci(arr)
            if result is not None:
                return result

        # Default: random
        return {
            'type': 'random',
            'confidence': 0.0,
            'next_predicted': float(arr[-1]),
        }

    def _check_constant(self, arr: np.ndarray) -> dict | None:
        """Check if the sequence is approximately constant.

        A sequence is constant if std/|mean| < 0.01 (and mean != 0),
        or if std itself is near-zero.

        Args:
            arr: Numpy array of the sequence values.

        Returns:
            Pattern dict if constant, else None.
        """
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))

        if mean_val != 0 and std_val / abs(mean_val) < 0.01:
            return {
                'type': 'constant',
                'confidence': 1.0 - (std_val / abs(mean_val)) if mean_val != 0 else 1.0,
                'next_predicted': mean_val,
            }
        elif mean_val == 0 and std_val < 1e-10:
            return {
                'type': 'constant',
                'confidence': 1.0,
                'next_predicted': 0.0,
            }
        return None

    def _check_arithmetic(self, arr: np.ndarray) -> dict | None:
        """Check if the sequence follows an arithmetic progression.

        Arithmetic if std(diffs)/|mean(diffs)| < 0.1.

        Args:
            arr: Numpy array of the sequence values.

        Returns:
            Pattern dict if arithmetic, else None.
        """
        diffs = np.diff(arr)
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))

        if mean_diff == 0:
            return None  # Handled by constant check

        if abs(mean_diff) > 0 and std_diff / abs(mean_diff) < 0.1:
            confidence = 1.0 - (std_diff / abs(mean_diff))
            next_predicted = float(arr[-1]) + mean_diff
            return {
                'type': 'arithmetic',
                'confidence': max(0.0, min(1.0, confidence)),
                'next_predicted': next_predicted,
                'common_diff': mean_diff,
            }
        return None

    def _check_geometric(self, arr: np.ndarray) -> dict | None:
        """Check if the sequence follows a geometric progression.

        Geometric if std(ratios)/|mean(ratios)| < 0.1. Requires all
        consecutive elements to be nonzero.

        Args:
            arr: Numpy array of the sequence values.

        Returns:
            Pattern dict if geometric, else None.
        """
        # All elements except last must be nonzero for ratio computation
        if np.any(arr[:-1] == 0):
            return None

        ratios = arr[1:] / arr[:-1]
        mean_ratio = float(np.mean(ratios))
        std_ratio = float(np.std(ratios))

        if mean_ratio == 0:
            return None

        if abs(mean_ratio) > 0 and std_ratio / abs(mean_ratio) < 0.1:
            confidence = 1.0 - (std_ratio / abs(mean_ratio))
            next_predicted = float(arr[-1]) * mean_ratio
            return {
                'type': 'geometric',
                'confidence': max(0.0, min(1.0, confidence)),
                'next_predicted': next_predicted,
                'common_ratio': mean_ratio,
            }
        return None

    def _check_fibonacci(self, arr: np.ndarray) -> dict | None:
        """Check if the sequence is Fibonacci-like.

        Each element (from index 2 onward) should approximately equal
        the sum of the two preceding elements.

        Args:
            arr: Numpy array of the sequence values (length >= 3).

        Returns:
            Pattern dict if Fibonacci-like, else None.
        """
        expected_sums = arr[:-2] + arr[1:-1]
        actual = arr[2:]

        # Avoid division by zero in relative error
        nonzero_mask = actual != 0
        if not np.any(nonzero_mask):
            return None

        relative_errors = np.zeros_like(actual)
        relative_errors[nonzero_mask] = np.abs(
            (actual[nonzero_mask] - expected_sums[nonzero_mask])
            / actual[nonzero_mask]
        )
        # For zero actuals where expected is also zero, error is 0
        # For zero actuals where expected is nonzero, treat as high error
        zero_actual_nonzero_expected = (~nonzero_mask) & (expected_sums != 0)
        relative_errors[zero_actual_nonzero_expected] = 1.0

        mean_error = float(np.mean(relative_errors))

        if mean_error < 0.1:
            confidence = max(0.0, min(1.0, 1.0 - mean_error))
            next_predicted = float(arr[-2] + arr[-1])
            return {
                'type': 'fibonacci-like',
                'confidence': confidence,
                'next_predicted': next_predicted,
            }
        return None


class NumericalCortex:
    """Integrated numerical reasoning module for Hyperion.

    Combines number encoding, numerical memory, and sequence detection
    into a unified interface for extracting, storing, comparing, and
    reasoning about numerical information in text.
    """

    # Regex for matching numbers: integers, floats, percentages, negatives,
    # and numbers with commas (e.g. 1,000,000).
    _NUMBER_PATTERN = re.compile(
        r'-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?'
    )

    def __init__(self) -> None:
        """Initialize sub-components: encoder, memory, and detector."""
        self.encoder: NumberEncoder = NumberEncoder()
        self.memory: NumericalMemory = NumericalMemory()
        self.detector: SequenceDetector = SequenceDetector()

    def extract_numbers(self, text: str) -> list[tuple[str, float]]:
        """Extract all numbers from a text string.

        Handles integers, floats, percentages, negatives, and numbers
        with commas (e.g. '1,000,000').

        Args:
            text: The input text to scan for numbers.

        Returns:
            A list of (original_string, parsed_float) tuples.
        """
        results: list[tuple[str, float]] = []
        for match in self._NUMBER_PATTERN.finditer(text):
            original = match.group()
            # Remove commas and percentage for parsing
            cleaned = original.replace(',', '')
            is_percent = cleaned.endswith('%')
            if is_percent:
                cleaned = cleaned[:-1]
            try:
                value = float(cleaned)
                if is_percent:
                    value = value / 100.0
                results.append((original, value))
            except ValueError:
                continue
        return results

    def learn_numerical(self, text: str, source: str = 'unknown') -> None:
        """Extract numbers from text and associate with nearby concepts.

        For each extracted number, looks backward through the preceding
        tokens (up to 5) to find a plausible noun/concept (alphabetic
        word of length >= 2). Stores as (concept, 'value', number) in
        NumericalMemory.

        Args:
            text: The input text to learn from.
            source: Source identifier for the text.
        """
        numbers = self.extract_numbers(text)
        if not numbers:
            return

        # Tokenize text for concept association
        tokens = text.split()

        for original_str, value in numbers:
            # Find the token index where this number appears
            num_index = None
            for i, token in enumerate(tokens):
                if original_str in token:
                    num_index = i
                    break

            if num_index is None:
                continue

            # Look backward up to 5 tokens for a concept (alphabetic word >= 2 chars)
            concept = source
            for j in range(num_index - 1, max(num_index - 6, -1), -1):
                candidate = re.sub(r'[^a-zA-Z]', '', tokens[j])
                if len(candidate) >= 2 and candidate.isalpha():
                    concept = candidate.lower()
                    break

            self.memory.store(concept, 'value', value)

    def predict_sequence(self, values: list[float]) -> dict:
        """Predict the next value in a numerical sequence.

        Delegates to the internal SequenceDetector.

        Args:
            values: A list of numerical values forming a sequence.

        Returns:
            Pattern detection result dict from SequenceDetector.
        """
        return self.detector.detect_pattern(values)

    def numerical_similarity(self, text_a: str, text_b: str) -> float:
        """Compute numerical similarity between two texts.

        Extracts numbers from both texts and computes the average
        pairwise similarity using NumberEncoder.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            A float in [0.0, 1.0] representing average pairwise
            numerical similarity. Returns 0.0 if either text contains
            no numbers.
        """
        nums_a = self.extract_numbers(text_a)
        nums_b = self.extract_numbers(text_b)

        if not nums_a or not nums_b:
            return 0.0

        similarities: list[float] = []
        for _, val_a in nums_a:
            for _, val_b in nums_b:
                similarities.append(self.encoder.similarity(val_a, val_b))

        if not similarities:
            return 0.0

        return float(np.mean(similarities))

    def status(self) -> dict:
        """Return statistics about the numerical cortex state.

        Returns:
            A dictionary with keys:
                - 'total_facts': Total number of stored observations.
                - 'unique_subjects': Number of distinct subjects.
                - 'unique_relations': Number of distinct relations.
        """
        total_facts = 0
        unique_relations: set[str] = set()

        for subject in self.facts:
            for relation in self.facts[subject]:
                unique_relations.add(relation)
                total_facts += len(self.facts[subject][relation])

        return {
            'total_facts': total_facts,
            'unique_subjects': len(self.facts),
            'unique_relations': len(unique_relations),
        }

    @property
    def facts(self) -> dict[str, dict[str, list[float]]]:
        """Proxy access to the NumericalMemory's facts store.

        Returns:
            The internal facts dictionary from NumericalMemory.
        """
        return self.memory.facts
