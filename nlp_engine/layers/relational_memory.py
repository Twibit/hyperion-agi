"""
relational_memory.py — Graph-based relational knowledge store for Hyperion.

Provides abstract relational reasoning by storing and querying knowledge as a
graph of (subject, predicate, object) triples, analogous to a biological
knowledge graph.  Supports heuristic triple extraction from natural-language
text, BFS-based path finding between entities, analogical reasoning, and
confidence-based pruning.
"""

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class KnowledgeTriple:
    """A single relational fact stored as a (subject, predicate, object) triple.

    Attributes:
        subject: The subject entity of the triple.
        predicate: The relationship connecting subject to object.
        object: The object entity of the triple.
        confidence: Confidence score in the range [0.0, 1.0].
        source: Provenance label describing where this fact was learned.
        timestamp: Unix timestamp recording when the triple was stored.
    """

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)


class RelationalMemory:
    """Graph-based knowledge store backed by indexed (S, P, O) triples.

    The store maintains three inverted indices (by subject, predicate, and
    object) for fast lookups, supports capacity limits with automatic eviction
    of the lowest-confidence triple, and offers several higher-level reasoning
    primitives built on top of the raw graph.

    Args:
        max_triples: Maximum number of triples the store will hold.  When
            capacity is reached the lowest-confidence triple is evicted before
            a new one is inserted.
    """

    # ------------------------------------------------------------------
    # Regex patterns used by ``extract_triples`` (compiled once at class
    # level for efficiency).
    # ------------------------------------------------------------------
    _EXTRACTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
        # Pattern 6 — passive "created_by" (must precede generic "is" patterns)
        (
            re.compile(
                r"(?P<subj>[^,;.]+?)\s+was\s+(?:invented|created|developed)\s+by\s+(?P<obj>[^,;.]+)",
                re.IGNORECASE,
            ),
            "created_by",
        ),
        # Pattern 7 — "used_in" / "used_for"
        (
            re.compile(
                r"(?P<subj>[^,;.]+?)\s+is\s+used\s+(?:in|for)\s+(?P<obj>[^,;.]+)",
                re.IGNORECASE,
            ),
            "used_in",
        ),
        # Pattern 9 — "part_of"
        (
            re.compile(
                r"(?P<subj>[^,;.]+?)\s+is\s+part\s+of\s+(?P<obj>[^,;.]+)",
                re.IGNORECASE,
            ),
            "part_of",
        ),
        # Pattern 8 — "greater_than"
        (
            re.compile(
                r"(?P<subj>[^,;.]+?)\s+is\s+greater\s+than\s+(?P<obj>[^,;.]+)",
                re.IGNORECASE,
            ),
            "greater_than",
        ),
        (
            re.compile(
                r"(?P<subj>[^,;.]+?)\s+exceeds\s+(?P<obj>[^,;.]+)",
                re.IGNORECASE,
            ),
            "greater_than",
        ),
        # Pattern 1 — "is a / is an" (must precede generic "is")
        (
            re.compile(
                r"(?P<subj>[^,;.]+?)\s+is\s+an?\s+(?P<obj>[^,;.]+)",
                re.IGNORECASE,
            ),
            "is_a",
        ),
        # Pattern 2 — generic "is" (excludes "is a/an", "is used", "is part of",
        #   "is greater than" to avoid double-matching higher-priority patterns)
        (
            re.compile(
                r"(?P<subj>[^,;.]+?)\s+is\s+(?!an?\s|used\s|part\s+of\s|greater\s+than\s)(?P<obj>[^,;.]+)",
                re.IGNORECASE,
            ),
            "is",
        ),
        # Pattern 3 — "has / have"
        (
            re.compile(
                r"(?P<subj>[^,;.]+?)\s+(?:has|have)\s+(?P<obj>[^,;.]+)",
                re.IGNORECASE,
            ),
            "has",
        ),
        # Pattern 4 — "causes / caused"
        (
            re.compile(
                r"(?P<subj>[^,;.]+?)\s+(?:causes|caused)\s+(?P<obj>[^,;.]+)",
                re.IGNORECASE,
            ),
            "causes",
        ),
        # Pattern 5 — "contains / containing"
        (
            re.compile(
                r"(?P<subj>[^,;.]+?)\s+(?:contains|containing)\s+(?P<obj>[^,;.]+)",
                re.IGNORECASE,
            ),
            "contains",
        ),
    ]

    _ARTICLES = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)

    def __init__(self, max_triples: int = 50000) -> None:
        """Initialise an empty relational memory store.

        Args:
            max_triples: Upper bound on the number of triples stored.
        """
        self.max_triples: int = max_triples
        self.triples: List[KnowledgeTriple] = []
        self.subject_index: Dict[str, List[int]] = defaultdict(list)
        self.predicate_index: Dict[str, List[int]] = defaultdict(list)
        self.object_index: Dict[str, List[int]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_indices(self) -> None:
        """Rebuild all three inverted indices from scratch."""
        self.subject_index = defaultdict(list)
        self.predicate_index = defaultdict(list)
        self.object_index = defaultdict(list)
        for idx, triple in enumerate(self.triples):
            self.subject_index[triple.subject].append(idx)
            self.predicate_index[triple.predicate].append(idx)
            self.object_index[triple.object].append(idx)

    def _add_to_indices(self, idx: int, triple: KnowledgeTriple) -> None:
        """Register a single triple in all three indices."""
        self.subject_index[triple.subject].append(idx)
        self.predicate_index[triple.predicate].append(idx)
        self.object_index[triple.object].append(idx)

    def _find_existing(
        self, subject: str, predicate: str, obj: str
    ) -> Optional[int]:
        """Return the index of an existing identical triple, or ``None``."""
        candidates = set(self.subject_index.get(subject, []))
        candidates &= set(self.predicate_index.get(predicate, []))
        candidates &= set(self.object_index.get(obj, []))
        for idx in candidates:
            t = self.triples[idx]
            if (
                t.subject == subject
                and t.predicate == predicate
                and t.object == obj
            ):
                return idx
        return None

    @classmethod
    def _clean_entity(cls, text: str) -> str:
        """Normalise an entity string extracted from natural language.

        * Strips leading/trailing whitespace.
        * Removes leading articles (a, an, the).
        * Lowercases the result.
        * Truncates to at most 5 words.
        """
        text = text.strip()
        text = cls._ARTICLES.sub("", text)
        text = text.lower().strip()
        words = text.split()
        return " ".join(words[:5])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "unknown",
    ) -> None:
        """Add a triple to the knowledge store.

        If an identical ``(subject, predicate, object)`` triple already exists
        its confidence is increased by 0.1 (capped at 1.0) rather than
        creating a duplicate.  When the store is at maximum capacity the
        lowest-confidence triple is evicted before insertion.

        Args:
            subject: The subject entity.
            predicate: The relationship.
            obj: The object entity.
            confidence: Initial confidence score (0.0–1.0).
            source: Provenance label.
        """
        existing_idx = self._find_existing(subject, predicate, obj)
        if existing_idx is not None:
            self.triples[existing_idx].confidence = min(
                self.triples[existing_idx].confidence + 0.1, 1.0
            )
            self.triples[existing_idx].timestamp = time.time()
            return

        # Evict lowest-confidence triple if at capacity.
        if len(self.triples) >= self.max_triples:
            min_idx = min(
                range(len(self.triples)),
                key=lambda i: self.triples[i].confidence,
            )
            self.triples.pop(min_idx)
            self._rebuild_indices()

        triple = KnowledgeTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source=source,
            timestamp=time.time(),
        )
        idx = len(self.triples)
        self.triples.append(triple)
        self._add_to_indices(idx, triple)

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> List[KnowledgeTriple]:
        """Query the store by any combination of subject, predicate, object.

        At least one parameter must be provided.  Returns all matching triples
        sorted by confidence in descending order.

        Args:
            subject: Filter by subject entity.
            predicate: Filter by predicate.
            obj: Filter by object entity.

        Returns:
            A list of matching ``KnowledgeTriple`` objects, highest confidence
            first.

        Raises:
            ValueError: If no filter parameters are provided.
        """
        if subject is None and predicate is None and obj is None:
            raise ValueError(
                "At least one of subject, predicate, or obj must be provided."
            )

        candidate_sets: List[set] = []
        if subject is not None:
            candidate_sets.append(set(self.subject_index.get(subject, [])))
        if predicate is not None:
            candidate_sets.append(set(self.predicate_index.get(predicate, [])))
        if obj is not None:
            candidate_sets.append(set(self.object_index.get(obj, [])))

        if not candidate_sets:
            return []

        result_indices = candidate_sets[0]
        for s in candidate_sets[1:]:
            result_indices &= s

        results = [self.triples[i] for i in result_indices]
        results.sort(key=lambda t: t.confidence, reverse=True)
        return results

    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract relational triples from *text* using regex heuristics.

        Applies a prioritised set of patterns to each sentence in *text* and
        returns a deduplicated list of ``(subject, predicate, object)`` tuples.
        Entities are cleaned (articles stripped, lowercased, max 5 words).

        Args:
            text: Free-form natural language text.

        Returns:
            A list of ``(subject, predicate, object)`` string tuples.
        """
        if not text or not text.strip():
            return []

        # Split text into sentences for more targeted matching.
        sentences = re.split(r"[.!?]+", text)
        extracted: List[Tuple[str, str, str]] = []
        seen: set = set()

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            for pattern, predicate in self._EXTRACTION_PATTERNS:
                for match in pattern.finditer(sentence):
                    subj = self._clean_entity(match.group("subj"))
                    obj = self._clean_entity(match.group("obj"))
                    if not subj or not obj:
                        continue
                    key = (subj, predicate, obj)
                    if key not in seen:
                        seen.add(key)
                        extracted.append(key)

        return extracted

    def learn_from_text(self, text: str, source: str = "unknown") -> int:
        """Extract and store triples from natural-language *text*.

        Args:
            text: Free-form natural language text.
            source: Provenance label attached to each stored triple.

        Returns:
            The number of triples stored.
        """
        triples = self.extract_triples(text)
        for subj, pred, obj in triples:
            self.store(subj, pred, obj, confidence=0.5, source=source)
        return len(triples)

    def find_connections(
        self, entity_a: str, entity_b: str, max_depth: int = 3
    ) -> List[List[KnowledgeTriple]]:
        """Find paths connecting *entity_a* to *entity_b* via BFS.

        A path is a sequence of triples where each triple shares an entity
        with the next.  The search proceeds up to *max_depth* hops.

        Args:
            entity_a: Starting entity.
            entity_b: Target entity.
            max_depth: Maximum number of hops (triples) in a path.

        Returns:
            A list of paths, where each path is a list of
            ``KnowledgeTriple`` objects connecting the two entities.
        """
        if not entity_a or not entity_b:
            return []

        # BFS: each queue element is (current_entity, path_so_far)
        paths: List[List[KnowledgeTriple]] = []
        queue: deque = deque()
        queue.append((entity_a, []))

        while queue:
            current, path = queue.popleft()

            if len(path) > max_depth:
                continue

            if len(path) > 0 and current == entity_b:
                paths.append(path)
                continue

            if len(path) == max_depth:
                continue

            # Collect neighbouring triples (entity as subject or object).
            neighbour_indices: set = set()
            neighbour_indices.update(self.subject_index.get(current, []))
            neighbour_indices.update(self.object_index.get(current, []))

            visited_entities = {entity_a}
            for t in path:
                visited_entities.add(t.subject)
                visited_entities.add(t.object)

            for idx in neighbour_indices:
                triple = self.triples[idx]
                # Determine the next entity along this edge.
                if triple.subject == current:
                    next_entity = triple.object
                elif triple.object == current:
                    next_entity = triple.subject
                else:
                    continue

                if next_entity in visited_entities and next_entity != entity_b:
                    continue

                queue.append((next_entity, path + [triple]))

        return paths

    def get_related(
        self, entity: str, max_results: int = 10
    ) -> List[KnowledgeTriple]:
        """Return triples where *entity* appears as subject or object.

        Results are sorted by confidence descending and capped at
        *max_results*.

        Args:
            entity: The entity to search for.
            max_results: Maximum number of triples to return.

        Returns:
            A list of ``KnowledgeTriple`` objects.
        """
        indices: set = set()
        indices.update(self.subject_index.get(entity, []))
        indices.update(self.object_index.get(entity, []))

        results = [self.triples[i] for i in indices]
        results.sort(key=lambda t: t.confidence, reverse=True)
        return results[:max_results]

    def analogical_reasoning(
        self, a: str, b: str, c: str
    ) -> List[str]:
        """Perform analogical reasoning: 'A is to B as C is to ?'.

        Finds every predicate *P* such that ``(A, P, B)`` exists in the store,
        then collects all *D* where ``(C, P, D)`` exists.

        Args:
            a: Source subject entity.
            b: Source object entity.
            c: Target subject entity.

        Returns:
            A list of candidate object strings *D*.
        """
        # Find predicates linking A -> B.
        a_indices = set(self.subject_index.get(a, []))
        predicates: List[str] = []
        for idx in a_indices:
            t = self.triples[idx]
            if t.object == b:
                predicates.append(t.predicate)

        if not predicates:
            return []

        # For each predicate, find D where (C, P, D).
        results: List[str] = []
        seen: set = set()
        c_indices = set(self.subject_index.get(c, []))
        for pred in predicates:
            for idx in c_indices:
                t = self.triples[idx]
                if t.predicate == pred and t.object not in seen:
                    seen.add(t.object)
                    results.append(t.object)

        return results

    def prune(self, min_confidence: float = 0.1) -> None:
        """Remove all triples whose confidence is below *min_confidence*.

        Indices are rebuilt after pruning.

        Args:
            min_confidence: Confidence threshold.  Triples strictly below this
                value are removed.
        """
        self.triples = [
            t for t in self.triples if t.confidence >= min_confidence
        ]
        self._rebuild_indices()

    def status(self) -> dict:
        """Return summary statistics about the current knowledge store.

        Returns:
            A dict with keys ``total_triples``, ``unique_subjects``,
            ``unique_predicates``, ``unique_objects``, and
            ``avg_confidence``.
        """
        total = len(self.triples)
        if total == 0:
            return {
                "total_triples": 0,
                "unique_subjects": 0,
                "unique_predicates": 0,
                "unique_objects": 0,
                "avg_confidence": 0.0,
            }

        unique_subjects = len(
            {t.subject for t in self.triples}
        )
        unique_predicates = len(
            {t.predicate for t in self.triples}
        )
        unique_objects = len(
            {t.object for t in self.triples}
        )
        avg_confidence = sum(t.confidence for t in self.triples) / total

        return {
            "total_triples": total,
            "unique_subjects": unique_subjects,
            "unique_predicates": unique_predicates,
            "unique_objects": unique_objects,
            "avg_confidence": avg_confidence,
        }
