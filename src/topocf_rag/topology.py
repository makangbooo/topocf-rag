"""Validated evidence-topology objects for topology counterfactuals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
from itertools import permutations
import json
import math
import re
from typing import Final, Literal


RelationType = Literal["retrieval", "title_mention"]
TopologySignature = tuple[int, tuple[tuple[str, str, str], ...]]
DegreeProfile = tuple[
    tuple[str, tuple[tuple[str, int, int], ...]], ...
]

_RELATIONS: Final[tuple[RelationType, ...]] = ("retrieval", "title_mention")
_RELATION_ORDER: Final[dict[str, int]] = {
    relation: index for index, relation in enumerate(_RELATIONS)
}
_DOCUMENT_ALIAS = re.compile(r"d(0|[1-9][0-9]*)\Z")


class TopologyInvariantError(ValueError):
    """Raised when an evidence topology violates its structural contract."""


def _is_document_alias(value: str) -> bool:
    return bool(_DOCUMENT_ALIAS.fullmatch(value))


def _endpoint_order(endpoint: str) -> int:
    if endpoint == "q":
        return -1
    match = _DOCUMENT_ALIAS.fullmatch(endpoint)
    if match is None:
        raise TopologyInvariantError("edge endpoint must be q or a neutral document alias")
    return int(match.group(1))


def _edge_order(edge: TypedEdge) -> tuple[int, int, int]:
    return (
        _RELATION_ORDER[edge.relation],
        _endpoint_order(edge.source),
        _endpoint_order(edge.target),
    )


def _document_text_sha256(title: str, sentences: tuple[str, ...]) -> str:
    payload = json.dumps(
        [title, list(sentences)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    """An immutable document node with a neutral, role-free alias."""

    alias: str
    title: str
    sentences: tuple[str, ...]
    text_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.alias, str) or not _is_document_alias(self.alias):
            raise TopologyInvariantError(
                "document alias must use the neutral form d0, d1, ..."
            )
        if not isinstance(self.title, str) or not self.title.strip():
            raise TopologyInvariantError("document title must be a non-empty string")
        if not isinstance(self.sentences, tuple):
            raise TopologyInvariantError("document sentences must be an immutable tuple")
        if any(not isinstance(sentence, str) for sentence in self.sentences):
            raise TopologyInvariantError("every document sentence must be a string")
        object.__setattr__(
            self,
            "text_sha256",
            _document_text_sha256(self.title, self.sentences),
        )

    @property
    def serialized_text(self) -> str:
        return f"{self.title}\n{' '.join(self.sentences)}"


@dataclass(frozen=True, slots=True)
class TypedEdge:
    """A directed typed edge; provenance is metadata, not graph structure."""

    relation: RelationType
    source: str
    target: str
    observed: bool
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.relation not in _RELATION_ORDER:
            raise TopologyInvariantError("unsupported edge relation")
        if not isinstance(self.source, str) or not isinstance(self.target, str):
            raise TopologyInvariantError("edge endpoints must be strings")
        if not isinstance(self.observed, bool):
            raise TopologyInvariantError("edge observed flag must be bool")
        if not isinstance(self.provenance, tuple) or any(
            not isinstance(item, str) or not item for item in self.provenance
        ):
            raise TopologyInvariantError(
                "edge provenance must be an immutable tuple of non-empty strings"
            )

        if self.relation == "retrieval":
            if self.source != "q" or not _is_document_alias(self.target):
                raise TopologyInvariantError(
                    "retrieval edges must be directed from q to a document"
                )
        elif not _is_document_alias(self.source) or not _is_document_alias(
            self.target
        ):
            raise TopologyInvariantError(
                "title-mention edges must connect two document aliases"
            )

        if self.source == self.target:
            raise TopologyInvariantError("self-loop edges are not allowed")

    @property
    def structural_key(self) -> tuple[str, str, str]:
        """Return the provenance-free identity of this edge."""

        return (self.relation, self.source, self.target)


@dataclass(frozen=True, slots=True)
class EvidenceTopology:
    """A question-conditioned typed evidence subgraph with implicit node q."""

    question_id: str
    question: str
    documents: tuple[EvidenceDocument, ...]
    edges: tuple[TypedEdge, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.question_id, str) or not self.question_id:
            raise TopologyInvariantError("question_id must be a non-empty string")
        if not isinstance(self.question, str) or not self.question.strip():
            raise TopologyInvariantError("question must be a non-empty string")
        if not isinstance(self.documents, tuple) or not self.documents:
            raise TopologyInvariantError("documents must be a non-empty tuple")
        if len(self.documents) > 6:
            raise TopologyInvariantError(
                "at most six documents are supported by canonicalization"
            )
        if any(not isinstance(document, EvidenceDocument) for document in self.documents):
            raise TopologyInvariantError("documents must contain EvidenceDocument values")

        expected_aliases = tuple(f"d{index}" for index in range(len(self.documents)))
        actual_aliases = tuple(document.alias for document in self.documents)
        if actual_aliases != expected_aliases:
            raise TopologyInvariantError(
                "documents must be ordered with consecutive aliases d0, d1, ..."
            )

        if not isinstance(self.edges, tuple):
            raise TopologyInvariantError("edges must be an immutable tuple")
        if any(not isinstance(edge, TypedEdge) for edge in self.edges):
            raise TopologyInvariantError("edges must contain TypedEdge values")
        if self.edges != tuple(sorted(self.edges, key=_edge_order)):
            raise TopologyInvariantError(
                "edges must use canonical relation/source/target order"
            )

        valid_documents = set(expected_aliases)
        identities: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            endpoints = {edge.target}
            if edge.source != "q":
                endpoints.add(edge.source)
            if not endpoints.issubset(valid_documents):
                raise TopologyInvariantError(
                    "edge endpoint is not declared by this topology"
                )
            if edge.structural_key in identities:
                raise TopologyInvariantError(
                    "duplicate typed edges are not allowed, regardless of provenance"
                )
            identities.add(edge.structural_key)

    @property
    def document_text_multiset(self) -> Counter[tuple[str, str]]:
        """Return alias-plus-text identities used by matched-pair validation."""

        return Counter(
            (document.alias, document.text_sha256) for document in self.documents
        )

    @property
    def relation_multiset(self) -> tuple[tuple[str, int], ...]:
        counts = Counter(edge.relation for edge in self.edges)
        return tuple(
            (relation, counts[relation])
            for relation in _RELATIONS
            if counts[relation]
        )

    @property
    def degree_profile(self) -> DegreeProfile:
        """Return per-node, per-relation directed in/out degrees."""

        nodes = ("q", *(document.alias for document in self.documents))
        profile: list[tuple[str, tuple[tuple[str, int, int], ...]]] = []
        for node in nodes:
            relation_degrees = tuple(
                (
                    relation,
                    sum(
                        edge.relation == relation and edge.target == node
                        for edge in self.edges
                    ),
                    sum(
                        edge.relation == relation and edge.source == node
                        for edge in self.edges
                    ),
                )
                for relation in _RELATIONS
            )
            profile.append((node, relation_degrees))
        return tuple(profile)

    @property
    def canonical_signature(self) -> TopologySignature:
        """Return a document-label-independent colored directed-graph signature.

        The question node is fixed, all document nodes share one color, and edge
        relations are edge colors. Observed flags and provenance are excluded.
        """

        aliases = tuple(document.alias for document in self.documents)
        best: tuple[tuple[str, str, str], ...] | None = None
        for old_aliases_in_new_order in permutations(aliases):
            relabel = {
                old_alias: f"d{new_index}"
                for new_index, old_alias in enumerate(old_aliases_in_new_order)
            }
            encoded = tuple(
                sorted(
                    (
                        edge.relation,
                        edge.source if edge.source == "q" else relabel[edge.source],
                        relabel[edge.target],
                    )
                    for edge in self.edges
                )
            )
            if best is None or encoded < best:
                best = encoded
        assert best is not None
        return (len(self.documents), best)


def validate_matched_pair(
    positive: EvidenceTopology,
    negative: EvidenceTopology,
    *,
    positive_token_count: int | None = None,
    negative_token_count: int | None = None,
    max_token_relative_difference: float = 0.05,
) -> float | None:
    """Validate topology-only matching and return the optional token gap.

    Token difference is measured relative to the positive serialization, which
    is the preregistered matching denominator.
    """

    if not isinstance(positive, EvidenceTopology) or not isinstance(
        negative, EvidenceTopology
    ):
        raise TopologyInvariantError(
            "matched-pair members must both be EvidenceTopology values"
        )
    if (positive.question_id, positive.question) != (
        negative.question_id,
        negative.question,
    ):
        raise TopologyInvariantError("matched-pair questions must be identical")
    if positive.document_text_multiset != negative.document_text_multiset:
        raise TopologyInvariantError(
            "matched-pair alias and document-text multisets must be identical"
        )
    if len(positive.edges) != len(negative.edges):
        raise TopologyInvariantError("matched-pair edge counts must be identical")
    if positive.relation_multiset != negative.relation_multiset:
        raise TopologyInvariantError(
            "matched-pair relation-type multisets must be identical"
        )

    if (positive_token_count is None) != (negative_token_count is None):
        raise TopologyInvariantError(
            "both exact token counts must be supplied together"
        )
    if (
        not isinstance(max_token_relative_difference, (int, float))
        or isinstance(max_token_relative_difference, bool)
        or not math.isfinite(float(max_token_relative_difference))
        or not 0 <= float(max_token_relative_difference) <= 1
    ):
        raise TopologyInvariantError(
            "max token relative difference must be finite and between zero and one"
        )
    if positive_token_count is None:
        return None
    if (
        not isinstance(positive_token_count, int)
        or isinstance(positive_token_count, bool)
        or positive_token_count <= 0
        or not isinstance(negative_token_count, int)
        or isinstance(negative_token_count, bool)
        or negative_token_count <= 0
    ):
        raise TopologyInvariantError("exact token counts must be positive integers")

    relative_difference = abs(negative_token_count - positive_token_count) / (
        positive_token_count
    )
    if relative_difference > float(max_token_relative_difference):
        raise TopologyInvariantError("matched-pair token difference exceeds the limit")
    return relative_difference
