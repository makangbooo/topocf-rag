"""Minimal query-document graph construction for HotpotQA diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .title_normalization import mentioning_sentence_indices, normalize_title


class HotpotInvariantError(ValueError):
    """Raised when a HotpotQA record violates a required data invariant."""


@dataclass(frozen=True, slots=True)
class DocumentNode:
    index: int
    title: str
    normalized_title: str
    sentences: tuple[str, ...]

    @property
    def serialized_text(self) -> str:
        return f"{self.title}\n{' '.join(self.sentences)}"


@dataclass(frozen=True, slots=True)
class RetrievalEdge:
    document_index: int
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class TitleMentionEdge:
    source_index: int
    target_index: int
    sentence_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CandidatePath:
    """A length-two q -> source -> target path."""

    source_index: int
    target_index: int
    retrieval_score: float
    retrieval_rank: int
    mention_sentence_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GoldCoverage:
    eligible: bool
    ordered_directed: bool
    reverse_directed: bool
    any_direction: bool
    undirected: bool
    reverse_only: bool
    bidirectional: bool


@dataclass(frozen=True, slots=True)
class QueryDocumentGraph:
    question_id: str
    question: str
    documents: tuple[DocumentNode, ...]
    retrieval_edges: tuple[RetrievalEdge, ...]
    mention_edges: tuple[TitleMentionEdge, ...]
    supporting_title_order: tuple[str, ...]

    @property
    def candidate_paths(self) -> tuple[CandidatePath, ...]:
        retrieval_by_index = {
            edge.document_index: edge for edge in self.retrieval_edges
        }
        paths: list[CandidatePath] = []
        for edge in self.mention_edges:
            retrieval = retrieval_by_index.get(edge.source_index)
            if retrieval is None:
                continue
            paths.append(
                CandidatePath(
                    source_index=edge.source_index,
                    target_index=edge.target_index,
                    retrieval_score=retrieval.score,
                    retrieval_rank=retrieval.rank,
                    mention_sentence_indices=edge.sentence_indices,
                )
            )
        return tuple(paths)

    @property
    def gold_paths(self) -> tuple[CandidatePath, ...]:
        if len(self.supporting_title_order) != 2:
            return ()
        gold = set(self.supporting_title_order)
        return tuple(
            path
            for path in self.candidate_paths
            if {
                self.documents[path.source_index].normalized_title,
                self.documents[path.target_index].normalized_title,
            }
            == gold
        )

    @property
    def natural_negative_paths(self) -> tuple[CandidatePath, ...]:
        """Return real retrieval-plus-mention paths that do not cover both golds."""

        gold = set(self.supporting_title_order)
        return tuple(
            path
            for path in self.candidate_paths
            if not gold.issubset(
                {
                    self.documents[path.source_index].normalized_title,
                    self.documents[path.target_index].normalized_title,
                }
            )
        )


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise HotpotInvariantError(f"{field} must be a list or tuple")
    return value


def validate_hotpot_example(example: Mapping[str, Any]) -> None:
    """Validate fields needed to build the Phase 0 graph.

    Error messages identify fields and indices but deliberately never include
    question, title, sentence, answer, or supporting-fact text.
    """

    if not isinstance(example, Mapping):
        raise HotpotInvariantError("example must be a mapping")
    if not isinstance(example.get("_id"), str) or not example["_id"]:
        raise HotpotInvariantError("_id must be a non-empty string")
    if not isinstance(example.get("question"), str) or not example["question"].strip():
        raise HotpotInvariantError("question must be a non-empty string")

    context = _require_sequence(example.get("context"), "context")
    if not context:
        raise HotpotInvariantError("context must contain at least one document")

    normalized_to_index: dict[str, int] = {}
    sentence_counts: list[int] = []
    for document_index, document in enumerate(context):
        pair = _require_sequence(document, f"context[{document_index}]")
        if len(pair) != 2:
            raise HotpotInvariantError(
                f"context[{document_index}] must contain title and sentences"
            )
        title, sentences_value = pair
        if not isinstance(title, str):
            raise HotpotInvariantError(f"context[{document_index}].title must be str")
        try:
            normalized = normalize_title(title)
        except (TypeError, ValueError) as error:
            raise HotpotInvariantError(
                f"context[{document_index}].title is invalid"
            ) from error
        if normalized in normalized_to_index:
            raise HotpotInvariantError("context normalized titles must be unique")
        normalized_to_index[normalized] = document_index

        sentences = _require_sequence(
            sentences_value, f"context[{document_index}].sentences"
        )
        for sentence_index, sentence in enumerate(sentences):
            if not isinstance(sentence, str):
                raise HotpotInvariantError(
                    f"context[{document_index}].sentences[{sentence_index}] must be str"
                )
        sentence_counts.append(len(sentences))

    supporting_facts = _require_sequence(
        example.get("supporting_facts"), "supporting_facts"
    )
    if not supporting_facts:
        raise HotpotInvariantError("supporting_facts must not be empty")
    for fact_index, fact in enumerate(supporting_facts):
        pair = _require_sequence(fact, f"supporting_facts[{fact_index}]")
        if len(pair) != 2:
            raise HotpotInvariantError(
                f"supporting_facts[{fact_index}] must contain title and sentence index"
            )
        title, sentence_index = pair
        if not isinstance(title, str):
            raise HotpotInvariantError(
                f"supporting_facts[{fact_index}].title must be str"
            )
        if not isinstance(sentence_index, int) or isinstance(sentence_index, bool):
            raise HotpotInvariantError(
                f"supporting_facts[{fact_index}].sentence_index must be int"
            )
        try:
            normalized = normalize_title(title)
        except (TypeError, ValueError) as error:
            raise HotpotInvariantError(
                f"supporting_facts[{fact_index}].title is invalid"
            ) from error
        document_index = normalized_to_index.get(normalized)
        if document_index is None:
            raise HotpotInvariantError(
                f"supporting_facts[{fact_index}] references a missing context title"
            )
        if not 0 <= sentence_index < sentence_counts[document_index]:
            raise HotpotInvariantError(
                f"supporting_facts[{fact_index}].sentence_index is out of range"
            )


def document_nodes(example: Mapping[str, Any]) -> tuple[DocumentNode, ...]:
    validate_hotpot_example(example)
    return tuple(
        DocumentNode(
            index=index,
            title=document[0],
            normalized_title=normalize_title(document[0]),
            sentences=tuple(document[1]),
        )
        for index, document in enumerate(example["context"])
    )


def supporting_title_order(example: Mapping[str, Any]) -> tuple[str, ...]:
    """Return unique normalized supporting titles in first-appearance order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for title, _sentence_index in example["supporting_facts"]:
        normalized = normalize_title(title)
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return tuple(ordered)


def build_title_mention_edges(
    documents: Sequence[DocumentNode],
) -> tuple[TitleMentionEdge, ...]:
    """Build directed d_i -> d_j edges from literal title mentions."""

    edges: list[TitleMentionEdge] = []
    for source in documents:
        for target in documents:
            if source.index == target.index:
                continue
            sentence_indices = mentioning_sentence_indices(
                source.sentences, target.title
            )
            if sentence_indices:
                edges.append(
                    TitleMentionEdge(
                        source_index=source.index,
                        target_index=target.index,
                        sentence_indices=sentence_indices,
                    )
                )
    return tuple(edges)


def build_query_document_graph(
    example: Mapping[str, Any],
    retrieval_scores: Sequence[float] | None = None,
    *,
    retrieval_top_k: int | None = None,
) -> QueryDocumentGraph:
    """Build the minimal q -> d_i -> d_j graph for one HotpotQA item."""

    documents = document_nodes(example)
    if retrieval_scores is None:
        scores = [0.0] * len(documents)
    else:
        if len(retrieval_scores) != len(documents):
            raise HotpotInvariantError(
                "retrieval score count must equal context document count"
            )
        scores = []
        for index, score in enumerate(retrieval_scores):
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise HotpotInvariantError(f"retrieval_scores[{index}] must be numeric")
            numeric_score = float(score)
            if numeric_score != numeric_score or numeric_score in (float("inf"), -float("inf")):
                raise HotpotInvariantError(f"retrieval_scores[{index}] must be finite")
            scores.append(numeric_score)

    if retrieval_top_k is None:
        retrieval_top_k = len(documents)
    if not isinstance(retrieval_top_k, int) or isinstance(retrieval_top_k, bool):
        raise HotpotInvariantError("retrieval_top_k must be int or None")
    if not 1 <= retrieval_top_k <= len(documents):
        raise HotpotInvariantError(
            "retrieval_top_k must be between 1 and the context document count"
        )

    ranked_indices = sorted(range(len(documents)), key=lambda i: (-scores[i], i))
    retrieval_edges = tuple(
        RetrievalEdge(document_index=index, score=scores[index], rank=rank)
        for rank, index in enumerate(ranked_indices[:retrieval_top_k], start=1)
    )
    return QueryDocumentGraph(
        question_id=example["_id"],
        question=example["question"],
        documents=documents,
        retrieval_edges=retrieval_edges,
        mention_edges=build_title_mention_edges(documents),
        supporting_title_order=supporting_title_order(example),
    )


def gold_coverage(graph: QueryDocumentGraph) -> GoldCoverage:
    """Measure gold-title connection coverage under explicit direction rules.

    ``ordered_directed`` uses the first-appearance order of unique titles in
    ``supporting_facts``. HotpotQA does not document that order as a semantic
    reasoning-chain direction, so this is a reproducible Phase 0 proxy only.
    ``any_direction`` counts real directed candidate paths in either direction.
    ``undirected`` ignores the mention edge direction while retaining a real
    mention edge and a retrieved endpoint.
    """

    if len(graph.supporting_title_order) != 2:
        return GoldCoverage(False, False, False, False, False, False, False)

    first, second = graph.supporting_title_order
    index_by_title = {document.normalized_title: document.index for document in graph.documents}
    if first not in index_by_title or second not in index_by_title:
        return GoldCoverage(False, False, False, False, False, False, False)
    first_index = index_by_title[first]
    second_index = index_by_title[second]
    edge_pairs = {
        (edge.source_index, edge.target_index) for edge in graph.mention_edges
    }
    retrieved = {edge.document_index for edge in graph.retrieval_edges}

    forward_edge = (first_index, second_index) in edge_pairs
    reverse_edge = (second_index, first_index) in edge_pairs
    ordered_directed = first_index in retrieved and forward_edge
    reverse_directed = second_index in retrieved and reverse_edge
    any_direction = ordered_directed or reverse_directed
    undirected = (
        (first_index in retrieved or second_index in retrieved)
        and (forward_edge or reverse_edge)
    )
    return GoldCoverage(
        eligible=True,
        ordered_directed=ordered_directed,
        reverse_directed=reverse_directed,
        any_direction=any_direction,
        undirected=undirected,
        reverse_only=reverse_directed and not ordered_directed,
        bidirectional=ordered_directed and reverse_directed,
    )
