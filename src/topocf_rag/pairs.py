"""Deterministic, content-free Phase 1 A+ pair construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Literal, Mapping, Sequence

from .graph import QueryDocumentGraph, build_query_document_graph
from .title_normalization import normalize_title
from .topology import TypedEdge


PAIR_SCHEMA_VERSION = 1
PAIR_SEED = 20260712

PairStratum = Literal[
    "synthetic_common", "all_observed_rewire", "t2_all_observed"
]
PairVariant = Literal[
    "t1", "t2", "t3", "fork", "double_collider", "t3_all_observed"
]

_COMMON_VARIANTS: tuple[PairVariant, ...] = (
    "t1",
    "t2",
    "t3",
    "fork",
    "double_collider",
)
_VALID_VARIANTS: dict[PairStratum, frozenset[PairVariant]] = {
    "synthetic_common": frozenset(_COMMON_VARIANTS),
    "all_observed_rewire": frozenset(("t3_all_observed",)),
    "t2_all_observed": frozenset(("t2",)),
}
_STRATUM_ORDER = {
    "synthetic_common": 0,
    "all_observed_rewire": 1,
    "t2_all_observed": 2,
}
_VARIANT_ORDER = {
    "t1": 0,
    "t2": 1,
    "t3": 2,
    "fork": 3,
    "double_collider": 4,
    "t3_all_observed": 5,
}


class PairInvariantError(ValueError):
    """Raised when a Phase 1 pair violates the frozen A+ contract."""


def _typed_edge_order(edge: TypedEdge) -> tuple[str, str, str]:
    return edge.structural_key


def _sorted_edges(edges: Iterable[TypedEdge]) -> tuple[TypedEdge, ...]:
    return tuple(sorted(edges, key=_typed_edge_order))


def _stable_hexdigest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_edge_schema(edges: tuple[TypedEdge, ...], aliases: set[str]) -> None:
    if len(edges) != 4:
        raise PairInvariantError("every topology must contain exactly four edges")
    relation_counts = Counter(edge.relation for edge in edges)
    if relation_counts != {"retrieval": 2, "title_mention": 2}:
        raise PairInvariantError(
            "every topology must contain two retrieval and two title-mention edges"
        )
    if len({edge.structural_key for edge in edges}) != len(edges):
        raise PairInvariantError("duplicate typed edges are not allowed")
    for edge in edges:
        document_endpoints = {edge.target}
        if edge.source != "q":
            document_endpoints.add(edge.source)
        if not document_endpoints.issubset(aliases):
            raise PairInvariantError("typed edge references an undeclared document alias")


@dataclass(frozen=True, slots=True)
class Phase1Pair:
    """A content-free matched pair suitable for a JSON manifest."""

    pair_id: str
    base_id: str
    qid: str
    stratum: PairStratum
    variant: PairVariant
    context_indices: tuple[int, int, int, int]
    neutral_alias_mapping: tuple[tuple[str, int], ...]
    sentence_indices: tuple[tuple[str, tuple[int, ...]], ...]
    positive_edges: tuple[TypedEdge, ...]
    negative_edges: tuple[TypedEdge, ...]

    def __post_init__(self) -> None:
        if not self.pair_id.startswith("pair-") or not self.base_id.startswith("base-"):
            raise PairInvariantError("pair and base IDs must use stable hashed prefixes")
        if not isinstance(self.qid, str) or not self.qid:
            raise PairInvariantError("qid must be a non-empty string")
        if self.stratum not in _VALID_VARIANTS:
            raise PairInvariantError("unknown pair stratum")
        if self.variant not in _VALID_VARIANTS[self.stratum]:
            raise PairInvariantError("variant is not valid for its stratum")

        if len(self.context_indices) != 4 or len(set(self.context_indices)) != 4:
            raise PairInvariantError("every pair must contain four distinct documents")
        expected_aliases = tuple(f"d{index}" for index in range(4))
        actual_aliases = tuple(alias for alias, _index in self.neutral_alias_mapping)
        if actual_aliases != expected_aliases:
            raise PairInvariantError("neutral aliases must be consecutive and ordered")
        mapped_indices = tuple(index for _alias, index in self.neutral_alias_mapping)
        if mapped_indices != self.context_indices:
            raise PairInvariantError(
                "context indices must follow the neutral alias mapping order"
            )
        if tuple(alias for alias, _indices in self.sentence_indices) != expected_aliases:
            raise PairInvariantError("sentence selections must cover every neutral alias")
        for _alias, indices in self.sentence_indices:
            if not indices or tuple(sorted(set(indices))) != indices:
                raise PairInvariantError(
                    "sentence indices must be non-empty, sorted, and unique"
                )
            if any(
                not isinstance(index, int) or isinstance(index, bool) or index < 0
                for index in indices
            ):
                raise PairInvariantError("sentence indices must be non-negative integers")

        alias_set = set(expected_aliases)
        _validate_edge_schema(self.positive_edges, alias_set)
        _validate_edge_schema(self.negative_edges, alias_set)
        if not all(edge.observed for edge in self.positive_edges):
            raise PairInvariantError("all positive edges must be observed")
        if self.variant == "t2" and not all(
            edge.observed for edge in self.negative_edges
        ):
            raise PairInvariantError("every T2 edge must be observed")
        if self.variant == "t3_all_observed" and not all(
            edge.observed for edge in self.negative_edges
        ):
            raise PairInvariantError("every all-observed T3 edge must be observed")

    @property
    def sort_key(self) -> tuple[str, str, int, int, str]:
        return (
            self.qid,
            self.base_id,
            _STRATUM_ORDER[self.stratum],
            _VARIANT_ORDER[self.variant],
            self.pair_id,
        )

    def to_manifest_record(self) -> dict[str, Any]:
        def edge_payload(edge: TypedEdge) -> dict[str, Any]:
            return {
                "relation": edge.relation,
                "source": edge.source,
                "target": edge.target,
                "observed": edge.observed,
            }

        return {
            "pair_id": self.pair_id,
            "base_id": self.base_id,
            "qid": self.qid,
            "stratum": self.stratum,
            "variant": self.variant,
            "context_indices": list(self.context_indices),
            "neutral_alias_mapping": dict(self.neutral_alias_mapping),
            "sentence_indices": {
                alias: list(indices) for alias, indices in self.sentence_indices
            },
            "positive": {
                "typed_edges": [edge_payload(edge) for edge in self.positive_edges]
            },
            "negative": {
                "typed_edges": [edge_payload(edge) for edge in self.negative_edges]
            },
        }


def _neutral_alias_mapping(
    *, qid: str, endpoints: tuple[int, int, int, int], seed: int
) -> tuple[tuple[str, int], ...]:
    if len(set(endpoints)) != 4:
        raise PairInvariantError("base endpoints must contain four distinct documents")
    ranked_indices = sorted(
        endpoints,
        key=lambda context_index: (
            _stable_hexdigest(
                {
                    "seed": seed,
                    "qid": qid,
                    "base_endpoints": endpoints,
                    "context_index": context_index,
                }
            ),
            context_index,
        ),
    )
    return tuple(
        (f"d{alias_index}", context_index)
        for alias_index, context_index in enumerate(ranked_indices)
    )


def _sentence_selection_by_context_index(
    *,
    example: Mapping[str, Any],
    graph: QueryDocumentGraph,
    endpoints: tuple[int, int, int, int],
    observed_mention_edges: Sequence[tuple[int, int]],
) -> dict[int, tuple[int, ...]]:
    endpoint_set = set(endpoints)
    selected: dict[int, set[int]] = {index: set() for index in endpoints}
    index_by_title = {
        document.normalized_title: document.index for document in graph.documents
    }

    for title, sentence_index in example["supporting_facts"]:
        context_index = index_by_title[normalize_title(title)]
        if context_index in endpoint_set:
            selected[context_index].add(int(sentence_index))

    evidence_by_edge = {
        (edge.source_index, edge.target_index): edge.sentence_indices
        for edge in graph.mention_edges
    }
    for edge in observed_mention_edges:
        evidence_indices = evidence_by_edge.get(edge)
        if not evidence_indices:
            raise PairInvariantError("a base mention edge lacks observed sentence evidence")
        selected[edge[0]].update(evidence_indices)

    result: dict[int, tuple[int, ...]] = {}
    for context_index in endpoints:
        sentence_count = len(graph.documents[context_index].sentences)
        if not selected[context_index]:
            if sentence_count == 0:
                raise PairInvariantError(
                    "a selected document has no sentence available for fallback"
                )
            selected[context_index].add(0)
        if any(index >= sentence_count for index in selected[context_index]):
            raise PairInvariantError("selected sentence index is out of range")
        result[context_index] = tuple(sorted(selected[context_index]))
    return result


def _retrieval(alias: str) -> TypedEdge:
    return TypedEdge("retrieval", "q", alias, True)


def _mention(source: str, target: str, *, observed: bool) -> TypedEdge:
    return TypedEdge("title_mention", source, target, observed)


def _edge_sets_for_base(
    alias_by_context_index: Mapping[int, str],
    *,
    endpoints: tuple[int, int, int, int],
) -> tuple[tuple[TypedEdge, ...], dict[PairVariant, tuple[TypedEdge, ...]]]:
    a_index, b_index, c_index, d_index = endpoints
    a = alias_by_context_index[a_index]
    b = alias_by_context_index[b_index]
    c = alias_by_context_index[c_index]
    d = alias_by_context_index[d_index]

    positive = _sorted_edges(
        (
            _retrieval(a),
            _retrieval(c),
            _mention(a, b, observed=True),
            _mention(c, d, observed=True),
        )
    )
    negatives: dict[PairVariant, tuple[TypedEdge, ...]] = {
        "t1": _sorted_edges(
            (
                _retrieval(a),
                _retrieval(c),
                _mention(b, a, observed=False),
                _mention(c, d, observed=True),
            )
        ),
        "t2": _sorted_edges(
            (
                _retrieval(b),
                _retrieval(c),
                _mention(a, b, observed=True),
                _mention(c, d, observed=True),
            )
        ),
        "t3": _sorted_edges(
            (
                _retrieval(a),
                _retrieval(c),
                _mention(a, d, observed=False),
                _mention(c, b, observed=False),
            )
        ),
        "fork": _sorted_edges(
            (
                _retrieval(a),
                _retrieval(c),
                _mention(c, b, observed=False),
                _mention(c, d, observed=True),
            )
        ),
        "double_collider": _sorted_edges(
            (
                _retrieval(a),
                _retrieval(c),
                _mention(b, a, observed=False),
                _mention(d, c, observed=False),
            )
        ),
        "t3_all_observed": _sorted_edges(
            (
                _retrieval(a),
                _retrieval(c),
                _mention(a, d, observed=True),
                _mention(c, b, observed=True),
            )
        ),
    }
    return positive, negatives


def _base_material(
    *,
    example: Mapping[str, Any],
    graph: QueryDocumentGraph,
    endpoints: tuple[int, int, int, int],
    stratum: PairStratum,
    seed: int,
) -> tuple[
    str,
    tuple[int, int, int, int],
    tuple[tuple[str, int], ...],
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[TypedEdge, ...],
    dict[PairVariant, tuple[TypedEdge, ...]],
]:
    qid = str(example["_id"])
    a_index, b_index, c_index, d_index = endpoints
    if stratum == "all_observed_rewire":
        sentence_selection_policy = "base_and_cross_observed_mentions"
        observed_mention_edges = (
            (a_index, b_index),
            (c_index, d_index),
            (a_index, d_index),
            (c_index, b_index),
        )
    else:
        sentence_selection_policy = "base_observed_mentions"
        observed_mention_edges = (
            (a_index, b_index),
            (c_index, d_index),
        )
    base_id = "base-" + _stable_hexdigest(
        {
            "seed": seed,
            "qid": qid,
            "endpoints": endpoints,
            "stratum": stratum,
            "sentence_selection_policy": sentence_selection_policy,
        }
    )[:24]
    aliases = _neutral_alias_mapping(qid=qid, endpoints=endpoints, seed=seed)
    alias_by_context_index = {
        context_index: alias for alias, context_index in aliases
    }
    context_indices = tuple(index for _alias, index in aliases)
    selection_by_index = _sentence_selection_by_context_index(
        example=example,
        graph=graph,
        endpoints=endpoints,
        observed_mention_edges=observed_mention_edges,
    )
    sentence_indices = tuple(
        (alias, selection_by_index[context_index])
        for alias, context_index in aliases
    )
    positive, negatives = _edge_sets_for_base(
        alias_by_context_index, endpoints=endpoints
    )
    return (
        base_id,
        context_indices,  # type: ignore[return-value]
        aliases,
        sentence_indices,
        positive,
        negatives,
    )


def generate_question_pairs(
    example: Mapping[str, Any],
    retrieval_scores: Sequence[float],
    *,
    retrieval_top_k: int = 10,
    seed: int = PAIR_SEED,
) -> tuple[Phase1Pair, ...]:
    """Generate every deduplicated A+ endpoint operation for one question."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise PairInvariantError("seed must be an integer")
    if not isinstance(retrieval_top_k, int) or retrieval_top_k < 1:
        raise PairInvariantError("retrieval_top_k must be a positive integer")
    top_k = min(retrieval_top_k, len(example["context"]))
    graph = build_query_document_graph(
        example, retrieval_scores, retrieval_top_k=top_k
    )
    if len(graph.supporting_title_order) != 2:
        return ()
    index_by_title = {
        document.normalized_title: document.index for document in graph.documents
    }
    first_index, second_index = (
        index_by_title[title] for title in graph.supporting_title_order
    )
    mention_edges = {
        (edge.source_index, edge.target_index) for edge in graph.mention_edges
    }
    forward = (first_index, second_index) in mention_edges
    reverse = (second_index, first_index) in mention_edges
    if forward == reverse:
        return ()
    a_index, b_index = (
        (first_index, second_index) if forward else (second_index, first_index)
    )
    retrieved = {edge.document_index for edge in graph.retrieval_edges}
    if not {a_index, b_index}.issubset(retrieved):
        return ()

    qid = str(example["_id"])
    pairs: list[Phase1Pair] = []
    seen_operations: set[tuple[PairStratum, tuple[int, int, int, int]]] = set()
    material_by_operation: dict[
        tuple[PairStratum, tuple[int, int, int, int]],
        tuple[
            str,
            tuple[int, int, int, int],
            tuple[tuple[str, int], ...],
            tuple[tuple[str, tuple[int, ...]], ...],
            tuple[TypedEdge, ...],
            dict[PairVariant, tuple[TypedEdge, ...]],
        ],
    ] = {}

    def add_pair(
        stratum: PairStratum,
        variant: PairVariant,
        endpoints: tuple[int, int, int, int],
    ) -> None:
        material_key = (stratum, endpoints)
        material = material_by_operation.get(material_key)
        if material is None:
            material = _base_material(
                example=example,
                graph=graph,
                endpoints=endpoints,
                stratum=stratum,
                seed=seed,
            )
            material_by_operation[material_key] = material
        (
            base_id,
            context_indices,
            aliases,
            sentence_indices,
            positive,
            negatives,
        ) = material
        pair_id = "pair-" + _stable_hexdigest(
            {"base_id": base_id, "stratum": stratum, "variant": variant}
        )[:24]
        pairs.append(
            Phase1Pair(
                pair_id=pair_id,
                base_id=base_id,
                qid=qid,
                stratum=stratum,
                variant=variant,
                context_indices=context_indices,
                neutral_alias_mapping=aliases,
                sentence_indices=sentence_indices,
                positive_edges=positive,
                negative_edges=negatives[variant],
            )
        )

    for c_index, d_index in sorted(mention_edges):
        endpoints = (a_index, b_index, c_index, d_index)
        if len(set(endpoints)) != 4 or c_index not in retrieved:
            continue

        cross_first = (a_index, d_index)
        cross_second = (c_index, b_index)
        cross_real_count = int(cross_first in mention_edges) + int(
            cross_second in mention_edges
        )
        reverse_control = (d_index, c_index)
        is_common = cross_real_count == 0 and reverse_control not in mention_edges
        if is_common:
            common_key = ("synthetic_common", endpoints)
            if common_key not in seen_operations:
                seen_operations.add(common_key)
                for variant in _COMMON_VARIANTS:
                    add_pair("synthetic_common", variant, endpoints)
        else:
            broad_key = ("t2_all_observed", endpoints)
            if broad_key not in seen_operations:
                seen_operations.add(broad_key)
                add_pair("t2_all_observed", "t2", endpoints)

        if cross_real_count == 2:
            observed_rewire_key = ("all_observed_rewire", endpoints)
            if observed_rewire_key not in seen_operations:
                seen_operations.add(observed_rewire_key)
                add_pair(
                    "all_observed_rewire", "t3_all_observed", endpoints
                )

    result = tuple(sorted(pairs, key=lambda pair: pair.sort_key))
    if len({pair.pair_id for pair in result}) != len(result):
        raise PairInvariantError("pair ID collision or duplicate endpoint operation")
    return result


def summarize_pairs(pairs: Sequence[Phase1Pair]) -> dict[str, Any]:
    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in _STRATUM_ORDER:
        selected = [pair for pair in pairs if pair.stratum == stratum]
        variants = Counter(pair.variant for pair in selected)
        by_stratum[stratum] = {
            "pair_count": len(selected),
            "base_count": len({(pair.qid, pair.base_id) for pair in selected}),
            "question_count": len({pair.qid for pair in selected}),
            "variant_pair_counts": dict(sorted(variants.items())),
        }
    common_t2 = [
        pair
        for pair in pairs
        if pair.stratum == "synthetic_common" and pair.variant == "t2"
    ]
    broad_t2 = [
        pair
        for pair in pairs
        if pair.stratum == "t2_all_observed" and pair.variant == "t2"
    ]
    combined_t2 = (*common_t2, *broad_t2)
    return {
        "pair_count": len(pairs),
        "base_count": len({(pair.qid, pair.base_id) for pair in pairs}),
        "question_count": len({pair.qid for pair in pairs}),
        "by_stratum": by_stratum,
        "combined_t2": {
            "pair_count": len(combined_t2),
            "base_count": len(
                {(pair.qid, pair.base_id) for pair in combined_t2}
            ),
            "question_count": len({pair.qid for pair in combined_t2}),
            "source_pair_counts": {
                "synthetic_common": len(common_t2),
                "t2_all_observed_extra": len(broad_t2),
            },
        },
    }


def build_pair_manifest(
    pairs: Sequence[Phase1Pair],
    *,
    official_split: str,
    source_sha256: str,
    ids_sha256: str,
    retrieval_cache_sha256: str,
    retrieval_top_k: int,
    seed: int = PAIR_SEED,
) -> dict[str, Any]:
    """Build a deterministic manifest containing no dataset text or titles."""

    ordered = tuple(sorted(pairs, key=lambda pair: pair.sort_key))
    if len({pair.pair_id for pair in ordered}) != len(ordered):
        raise PairInvariantError("manifest contains duplicate pair IDs")
    for name, digest in (
        ("source", source_sha256),
        ("IDs", ids_sha256),
        ("retrieval cache", retrieval_cache_sha256),
    ):
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise PairInvariantError(f"{name} SHA256 must be lowercase hexadecimal")

    return {
        "schema_version": PAIR_SCHEMA_VERSION,
        "dataset": "HotpotQA",
        "official_split": official_split,
        "seed": seed,
        "retrieval_top_k": retrieval_top_k,
        "schema": {
            "document_count_per_topology": 4,
            "edge_count_per_topology": 4,
            "relation_multiset": {"retrieval": 2, "title_mention": 2},
            "sentence_selection": (
                "per-document union of gold supporting indices and every observed "
                "mention edge used by the shared positive/negative payload: A->B/C->D "
                "for synthetic-common and broad T2, plus A->D/C->B for all-observed "
                "rewire; first sentence fallback; fixed within each endpoint base"
            ),
            "neutral_aliases": (
                "SHA256 ordering of seed, qid, ordered base endpoints, and context "
                "index; fixed across every variant of an endpoint base"
            ),
            "strata": {
                "synthetic_common": (
                    "one shared payload base with T1/T2/T3/fork/double-collider; "
                    "cross A->D/C->B and reverse control D->C are absent"
                ),
                "all_observed_rewire": (
                    "A->B/C->D/A->D/C->B are observed and all four mention-evidence "
                    "sentence sets are shared by positive and negative"
                ),
                "t2_all_observed": (
                    "broad all-observed T2 endpoint operations excluding every "
                    "synthetic-common endpoint; combined T2 is common T2 plus this "
                    "extra stratum"
                ),
            },
            "base_identity": (
                "SHA256 includes seed, qid, ordered endpoints, stratum, and sentence "
                "selection policy, so different shared text payloads cannot collide"
            ),
        },
        "provenance": {
            "source_sha256": source_sha256,
            "ids_sha256": ids_sha256,
            "retrieval_cache_sha256": retrieval_cache_sha256,
        },
        "counts": summarize_pairs(ordered),
        "pairs": [pair.to_manifest_record() for pair in ordered],
    }
