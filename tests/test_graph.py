from __future__ import annotations

from copy import deepcopy

import pytest

from topocf_rag.graph import (
    HotpotInvariantError,
    build_query_document_graph,
    gold_coverage,
    validate_hotpot_example,
)


def example() -> dict[str, object]:
    return {
        "_id": "synthetic-id",
        "question": "Which item is connected?",
        "answer": "Synthetic answer",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": [["Alpha", 0], ["Beta", 0]],
        "context": [
            ["Alpha", ["Alpha explicitly refers to Beta."]],
            ["Beta", ["Beta contains the second fact."]],
            ["Gamma", ["Gamma also refers to Beta."]],
            ["Alphabet", ["This document is unrelated."]],
        ],
    }


def test_build_graph_preserves_edge_direction_and_sentence_evidence() -> None:
    graph = build_query_document_graph(example(), [0.9, 0.8, 0.7, 0.6])
    edge_pairs = {
        (edge.source_index, edge.target_index): edge.sentence_indices
        for edge in graph.mention_edges
    }
    assert edge_pairs[(0, 1)] == (0,)
    assert edge_pairs[(2, 1)] == (0,)
    assert (1, 0) not in edge_pairs


def test_candidate_and_natural_negative_paths_require_real_edges() -> None:
    graph = build_query_document_graph(example(), [0.9, 0.8, 0.7, 0.6])
    assert {(path.source_index, path.target_index) for path in graph.gold_paths} == {
        (0, 1)
    }
    assert {
        (path.source_index, path.target_index)
        for path in graph.natural_negative_paths
    } == {(2, 1)}


def test_retrieval_top_k_limits_candidate_source_without_inventing_edges() -> None:
    graph = build_query_document_graph(
        example(), [0.1, 0.2, 0.9, 0.8], retrieval_top_k=1
    )
    assert [edge.document_index for edge in graph.retrieval_edges] == [2]
    assert {(path.source_index, path.target_index) for path in graph.candidate_paths} == {
        (2, 1)
    }
    assert graph.gold_paths == ()


def test_gold_coverage_reports_ordered_and_undirected_separately() -> None:
    forward = gold_coverage(
        build_query_document_graph(example(), [0.9, 0.8, 0.7, 0.6])
    )
    assert forward.eligible
    assert forward.ordered_directed
    assert not forward.reverse_directed
    assert forward.any_direction
    assert forward.undirected
    assert not forward.reverse_only
    assert not forward.bidirectional

    reversed_example = example()
    reversed_example["context"] = [
        ["Alpha", ["Alpha contains the first fact."]],
        ["Beta", ["Beta explicitly refers to Alpha."]],
        ["Gamma", ["No cross-document mention."]],
    ]
    reverse = gold_coverage(
        build_query_document_graph(reversed_example, [0.9, 0.8, 0.7])
    )
    assert not reverse.ordered_directed
    assert reverse.reverse_directed
    assert reverse.any_direction
    assert reverse.undirected
    assert reverse.reverse_only


def test_undirected_coverage_can_use_edge_from_nonretrieved_endpoint() -> None:
    reversed_example = example()
    reversed_example["context"] = [
        ["Alpha", ["Alpha contains the first fact."]],
        ["Beta", ["Beta explicitly refers to Alpha."]],
        ["Gamma", ["No cross-document mention."]],
    ]
    graph = build_query_document_graph(
        reversed_example, [0.9, 0.1, 0.8], retrieval_top_k=1
    )
    coverage = gold_coverage(graph)
    assert not coverage.any_direction
    assert coverage.undirected


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda item: item["context"].append(["ALPHA", ["Duplicate title."]]),
            "normalized titles must be unique",
        ),
        (
            lambda item: item["supporting_facts"].append(["Missing", 0]),
            "missing context title",
        ),
        (
            lambda item: item["supporting_facts"].append(["Alpha", 99]),
            "out of range",
        ),
    ],
)
def test_data_invariants_reject_invalid_records(mutation, message: str) -> None:
    item = deepcopy(example())
    mutation(item)
    with pytest.raises(HotpotInvariantError, match=message):
        validate_hotpot_example(item)


def test_retrieval_scores_must_align_and_be_finite() -> None:
    with pytest.raises(HotpotInvariantError, match="score count"):
        build_query_document_graph(example(), [0.1])
    with pytest.raises(HotpotInvariantError, match="finite"):
        build_query_document_graph(example(), [0.1, 0.2, float("nan"), 0.4])
