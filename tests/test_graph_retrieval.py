import json

import pytest

from topocf_rag.graph_retrieval import (
    GraphMethodConfig,
    GraphRetrievalInvariantError,
    RetrievalQuestion,
    build_kill_test_report,
    evaluate_baseline,
    evaluate_graph_method,
    graph_candidate_grid,
    graph_diagnostics,
    graph_method_scores,
    personalized_pagerank_scores,
    prepare_retrieval_question,
    reciprocal_rank_scores,
    rrf_fusion_scores,
    stable_rank,
)


def _question(
    qid: str,
    *,
    dense: tuple[float, ...] = (0.9, 0.2, 0.8, 0.1),
    bm25: tuple[float, ...] = (0.9, 0.2, 0.8, 0.1),
    gold: tuple[int, int] = (0, 1),
    edges: tuple[tuple[int, int], ...] = ((0, 1),),
) -> RetrievalQuestion:
    return RetrievalQuestion(
        qid=qid,
        dense_scores=dense,
        bm25_scores=bm25,
        gold_indices=gold,
        mention_edges=edges,
    )


def _example() -> dict[str, object]:
    return {
        "_id": "q-example",
        "type": "bridge",
        "level": "hard",
        "question": "Which sentinel follows the bridge?",
        "answer": "Sentinel",
        "context": [
            ["Gold Alpha", ["Gold Alpha points to Gold Beta."]],
            ["Gold Beta", ["The answer is Sentinel."]],
            ["Distractor X", ["Distractor text."]],
            ["Distractor Y", ["Other distractor text."]],
        ],
        "supporting_facts": [["Gold Alpha", 0], ["Gold Beta", 0]],
    }


def test_stable_rank_and_rrf_are_scale_free_and_tie_stable() -> None:
    assert stable_rank([0.8, 0.8, 0.2]) == (0, 1, 2)
    assert stable_rank([8.0, 8.0, 2.0]) == (0, 1, 2)
    reciprocal = reciprocal_rank_scores([0.8, 0.8, 0.2])
    assert sum(reciprocal) == pytest.approx(1.0)
    assert reciprocal[0] > reciprocal[1] > reciprocal[2]

    fused = rrf_fusion_scores([0.9, 0.1, 0.2], [0.1, 0.9, 0.2])
    assert sum(fused) == pytest.approx(1.0)
    assert fused[0] == pytest.approx(fused[1])


def test_outgoing_one_hop_graph_recovers_missing_gold_document() -> None:
    question = _question("q-recover")
    config = GraphMethodConfig(
        family="one_hop_max",
        seed="dense",
        direction="outgoing",
        graph_weight=0.75,
    )
    assert set(stable_rank(question.dense_scores)[:2]) == {0, 2}
    assert set(stable_rank(graph_method_scores(question, config))[:2]) == {0, 1}

    incoming = GraphMethodConfig(
        family="one_hop_max",
        seed="dense",
        direction="incoming",
        graph_weight=0.75,
    )
    assert set(stable_rank(graph_method_scores(question, incoming))[:2]) != {0, 1}


def test_personalized_pagerank_is_normalized_and_direction_sensitive() -> None:
    seed = reciprocal_rank_scores([0.9, 0.2, 0.8, 0.1])
    outgoing = personalized_pagerank_scores(
        seed,
        [(0, 1)],
        direction="outgoing",
        restart_probability=0.5,
    )
    incoming = personalized_pagerank_scores(
        seed,
        [(0, 1)],
        direction="incoming",
        restart_probability=0.5,
    )
    assert sum(outgoing) == pytest.approx(1.0)
    assert sum(incoming) == pytest.approx(1.0)
    assert outgoing[1] > incoming[1]


def test_prepare_question_builds_graph_without_using_gold_for_scores() -> None:
    example_a = _example()
    prepared_a = prepare_retrieval_question(example_a, [0.9, 0.2, 0.8, 0.1])
    assert prepared_a.gold_indices == (0, 1)
    assert (0, 1) in prepared_a.mention_edges

    example_b = _example()
    example_b["supporting_facts"] = [["Distractor X", 0], ["Distractor Y", 0]]
    prepared_b = prepare_retrieval_question(example_b, [0.9, 0.2, 0.8, 0.1])
    config = GraphMethodConfig(
        family="personalized_pagerank",
        seed="dense_bm25_rrf",
        direction="undirected",
        restart_probability=0.5,
    )
    assert prepared_b.gold_indices == (2, 3)
    assert prepared_a.mention_edges == prepared_b.mention_edges
    assert prepared_a.bm25_scores == prepared_b.bm25_scores
    assert graph_method_scores(prepared_a, config) == graph_method_scores(
        prepared_b, config
    )


def test_evaluation_reports_complete_evidence_and_full_evidence_mrr() -> None:
    questions = [_question("q-1"), _question("q-2")]
    dense = evaluate_baseline(questions, "dense", top_ks=(2, 3))
    assert dense["by_top_k"]["2"]["complete_gold_evidence_rate"] == 0.0
    assert dense["by_top_k"]["3"]["complete_gold_evidence_rate"] == 1.0
    assert dense["full_evidence_mrr"] == pytest.approx(1 / 3)

    config = GraphMethodConfig(
        family="one_hop_max",
        seed="dense",
        direction="outgoing",
        graph_weight=0.75,
    )
    graph = evaluate_graph_method(questions, config, top_ks=(2, 3))
    assert graph["by_top_k"]["2"]["complete_gold_evidence_rate"] == 1.0
    assert graph["full_evidence_mrr"] == pytest.approx(0.5)


def test_graph_diagnostics_quantify_recoverable_dense_failures() -> None:
    diagnostics = graph_diagnostics([_question("q-diagnostic")], top_ks=(2, 3))
    assert diagnostics["gold_pair_any_direction_rate"] == 1.0
    assert diagnostics["dense_recovery_opportunity_by_top_k"]["2"] == {
        "dense_partial_failure_count": 1,
        "gold_edge_recoverable_count": 1,
        "recoverable_rate_among_dense_partial_failures": 1.0,
    }


def test_report_selects_on_train_only_and_contains_no_private_content() -> None:
    train = [_question(f"train-{index}") for index in range(4)]
    dev_a = [_question(f"dev-a-{index}") for index in range(3)]
    dev_b = [
        _question(
            f"dev-b-{index}",
            dense=(0.9, 0.8, 0.2, 0.1),
            bm25=(0.9, 0.8, 0.2, 0.1),
            edges=(),
        )
        for index in range(3)
    ]
    provenance = {"train": {"source_sha256": "a" * 64}}
    report_a = build_kill_test_report(train, dev_a, provenance=provenance)
    report_b = build_kill_test_report(train, dev_b, provenance=provenance)

    assert report_a["selection"] == report_b["selection"]
    assert report_a["gate"]["passed"] is True
    encoded = json.dumps(report_a, sort_keys=True)
    for forbidden in ("train-0", "dev-a-0", "Which sentinel", "Gold Alpha"):
        assert forbidden not in encoded


def test_candidate_grid_is_frozen_unique_and_validated() -> None:
    candidates = graph_candidate_grid()
    assert len(candidates) == 36
    assert len({candidate.name for candidate in candidates}) == len(candidates)
    with pytest.raises(GraphRetrievalInvariantError, match="restart"):
        GraphMethodConfig(
            family="personalized_pagerank",
            seed="dense",
            direction="outgoing",
            restart_probability=1.0,
        )
