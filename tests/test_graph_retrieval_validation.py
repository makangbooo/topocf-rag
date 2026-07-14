import json

import pytest

from topocf_rag.graph_retrieval import (
    GraphMethodConfig,
    RetrievalQuestion,
    build_kill_test_report,
    degree_prior_scores,
    evaluate_graph_method,
    graph_rankings,
)
from topocf_rag.graph_retrieval_validation import (
    build_validation_report,
    complete_evidence_flags,
    deterministic_node_permutation,
    exact_mcnemar_p_value,
    paired_transition_report,
    permutation_null_report,
    permute_question_graph,
)


def _question(qid: str) -> RetrievalQuestion:
    return RetrievalQuestion(
        qid=qid,
        dense_scores=(0.9, 0.2, 0.8, 0.1),
        bm25_scores=(0.9, 0.2, 0.8, 0.1),
        gold_indices=(0, 1),
        mention_edges=((0, 1), (0, 2), (3, 2)),
    )


def _config() -> GraphMethodConfig:
    return GraphMethodConfig(
        family="one_hop_max",
        seed="dense",
        direction="outgoing",
        graph_weight=0.75,
    )


def _degrees(edges: tuple[tuple[int, int], ...], size: int):
    indegree = [0] * size
    outdegree = [0] * size
    for source, target in edges:
        outdegree[source] += 1
        indegree[target] += 1
    return sorted(indegree), sorted(outdegree)


def test_node_permutation_is_deterministic_nonidentity_and_isomorphic() -> None:
    question = _question("q-permute")
    permutation = deterministic_node_permutation(
        question.qid, question.document_count, replicate=3
    )
    assert permutation == deterministic_node_permutation(
        question.qid, question.document_count, replicate=3
    )
    assert permutation != tuple(range(question.document_count))
    assert set(permutation) == set(range(question.document_count))

    permuted = permute_question_graph(question, replicate=3)
    expected_edges = tuple(
        sorted(
            (permutation[source], permutation[target])
            for source, target in question.mention_edges
        )
    )
    assert permuted.mention_edges == expected_edges
    assert len(permuted.mention_edges) == len(question.mention_edges)
    assert _degrees(permuted.mention_edges, question.document_count) == _degrees(
        question.mention_edges, question.document_count
    )
    assert permuted.dense_scores == question.dense_scores
    assert permuted.gold_indices == question.gold_indices


def test_exact_mcnemar_handles_no_discordance_and_one_sided_extreme() -> None:
    assert exact_mcnemar_p_value(0, 0) == 1.0
    assert exact_mcnemar_p_value(5, 0) == pytest.approx(0.0625)
    assert exact_mcnemar_p_value(10, 0) == pytest.approx(0.001953125)
    assert exact_mcnemar_p_value(0, 10) == pytest.approx(0.001953125)


def test_paired_transition_report_counts_wins_and_losses() -> None:
    questions = [_question("q-1"), _question("q-2")]
    baseline = ((0, 2, 1, 3), (0, 1, 2, 3))
    graph = ((0, 1, 2, 3), (0, 2, 1, 3))
    report = paired_transition_report(
        questions, baseline, graph, top_ks=(2,)
    )
    row = report["by_top_k"]["2"]
    assert row["graph_only_complete_count"] == 1
    assert row["baseline_only_complete_count"] == 1
    assert row["net_complete_count_gain"] == 0
    assert row["exact_mcnemar_two_sided_p"] == 1.0


def test_complete_evidence_flags_use_two_gold_documents() -> None:
    questions = [_question("q-flags")]
    assert complete_evidence_flags(
        questions, [(0, 2, 1, 3)], top_k=2
    ) == (False,)
    assert complete_evidence_flags(
        questions, [(0, 1, 2, 3)], top_k=2
    ) == (True,)


def test_degree_prior_control_is_finite_and_deterministic() -> None:
    question = _question("q-degree")
    scores = degree_prior_scores(question, _config())
    assert scores == degree_prior_scores(question, _config())
    assert len(scores) == question.document_count
    assert all(value >= 0.0 for value in scores)


def test_permutation_null_report_is_aggregate_and_content_safe() -> None:
    questions = [_question(f"private-q-{index}") for index in range(4)]
    config = _config()
    actual = evaluate_graph_method(questions, config, top_ks=(2, 3))
    report = permutation_null_report(
        questions,
        config,
        actual,
        top_ks=(2, 3),
        repetitions=5,
        seed=17,
    )
    assert report["repetitions"] == 5
    assert report["mean_complete_gold_evidence_rate"][
        "empirical_one_sided_p"
    ] >= 1 / 6
    encoded = json.dumps(report, sort_keys=True)
    assert "private-q" not in encoded

    rankings = graph_rankings(questions, config)
    assert len(rankings) == len(questions)


def test_full_validation_report_is_reproducible_and_content_safe() -> None:
    train = [_question(f"private-train-{index}") for index in range(4)]
    dev = [_question(f"private-dev-{index}") for index in range(3)]
    base = build_kill_test_report(train, dev, provenance={"synthetic": True})
    report = build_validation_report(
        train,
        dev,
        base_report=base,
        base_report_sha256="a" * 64,
        provenance={"synthetic": True},
        permutation_repetitions=3,
        permutation_seed=17,
    )
    assert report["protocol"]["degree_control_selection"].endswith(
        "train only"
    )
    assert report["dev"]["degree_only_control"]["selected_config"]
    encoded = json.dumps(report, sort_keys=True)
    assert "private-train" not in encoded
    assert "private-dev" not in encoded
