from __future__ import annotations

from dataclasses import asdict
import json

import pytest

from topocf_rag.graph_retrieval import RetrievalQuestion
from topocf_rag.twowiki import QUESTION_TYPES
from topocf_rag.twowiki_graph_retrieval import (
    FROZEN_HOTPOT_TRANSFER_CONFIG,
    build_twowiki_graph_retrieval_report,
    evaluate_twowiki_baseline,
    evaluate_twowiki_graph,
    evaluate_twowiki_rankings,
    graph_diagnostics,
    paired_transition_report,
    prepare_twowiki_retrieval_question,
)


def _question(qid: str, question_type: str) -> RetrievalQuestion:
    if question_type == "bridge_comparison":
        gold = (0, 1, 2, 3)
        dense = (0.9, 0.8, 0.7, 0.1, 0.85, 0.5, 0.4, 0.3, 0.2, 0.0)
        edges = ((0, 3), (1, 2))
    else:
        gold = (0, 1)
        dense = (0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0)
        edges = ((0, 1),)
    return RetrievalQuestion(
        qid=qid,
        dense_scores=dense,
        bm25_scores=dense,
        gold_indices=gold,
        mention_edges=edges,
        question_type=question_type,
    )


def _balanced(prefix: str) -> tuple[RetrievalQuestion, ...]:
    return tuple(_question(f"{prefix}-{kind}", kind) for kind in QUESTION_TYPES)


def _example(question_type: str) -> dict[str, object]:
    gold_count = 4 if question_type == "bridge_comparison" else 2
    return {
        "_id": "synthetic-example",
        "question": "Which documents form the chain?",
        "answer": "Synthetic",
        "type": question_type,
        "context": [
            [
                f"Title {index}",
                [
                    (
                        f"Title {index} mentions Title {index + 1}."
                        if index < 9
                        else "Terminal document."
                    )
                ],
            ]
            for index in range(10)
        ],
        "supporting_facts": [
            [f"Title {index}", 0] for index in range(gold_count)
        ],
    }


def test_prepare_question_supports_two_and_four_gold_documents() -> None:
    two = prepare_twowiki_retrieval_question(
        _example("compositional"), [1.0 - index / 20 for index in range(10)]
    )
    four = prepare_twowiki_retrieval_question(
        _example("bridge_comparison"),
        [1.0 - index / 20 for index in range(10)],
    )
    assert two.gold_indices == (0, 1)
    assert four.gold_indices == (0, 1, 2, 3)
    assert two.question_type == "compositional"
    assert (0, 1) in two.mention_edges


def test_variable_budget_metrics_use_type_specific_top_k_and_macro_average() -> None:
    questions = _balanced("metrics")
    rankings = tuple(
        tuple(
            sorted(
                range(10),
                key=lambda index: (-question.dense_scores[index], index),
            )
        )
        for question in questions
    )
    metrics = evaluate_twowiki_rankings(questions, rankings)
    assert metrics["question_count_by_type"] == {kind: 1 for kind in QUESTION_TYPES}
    assert metrics["by_question_type"]["bridge_comparison"][
        "top_k_by_extra_budget"
    ] == {"0": 4, "1": 5, "3": 7}
    assert metrics["by_question_type"]["compositional"][
        "top_k_by_extra_budget"
    ] == {"0": 2, "1": 3, "3": 5}
    assert metrics["macro"]["by_extra_budget"]["0"][
        "macro_complete_gold_evidence_rate"
    ] == 0.0
    assert all(
        0.0 <= row["complete_gold_evidence_rate"] <= 1.0
        for row in metrics["micro"]["by_extra_budget"].values()
    )


def test_graph_improves_exact_budget_and_paired_report_is_consistent() -> None:
    questions = _balanced("recover")
    baseline = evaluate_twowiki_baseline(questions, "dense_bm25_rrf")
    graph = evaluate_twowiki_graph(questions, FROZEN_HOTPOT_TRANSFER_CONFIG)
    assert graph["macro"]["mean_complete_gold_evidence_rate"] > baseline[
        "macro"
    ]["mean_complete_gold_evidence_rate"]

    from topocf_rag.graph_retrieval import baseline_rankings, graph_rankings

    paired = paired_transition_report(
        questions,
        baseline_rankings(questions, "dense_bm25_rrf"),
        graph_rankings(questions, FROZEN_HOTPOT_TRANSFER_CONFIG),
    )
    assert paired["overall"]["1"]["graph_only_complete_count"] == 4
    assert paired["overall"]["1"]["baseline_only_complete_count"] == 0


def test_graph_diagnostics_handle_four_gold_connectivity() -> None:
    diagnostics = graph_diagnostics(_balanced("diagnostic"))
    bridge = diagnostics["by_question_type"]["bridge_comparison"]
    assert bridge["any_gold_to_gold_mention_edge_rate"] == 1.0
    assert bridge["direction_ignored_gold_subgraph_connected_rate"] == 0.0


def test_report_selects_only_on_train_and_serializes_no_private_content() -> None:
    train = _balanced("train-private")
    dev_a = _balanced("dev-a-private")
    dev_b = tuple(
        RetrievalQuestion(
            qid=f"dev-b-private-{kind}",
            dense_scores=tuple(reversed(_question("x", kind).dense_scores)),
            bm25_scores=tuple(reversed(_question("x", kind).bm25_scores)),
            gold_indices=_question("x", kind).gold_indices,
            mention_edges=(),
            question_type=kind,
        )
        for kind in QUESTION_TYPES
    )
    kwargs = {
        "provenance": {"train": {"cache_sha256": "a" * 64}},
        "hotpot_transfer_config": FROZEN_HOTPOT_TRANSFER_CONFIG,
        "hotpot_authorization": {"base_report_sha256": "b" * 64},
    }
    report_a = build_twowiki_graph_retrieval_report(train, dev_a, **kwargs)
    report_b = build_twowiki_graph_retrieval_report(train, dev_b, **kwargs)
    assert report_a["selection"] == report_b["selection"]
    encoded = json.dumps(report_a, sort_keys=True)
    for forbidden in (
        "train-private",
        "dev-a-private",
        "Which documents",
        "Title 0",
    ):
        assert forbidden not in encoded
    assert report_a["protocol"]["dev_used_for_selection"] is False
    assert report_a["selection"]["hotpot_transfer_config"] == asdict(
        FROZEN_HOTPOT_TRANSFER_CONFIG
    )
