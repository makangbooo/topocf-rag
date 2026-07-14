from __future__ import annotations

from topocf_rag.graph_retrieval import RetrievalQuestion
from topocf_rag.twowiki import QUESTION_TYPES
from topocf_rag.twowiki_graph_retrieval import (
    FROZEN_HOTPOT_TRANSFER_CONFIG,
    build_twowiki_graph_retrieval_report,
    evaluate_twowiki_graph,
)
from topocf_rag.twowiki_graph_validation import (
    build_validation_report,
    evaluate_degree_prior,
    permutation_null_report,
)


def _question(qid: str, question_type: str) -> RetrievalQuestion:
    if question_type == "bridge_comparison":
        gold = (0, 1, 2, 3)
        scores = (0.9, 0.8, 0.7, 0.1, 0.85, 0.5, 0.4, 0.3, 0.2, 0.0)
        edges = ((0, 3), (1, 2))
    else:
        gold = (0, 1)
        scores = (0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0)
        edges = ((0, 1),)
    return RetrievalQuestion(
        qid=qid,
        dense_scores=scores,
        bm25_scores=scores,
        gold_indices=gold,
        mention_edges=edges,
        question_type=question_type,
    )


def _balanced(prefix: str) -> tuple[RetrievalQuestion, ...]:
    return tuple(_question(f"{prefix}-{kind}", kind) for kind in QUESTION_TYPES)


def _base_report(train, dev):
    return build_twowiki_graph_retrieval_report(
        train,
        dev,
        provenance={"train": {"cache_sha256": "a" * 64}},
        hotpot_transfer_config=FROZEN_HOTPOT_TRANSFER_CONFIG,
        hotpot_authorization={"base_report_sha256": "b" * 64},
    )


def test_degree_prior_and_permutation_null_are_text_free_and_deterministic() -> None:
    dev = _balanced("dev")
    actual = evaluate_twowiki_graph(dev, FROZEN_HOTPOT_TRANSFER_CONFIG)
    degree = evaluate_degree_prior(dev, FROZEN_HOTPOT_TRANSFER_CONFIG)
    assert degree["question_count"] == 4

    first = permutation_null_report(
        dev,
        FROZEN_HOTPOT_TRANSFER_CONFIG,
        actual,
        repetitions=5,
        seed=20260714,
    )
    second = permutation_null_report(
        dev,
        FROZEN_HOTPOT_TRANSFER_CONFIG,
        actual,
        repetitions=5,
        seed=20260714,
    )
    assert first == second
    assert first["repetitions"] == 5
    assert "dev-comparison" not in str(first)


def test_validation_reproduces_base_and_declares_post_primary_status() -> None:
    train = _balanced("train")
    dev = _balanced("dev")
    base = _base_report(train, dev)
    report = build_validation_report(
        train,
        dev,
        base_report=base,
        base_report_sha256="c" * 64,
        provenance={"base_report_path": "/private/base.json"},
        permutation_repetitions=5,
        permutation_seed=20260714,
    )
    assert report["protocol"]["validation_declared_after_primary_2wiki_result"]
    assert report["protocol"]["dev_used_for_additional_selection"] is False
    assert report["dev"]["actual_vs_strongest_baseline"][
        "absolute_macro_mean_complete_delta"
    ] > 0.0
    assert report["train"]["selected_degree_only_name"]
    assert report["base_report"]["sha256"] == "c" * 64
