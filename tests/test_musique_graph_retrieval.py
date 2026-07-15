from __future__ import annotations

import json

import pytest

import topocf_rag.musique_graph_retrieval as musique_graph
from topocf_rag.graph_retrieval import GraphMethodConfig, RetrievalQuestion
from topocf_rag.musique_graph_retrieval import (
    build_musique_graph_report,
    build_occurrence_mention_edges,
    evaluate_musique_rankings,
    permutation_null_report,
    prepare_musique_retrieval_question,
)
from topocf_rag.musique_splits import all_cells


CONFIG = GraphMethodConfig(
    family="one_hop_max",
    seed="dense",
    direction="undirected",
    graph_weight=0.75,
)


def _record(qid: str, hop: int, status: str) -> dict:
    paragraphs = [
        {
            "idx": index,
            "title": f"Title {qid} {index}",
            "paragraph_text": f"Body {qid} {index}.",
            "is_supporting": index < hop,
        }
        for index in range(20)
    ]
    if status == "distractor_only_title_collision":
        paragraphs[hop + 1]["title"] = paragraphs[hop]["title"]
    elif status == "mixed_text_support_title_collision":
        paragraphs[hop]["title"] = paragraphs[0]["title"]
    return {
        "id": qid,
        "question": "Which evidence completes the chain?",
        "answerable": True,
        "answer": "Synthetic answer",
        "answer_aliases": [],
        "question_decomposition": [
            {
                "id": index,
                "question": f"Step {index}?",
                "answer": f"Answer {index}",
                "paragraph_support_idx": index,
            }
            for index in range(hop)
        ],
        "paragraphs": paragraphs,
    }


def _cell_parts(cell: str) -> tuple[int, str]:
    hop_text, status = cell.split("__", maxsplit=1)
    return int(hop_text.removesuffix("hop")), status


def _question(qid: str, cell: str) -> RetrievalQuestion:
    hop, _status = _cell_parts(cell)
    dense = tuple(
        1.0 - index / 25 if index == 0 else 0.2 - index / 100
        for index in range(20)
    )
    return RetrievalQuestion(
        qid=qid,
        dense_scores=dense,
        bm25_scores=dense,
        gold_indices=tuple(range(hop)),
        mention_edges=tuple((index, index + 1) for index in range(hop - 1)),
        question_type=cell,
    )


def _balanced_train() -> tuple[RetrievalQuestion, ...]:
    return tuple(
        _question(f"train-private-{cell}-{index}", cell)
        for cell in all_cells()
        for index in range(200)
    )


def _dev(prefix: str = "dev-private") -> tuple[RetrievalQuestion, ...]:
    return tuple(_question(f"{prefix}-{cell}", cell) for cell in all_cells())


def test_occurrence_graph_fans_one_mention_to_duplicate_title_nodes() -> None:
    record = _record("fanout", 2, "mixed_text_support_title_collision")
    duplicate_title = record["paragraphs"][0]["title"]
    record["paragraphs"][19]["paragraph_text"] = (
        f"This body mentions {duplicate_title}."
    )
    edges = build_occurrence_mention_edges(record["paragraphs"])
    assert (19, 0) in edges
    assert (19, 2) in edges
    assert (19, 19) not in edges


def test_prepare_question_preserves_idx_gold_and_collision_cell() -> None:
    record = _record("prepare", 3, "mixed_text_support_title_collision")
    cache = {
        "paragraph_indices": list(range(20)),
        "scores": [1.0 - index / 25 for index in range(20)],
        "document_token_lengths": [20] * 20,
    }
    question = prepare_musique_retrieval_question(record, cache)
    assert question.gold_indices == (0, 1, 2)
    assert question.question_type == (
        "3hop__mixed_text_support_title_collision"
    )
    assert question.document_count == 20


def test_metrics_use_variable_budget_and_nine_cell_macro() -> None:
    questions = _dev("metrics")
    rankings = tuple(tuple(range(20)) for _ in questions)
    metrics = evaluate_musique_rankings(questions, rankings)
    assert metrics["question_count"] == 9
    assert metrics["macro"]["cell_count"] == 9
    assert metrics["by_cell"]["4hop__unique_titles"]["by_extra_budget"][
        "0"
    ]["top_k"] == 4
    assert metrics["by_cell"]["2hop__unique_titles"]["by_extra_budget"][
        "3"
    ]["top_k"] == 5
    assert metrics["macro"]["mean_complete_gold_evidence_rate"] == 1.0


def test_permutation_null_is_deterministic_and_content_free() -> None:
    dev = _dev("null-private")
    actual = musique_graph.evaluate_graph(dev, CONFIG)
    first = permutation_null_report(
        dev, CONFIG, actual, repetitions=3, seed=20260715
    )
    second = permutation_null_report(
        dev, CONFIG, actual, repetitions=3, seed=20260715
    )
    assert first == second
    assert first["repetitions"] == 3
    assert "null-private" not in json.dumps(first, sort_keys=True)


def test_report_selection_is_train_only_and_contains_no_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(musique_graph, "graph_candidate_grid", lambda: (CONFIG,))
    train = _balanced_train()
    dev_a = _dev("dev-a-private")
    dev_b = tuple(
        RetrievalQuestion(
            qid=f"dev-b-private-{question.question_type}",
            dense_scores=tuple(reversed(question.dense_scores)),
            bm25_scores=tuple(reversed(question.bm25_scores)),
            gold_indices=question.gold_indices,
            mention_edges=(),
            question_type=question.question_type,
        )
        for question in dev_a
    )
    kwargs = {
        "provenance": {"stage_d_report_sha256": "a" * 64},
        "hotpot_transfer_config": CONFIG,
        "hotpot_authorization": {"base_report_sha256": "b" * 64},
        "permutation_repetitions": 2,
        "permutation_seed": 20260715,
    }
    report_a = build_musique_graph_report(train, dev_a, **kwargs)
    report_b = build_musique_graph_report(train, dev_b, **kwargs)
    assert report_a["selection"] == report_b["selection"]
    assert report_a["protocol"]["dev_used_for_selection"] is False
    encoded = json.dumps(report_a, sort_keys=True)
    for forbidden in (
        "train-private",
        "dev-a-private",
        "Which evidence",
        "Title prepare",
    ):
        assert forbidden not in encoded
