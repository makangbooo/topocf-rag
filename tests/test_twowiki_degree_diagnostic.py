from __future__ import annotations

from copy import deepcopy

from topocf_rag.graph_retrieval import RetrievalQuestion
from topocf_rag.twowiki import QUESTION_TYPES
from topocf_rag.twowiki_degree_diagnostic import (
    build_degree_diagnostic_report,
    centrality_shortcut_report,
    complementarity_report,
)
from topocf_rag.twowiki_graph_retrieval import (
    FROZEN_HOTPOT_TRANSFER_CONFIG,
    build_twowiki_graph_retrieval_report,
)
from topocf_rag.twowiki_graph_validation import build_validation_report


def _question(qid: str, question_type: str) -> RetrievalQuestion:
    if question_type == "bridge_comparison":
        gold = (0, 1, 2, 3)
        scores = (0.9, 0.8, 0.7, 0.1, 0.85, 0.5, 0.4, 0.3, 0.2, 0.0)
        edges = ((0, 3), (1, 2), (4, 0))
    else:
        gold = (0, 1)
        scores = (0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0)
        edges = ((0, 1), (2, 0))
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


def test_centrality_and_complementarity_are_aggregate_and_deterministic() -> None:
    dev = _balanced("private-dev")
    left = tuple(tuple(range(10)) for _ in dev)
    right = tuple(tuple(reversed(range(10))) for _ in dev)

    complementarity = complementarity_report(
        dev,
        left,
        right,
        left_name="left",
        right_name="right",
    )
    centrality = centrality_shortcut_report(dev)

    assert complementarity["macro_mean_oracle_headroom"] >= 0.0
    assert set(centrality) == {"outgoing", "incoming", "undirected"}
    for direction in centrality.values():
        auc = direction["overall"]["gold_vs_non_gold_degree_pairwise_auc"]
        assert 0.0 <= auc <= 1.0
    assert "private-dev" not in str(complementarity)
    assert "private-dev" not in str(centrality)


def test_degree_diagnostic_reproduces_inputs_and_records_post_result_status() -> None:
    train = _balanced("private-train")
    dev = _balanced("private-dev")
    base = _base_report(train, dev)
    validation = build_validation_report(
        train,
        dev,
        base_report=base,
        base_report_sha256="c" * 64,
        provenance={"base_report_path": "/private/base.json"},
        permutation_repetitions=2,
        permutation_seed=20260714,
    )
    validation = deepcopy(validation)
    for check in validation["gate"]["checks"].values():
        check["passed"] = True
    validation["gate"]["checks"]["degree_only_control"]["passed"] = False
    validation["gate"]["passed"] = False
    hotpot = {
        "gate": {"passed": True},
        "dev": {
            "degree_only_control": {
                "actual_graph_minus_degree_mean": 0.08133333333333337
            }
        },
    }

    report = build_degree_diagnostic_report(
        train,
        dev,
        base_report=base,
        base_report_sha256="c" * 64,
        validation_report=validation,
        validation_report_sha256="d" * 64,
        hotpot_validation_report=hotpot,
        hotpot_validation_report_sha256="e" * 64,
        provenance={"validation_report_path": "/private/validation.json"},
    )

    assert report["protocol"]["declared_after_2wiki_degree_control_failure"]
    assert report["protocol"]["dev_used_for_method_or_degree_selection"] is False
    assert report["decision"]["status"] in {
        "repair_cross_dataset_transfer_then_validate_on_musique",
        "develop_label_free_selective_router_on_train_only",
        "pivot_to_degree_shortcut_benchmark_audit",
    }
    assert report["artifacts"]["validation_report_sha256"] == "d" * 64
    assert "private-train" not in str(report)
    assert "private-dev" not in str(report)
