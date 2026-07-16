from __future__ import annotations

from dataclasses import asdict
import json

import pytest

import topocf_rag.musique_graph_validation as validation
from topocf_rag.graph_retrieval import (
    GraphMethodConfig,
    RetrievalQuestion,
)
from topocf_rag.musique_graph_retrieval import (
    evaluate_baseline,
    evaluate_degree_prior,
    evaluate_graph,
    select_best_result,
)
from topocf_rag.musique_graph_validation import (
    EXPECTED_STAGE_E_REPORT_SHA256,
    MuSiQueGraphValidationError,
    build_musique_validation_report,
    paired_stratified_bootstrap,
)
from topocf_rag.musique_splits import all_cells


MINI_GRID = tuple(
    GraphMethodConfig(
        family="one_hop_max",
        seed="dense",
        direction=direction,
        graph_weight=0.75,
    )
    for direction in ("undirected", "outgoing", "incoming")
)


def _cell_parts(cell: str) -> tuple[int, str]:
    hop_text, status = cell.split("__", maxsplit=1)
    return int(hop_text.removesuffix("hop")), status


def _question(qid: str, cell: str) -> RetrievalQuestion:
    hop, _status = _cell_parts(cell)
    dense = tuple(1.0 - index / 25 for index in range(20))
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
        _question(f"private-train-{cell}-{index}", cell)
        for cell in all_cells()
        for index in range(200)
    )


def _dev(prefix: str = "private-dev") -> tuple[RetrievalQuestion, ...]:
    return tuple(_question(f"{prefix}-{cell}", cell) for cell in all_cells())


def _base_report(
    train: tuple[RetrievalQuestion, ...],
    dev: tuple[RetrievalQuestion, ...],
) -> dict:
    train_baselines = {
        name: evaluate_baseline(train, name)
        for name in ("dense", "bm25", "dense_bm25_rrf")
    }
    baseline_name, baseline_train = select_best_result(
        list(train_baselines.items())
    )
    graph_results = [
        (config.name, evaluate_graph(train, config)) for config in MINI_GRID
    ]
    graph_name, graph_train = select_best_result(graph_results)
    graph_config = {config.name: config for config in MINI_GRID}[graph_name]
    degree_results = [
        (config.name, evaluate_degree_prior(train, config))
        for config in MINI_GRID
    ]
    degree_name, degree_train = select_best_result(degree_results)
    degree_config = {config.name: config for config in MINI_GRID}[degree_name]
    return {
        "task": "musique_occurrence_graph_shortcut_controlled_retrieval",
        "selection": {
            "selected_baseline": baseline_name,
            "selected_graph_method": graph_name,
            "selected_graph_config": asdict(graph_config),
            "selected_degree_only_method": degree_name,
            "selected_degree_only_config": asdict(degree_config),
        },
        "train": {
            "baselines": train_baselines,
            "selected_baseline_metrics": baseline_train,
            "selected_graph_metrics": graph_train,
            "selected_degree_only_metrics": degree_train,
        },
        "dev": {
            "baselines": {
                name: evaluate_baseline(dev, name)
                for name in ("dense", "bm25", "dense_bm25_rrf")
            },
            "selected_graph": evaluate_graph(dev, graph_config),
            "selected_degree_only": evaluate_degree_prior(dev, degree_config),
        },
        "gate": {"passed": True},
    }


def test_paired_bootstrap_is_stratified_deterministic_and_content_free() -> None:
    questions = _dev("bootstrap-private")
    candidate_rankings = tuple(tuple(range(20)) for _ in questions)
    control_rankings = tuple(tuple(reversed(range(20))) for _ in questions)
    candidate = validation._ranking_outcomes(questions, candidate_rankings)
    control = validation._ranking_outcomes(questions, control_rankings)
    first = paired_stratified_bootstrap(
        candidate, control, repetitions=5, seed=20260716
    )
    second = paired_stratified_bootstrap(
        candidate, control, repetitions=5, seed=20260716
    )
    assert first == second
    assert first["overall"]["mean_across_extra_budgets"]["actual"] == 1.0
    assert first["by_cell"]["4hop__unique_titles"][
        "mean_across_extra_budgets"
    ]["percentile_95_low"] == 1.0
    assert "bootstrap-private" not in json.dumps(first, sort_keys=True)


def test_validation_reproduces_stage_e_without_dev_selection_or_private_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation, "graph_candidate_grid", lambda: MINI_GRID)
    train = _balanced_train()
    dev = _dev()
    base = _base_report(train, dev)
    report = build_musique_validation_report(
        train,
        dev,
        base_report=base,
        base_report_sha256=EXPECTED_STAGE_E_REPORT_SHA256,
        provenance={"base_report_path": "reports/musique/private.json"},
        bootstrap_repetitions=3,
        bootstrap_seed=20260716,
        config_bootstrap_repetitions=3,
        config_bootstrap_seed=20260717,
    )
    assert all(report["stage_e_reproduction"]["checks"].values())
    assert report["protocol"]["dev_used_for_additional_selection"] is False
    assert report["train"]["leave_one_cell_out"]["protocol"]["dev_used"] is False
    assert report["claim_boundary"]["directed_topology_reasoning_authorized"] is False
    encoded = json.dumps(report, sort_keys=True)
    assert "private-train" not in encoded
    assert "private-dev" not in encoded


def test_validation_rejects_changed_stage_e_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation, "graph_candidate_grid", lambda: MINI_GRID)
    train = _balanced_train()
    dev = _dev("hash-private")
    with pytest.raises(MuSiQueGraphValidationError, match="SHA256 changed"):
        build_musique_validation_report(
            train,
            dev,
            base_report=_base_report(train, dev),
            base_report_sha256="0" * 64,
            provenance={},
            bootstrap_repetitions=1,
            config_bootstrap_repetitions=1,
        )
