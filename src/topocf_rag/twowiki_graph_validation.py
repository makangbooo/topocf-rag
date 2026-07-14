"""Post-primary robustness controls for the 2Wiki graph retrieval result."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import math
import statistics
from typing import Any

from .graph_retrieval import (
    GraphMethodConfig,
    RetrievalQuestion,
    baseline_rankings,
    degree_prior_rankings,
    graph_candidate_grid,
    graph_rankings,
)
from .graph_retrieval_validation import permute_question_graph
from .twowiki_graph_retrieval import (
    EXTRA_BUDGETS,
    FROZEN_HOTPOT_TRANSFER_CONFIG,
    evaluate_twowiki_rankings,
    paired_transition_report,
    select_best_twowiki_result,
)


VALIDATION_SCHEMA_VERSION = 1
PERMUTATION_REPETITIONS = 200
PERMUTATION_SEED = 20260714
STRONG_BASELINE_MEAN_DELTA_THRESHOLD = 0.02
DEGREE_CONTROL_MEAN_DELTA_THRESHOLD = 0.02
MAX_PER_BUDGET_REGRESSION = 0.01
PERMUTATION_P_THRESHOLD = 0.01
MCNEMAR_P_THRESHOLD = 0.01


class TwoWikiGraphValidationError(ValueError):
    """Raised when a 2Wiki validation invariant is violated."""


def evaluate_degree_prior(
    questions: Sequence[RetrievalQuestion], config: GraphMethodConfig
) -> dict[str, Any]:
    return evaluate_twowiki_rankings(
        questions, degree_prior_rankings(questions, config)
    )


def _macro_comparison(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    mean_delta = (
        candidate["macro"]["mean_complete_gold_evidence_rate"]
        - baseline["macro"]["mean_complete_gold_evidence_rate"]
    )
    per_budget = {
        str(extra): (
            candidate["macro"]["by_extra_budget"][str(extra)][
                "macro_complete_gold_evidence_rate"
            ]
            - baseline["macro"]["by_extra_budget"][str(extra)][
                "macro_complete_gold_evidence_rate"
            ]
        )
        for extra in EXTRA_BUDGETS
    }
    return {
        "absolute_macro_mean_complete_delta": mean_delta,
        "absolute_macro_complete_delta_by_extra_budget": per_budget,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise TwoWikiGraphValidationError("distribution must not be empty")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _null_summary(values: Sequence[float], actual: float) -> dict[str, Any]:
    if not values:
        raise TwoWikiGraphValidationError("null distribution must not be empty")
    exceed_count = sum(value >= actual for value in values)
    return {
        "actual": actual,
        "null_mean": statistics.fmean(values),
        "null_std": statistics.pstdev(values),
        "null_min": min(values),
        "null_p50": _percentile(values, 0.50),
        "null_p95": _percentile(values, 0.95),
        "null_max": max(values),
        "actual_minus_null_mean": actual - statistics.fmean(values),
        "actual_minus_null_max": actual - max(values),
        "null_at_least_actual_count": exceed_count,
        "empirical_one_sided_p": (exceed_count + 1) / (len(values) + 1),
    }


def permutation_null_report(
    questions: Sequence[RetrievalQuestion],
    config: GraphMethodConfig,
    actual_metrics: Mapping[str, Any],
    *,
    repetitions: int = PERMUTATION_REPETITIONS,
    seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """Break graph/document identity while preserving each graph topology."""

    if repetitions < 1:
        raise TwoWikiGraphValidationError("permutation repetitions must be positive")
    distributions: dict[str, list[float]] = {
        "macro_mean_complete": [],
        "macro_full_evidence_mrr": [],
    }
    for extra in EXTRA_BUDGETS:
        distributions[f"macro_complete_extra_{extra}"] = []

    for replicate in range(repetitions):
        permuted = tuple(
            permute_question_graph(question, replicate=replicate, seed=seed)
            for question in questions
        )
        metrics = evaluate_twowiki_rankings(
            permuted, graph_rankings(permuted, config)
        )
        distributions["macro_mean_complete"].append(
            metrics["macro"]["mean_complete_gold_evidence_rate"]
        )
        distributions["macro_full_evidence_mrr"].append(
            metrics["macro"]["full_evidence_mrr"]
        )
        for extra in EXTRA_BUDGETS:
            distributions[f"macro_complete_extra_{extra}"].append(
                metrics["macro"]["by_extra_budget"][str(extra)][
                    "macro_complete_gold_evidence_rate"
                ]
            )

    return {
        "null": "question-local node-label permutation",
        "preserved": (
            "directed graph isomorphism, edge count, degree sequence, document "
            "scores, gold labels, question types, and context budget"
        ),
        "destroyed": "alignment between graph nodes and document/title identities",
        "repetitions": repetitions,
        "seed": seed,
        "macro_mean_complete_gold_evidence_rate": _null_summary(
            distributions["macro_mean_complete"],
            actual_metrics["macro"]["mean_complete_gold_evidence_rate"],
        ),
        "macro_full_evidence_mrr": _null_summary(
            distributions["macro_full_evidence_mrr"],
            actual_metrics["macro"]["full_evidence_mrr"],
        ),
        "macro_complete_rate_by_extra_budget": {
            str(extra): _null_summary(
                distributions[f"macro_complete_extra_{extra}"],
                actual_metrics["macro"]["by_extra_budget"][str(extra)][
                    "macro_complete_gold_evidence_rate"
                ],
            )
            for extra in EXTRA_BUDGETS
        },
    }


def _metrics_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "question_count",
        "question_count_by_type",
        "extra_budgets",
        "by_question_type",
        "macro",
        "micro",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def build_validation_report(
    train_questions: Sequence[RetrievalQuestion],
    dev_questions: Sequence[RetrievalQuestion],
    *,
    base_report: Mapping[str, Any],
    base_report_sha256: str,
    provenance: Mapping[str, Any],
    permutation_repetitions: int = PERMUTATION_REPETITIONS,
    permutation_seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """Validate against strongest baseline, degree-only, and topology nulls."""

    selection = base_report.get("selection")
    if not isinstance(selection, Mapping):
        raise TwoWikiGraphValidationError("base report selection is missing")
    base_gate = base_report.get("gate")
    if not isinstance(base_gate, Mapping) or not base_gate.get("passed"):
        raise TwoWikiGraphValidationError("primary 2Wiki gate did not pass")
    transfer_payload = selection.get("hotpot_transfer_config")
    if not isinstance(transfer_payload, Mapping):
        raise TwoWikiGraphValidationError("transfer configuration is missing")
    transfer_config = GraphMethodConfig(**transfer_payload)
    if transfer_config != FROZEN_HOTPOT_TRANSFER_CONFIG:
        raise TwoWikiGraphValidationError("transfer configuration changed")
    strongest_baseline = selection.get("selected_2wiki_baseline")
    if strongest_baseline not in {"dense", "bm25", "dense_bm25_rrf"}:
        raise TwoWikiGraphValidationError("strongest baseline selection is invalid")

    base_dev = base_report.get("dev")
    base_train = base_report.get("train")
    if not isinstance(base_dev, Mapping) or not isinstance(base_train, Mapping):
        raise TwoWikiGraphValidationError("base report splits are missing")
    actual_train = evaluate_twowiki_rankings(
        train_questions, graph_rankings(train_questions, transfer_config)
    )
    actual_dev = evaluate_twowiki_rankings(
        dev_questions, graph_rankings(dev_questions, transfer_config)
    )
    if not _metrics_match(actual_train, base_train["hotpot_zero_shot_graph"]):
        raise TwoWikiGraphValidationError("train transfer metrics do not reproduce")
    if not _metrics_match(actual_dev, base_dev["hotpot_zero_shot_graph"]):
        raise TwoWikiGraphValidationError("dev transfer metrics do not reproduce")

    strongest_train = evaluate_twowiki_rankings(
        train_questions, baseline_rankings(train_questions, strongest_baseline)
    )
    strongest_dev = evaluate_twowiki_rankings(
        dev_questions, baseline_rankings(dev_questions, strongest_baseline)
    )
    if not _metrics_match(strongest_train, base_train["baselines"][strongest_baseline]):
        raise TwoWikiGraphValidationError("train baseline metrics do not reproduce")
    if not _metrics_match(strongest_dev, base_dev["baselines"][strongest_baseline]):
        raise TwoWikiGraphValidationError("dev baseline metrics do not reproduce")

    degree_results = [
        (config.name, evaluate_degree_prior(train_questions, config))
        for config in graph_candidate_grid()
    ]
    selected_degree_name, selected_degree_train = select_best_twowiki_result(
        degree_results
    )
    config_by_name = {config.name: config for config in graph_candidate_grid()}
    selected_degree_config = config_by_name[selected_degree_name]
    selected_degree_dev = evaluate_degree_prior(
        dev_questions, selected_degree_config
    )

    strong_comparison = _macro_comparison(actual_dev, strongest_dev)
    degree_comparison = _macro_comparison(actual_dev, selected_degree_dev)
    paired = paired_transition_report(
        dev_questions,
        baseline_rankings(dev_questions, strongest_baseline),
        graph_rankings(dev_questions, transfer_config),
    )
    null = permutation_null_report(
        dev_questions,
        transfer_config,
        actual_dev,
        repetitions=permutation_repetitions,
        seed=permutation_seed,
    )

    paired_pass = all(
        row["graph_only_complete_count"] > row["baseline_only_complete_count"]
        and row["exact_mcnemar_two_sided_p"] <= MCNEMAR_P_THRESHOLD
        for row in paired["overall"].values()
    )
    null_mean = null["macro_mean_complete_gold_evidence_rate"]
    checks = {
        "strongest_non_graph_baseline": {
            "baseline": strongest_baseline,
            "required_mean_delta": STRONG_BASELINE_MEAN_DELTA_THRESHOLD,
            "maximum_per_budget_regression": MAX_PER_BUDGET_REGRESSION,
            "observed_mean_delta": strong_comparison[
                "absolute_macro_mean_complete_delta"
            ],
            "observed_per_budget_deltas": strong_comparison[
                "absolute_macro_complete_delta_by_extra_budget"
            ],
            "passed": (
                strong_comparison["absolute_macro_mean_complete_delta"]
                >= STRONG_BASELINE_MEAN_DELTA_THRESHOLD
                and min(
                    strong_comparison[
                        "absolute_macro_complete_delta_by_extra_budget"
                    ].values()
                )
                >= -MAX_PER_BUDGET_REGRESSION
            ),
        },
        "degree_only_control": {
            "required_mean_delta": DEGREE_CONTROL_MEAN_DELTA_THRESHOLD,
            "observed_mean_delta": degree_comparison[
                "absolute_macro_mean_complete_delta"
            ],
            "passed": (
                degree_comparison["absolute_macro_mean_complete_delta"]
                >= DEGREE_CONTROL_MEAN_DELTA_THRESHOLD
            ),
        },
        "node_permutation_null": {
            "empirical_p_threshold": PERMUTATION_P_THRESHOLD,
            "empirical_p": null_mean["empirical_one_sided_p"],
            "actual_minus_null_max": null_mean["actual_minus_null_max"],
            "passed": (
                null_mean["empirical_one_sided_p"] <= PERMUTATION_P_THRESHOLD
                and null_mean["actual_minus_null_max"] > 0.0
            ),
        },
        "paired_mcnemar_every_budget": {
            "p_threshold": MCNEMAR_P_THRESHOLD,
            "passed": paired_pass,
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "task": "2wiki_graph_retrieval_post_primary_robustness_validation",
        "base_report": {
            "path": provenance.get("base_report_path"),
            "sha256": base_report_sha256,
            "transfer_config": asdict(transfer_config),
        },
        "protocol": {
            "validation_declared_after_primary_2wiki_result": True,
            "strongest_baseline_selected_on_2wiki_train": strongest_baseline,
            "degree_control_selected_on_2wiki_train": True,
            "permutation_repetitions": permutation_repetitions,
            "permutation_seed": permutation_seed,
            "dev_used_for_additional_selection": False,
        },
        "provenance": dict(provenance),
        "train": {
            "actual_transfer": actual_train,
            "strongest_baseline": strongest_train,
            "selected_degree_only_name": selected_degree_name,
            "selected_degree_only_config": asdict(selected_degree_config),
            "selected_degree_only_metrics": selected_degree_train,
        },
        "dev": {
            "actual_transfer": actual_dev,
            "strongest_baseline": strongest_dev,
            "actual_vs_strongest_baseline": strong_comparison,
            "selected_degree_only_metrics": selected_degree_dev,
            "actual_vs_degree_only": degree_comparison,
            "paired_transitions_vs_strongest_baseline": paired,
            "node_permutation_null": null,
        },
        "gate": {
            "name": "2wiki_graph_alignment_post_primary_validation_gate_v1",
            "checks": checks,
            "passed": passed,
            "status": (
                "continue_to_musique_adapter"
                if passed
                else "diagnose_2wiki_graph_artifact"
            ),
            "interpretation": (
                "Passing shows the transferred 2Wiki gain exceeds the strongest "
                "non-graph and degree-only controls and depends on observed "
                "graph/document alignment. This validation was declared after "
                "the primary 2Wiki outcome and is not preregistered evidence."
            ),
        },
        "content_contract": (
            "aggregate metrics, configurations, paths, and hashes only; no question "
            "IDs, questions, answers, titles, sentences, evidence text, or labels"
        ),
    }
