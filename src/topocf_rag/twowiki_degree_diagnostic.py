"""Post-validation diagnosis of the 2Wiki receiving-degree shortcut."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import statistics
from typing import Any

from .graph_retrieval import (
    GraphMethodConfig,
    RetrievalQuestion,
    baseline_rankings,
    degree_prior_rankings,
    graph_rankings,
)
from .twowiki import QUESTION_TYPES
from .twowiki_graph_retrieval import (
    EXTRA_BUDGETS,
    evaluate_twowiki_rankings,
    paired_transition_report,
)


DIAGNOSTIC_SCHEMA_VERSION = 1
MATERIAL_METHOD_DELTA = 0.02
ROUTER_ORACLE_HEADROOM = 0.02
ROUTER_UNIQUE_WIN_RATE = 0.02
MAX_PER_BUDGET_REGRESSION = 0.01


class TwoWikiDegreeDiagnosticError(ValueError):
    """Raised when a degree-shortcut diagnostic input is inconsistent."""


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


def compare_metrics(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare type-macro and per-type complete-evidence retrieval."""

    return {
        "macro_mean_complete_delta": (
            candidate["macro"]["mean_complete_gold_evidence_rate"]
            - baseline["macro"]["mean_complete_gold_evidence_rate"]
        ),
        "macro_complete_delta_by_extra_budget": {
            str(extra): (
                candidate["macro"]["by_extra_budget"][str(extra)][
                    "macro_complete_gold_evidence_rate"
                ]
                - baseline["macro"]["by_extra_budget"][str(extra)][
                    "macro_complete_gold_evidence_rate"
                ]
            )
            for extra in EXTRA_BUDGETS
        },
        "by_question_type": {
            question_type: {
                "mean_complete_delta": (
                    candidate["by_question_type"][question_type][
                        "mean_complete_gold_evidence_rate"
                    ]
                    - baseline["by_question_type"][question_type][
                        "mean_complete_gold_evidence_rate"
                    ]
                ),
                "complete_delta_by_extra_budget": {
                    str(extra): (
                        candidate["by_question_type"][question_type][
                            "by_extra_budget"
                        ][str(extra)]["complete_gold_evidence_rate"]
                        - baseline["by_question_type"][question_type][
                            "by_extra_budget"
                        ][str(extra)]["complete_gold_evidence_rate"]
                    )
                    for extra in EXTRA_BUDGETS
                },
            }
            for question_type in QUESTION_TYPES
        },
    }


def _augment_transition_row(
    row: Mapping[str, Any], *, left_name: str, right_name: str
) -> dict[str, Any]:
    count = int(row["question_count"])
    both = int(row["both_complete_count"])
    left_only = int(row["baseline_only_complete_count"])
    right_only = int(row["graph_only_complete_count"])
    neither = int(row["neither_complete_count"])
    left_rate = (both + left_only) / count
    right_rate = (both + right_only) / count
    oracle_rate = (both + left_only + right_only) / count
    return {
        "question_count": count,
        "left_method": left_name,
        "right_method": right_name,
        "both_complete_count": both,
        "left_only_complete_count": left_only,
        "right_only_complete_count": right_only,
        "neither_complete_count": neither,
        "left_complete_rate": left_rate,
        "right_complete_rate": right_rate,
        "oracle_union_complete_rate": oracle_rate,
        "oracle_headroom_over_best_single": (
            oracle_rate - max(left_rate, right_rate)
        ),
        "exact_mcnemar_two_sided_p": row["exact_mcnemar_two_sided_p"],
    }


def complementarity_report(
    questions: Sequence[RetrievalQuestion],
    left_rankings: Sequence[Sequence[int]],
    right_rankings: Sequence[Sequence[int]],
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    """Measure oracle routing headroom without exposing individual examples."""

    paired = paired_transition_report(questions, left_rankings, right_rankings)
    overall = {
        str(extra): _augment_transition_row(
            paired["overall"][str(extra)],
            left_name=left_name,
            right_name=right_name,
        )
        for extra in EXTRA_BUDGETS
    }
    by_type = {
        question_type: {
            str(extra): _augment_transition_row(
                paired["by_question_type"][question_type][str(extra)],
                left_name=left_name,
                right_name=right_name,
            )
            for extra in EXTRA_BUDGETS
        }
        for question_type in QUESTION_TYPES
    }
    macro_by_budget = {}
    for extra in EXTRA_BUDGETS:
        rows = [by_type[kind][str(extra)] for kind in QUESTION_TYPES]
        macro_by_budget[str(extra)] = {
            "oracle_union_complete_rate": statistics.fmean(
                row["oracle_union_complete_rate"] for row in rows
            ),
            "oracle_headroom_over_best_single": statistics.fmean(
                row["oracle_headroom_over_best_single"] for row in rows
            ),
            "left_only_complete_rate": statistics.fmean(
                row["left_only_complete_count"] / row["question_count"]
                for row in rows
            ),
            "right_only_complete_rate": statistics.fmean(
                row["right_only_complete_count"] / row["question_count"]
                for row in rows
            ),
        }
    return {
        "left_method": left_name,
        "right_method": right_name,
        "overall": overall,
        "by_question_type": by_type,
        "macro_by_extra_budget": macro_by_budget,
        "macro_mean_oracle_headroom": statistics.fmean(
            row["oracle_headroom_over_best_single"]
            for row in macro_by_budget.values()
        ),
        "macro_mean_left_only_rate": statistics.fmean(
            row["left_only_complete_rate"] for row in macro_by_budget.values()
        ),
        "macro_mean_right_only_rate": statistics.fmean(
            row["right_only_complete_rate"] for row in macro_by_budget.values()
        ),
    }


def _receiving_degrees(
    question: RetrievalQuestion, direction: str
) -> tuple[int, ...]:
    if direction == "outgoing":
        arcs = set(question.mention_edges)
    elif direction == "incoming":
        arcs = {(target, source) for source, target in question.mention_edges}
    elif direction == "undirected":
        arcs = set(question.mention_edges).union(
            (target, source) for source, target in question.mention_edges
        )
    else:
        raise TwoWikiDegreeDiagnosticError("invalid degree direction")
    degrees = [0] * question.document_count
    for _source, target in arcs:
        if not 0 <= target < question.document_count:
            raise TwoWikiDegreeDiagnosticError("mention edge is out of range")
        degrees[target] += 1
    return tuple(degrees)


def _centrality_subset(
    questions: Sequence[RetrievalQuestion], direction: str
) -> dict[str, Any]:
    if not questions:
        raise TwoWikiDegreeDiagnosticError("centrality subset must not be empty")
    gold_values: list[int] = []
    non_gold_values: list[int] = []
    pairwise_wins = 0
    pairwise_ties = 0
    pairwise_total = 0
    per_question_advantage: list[float] = []
    any_gold_positive = 0
    all_gold_positive = 0
    rankings: list[tuple[int, ...]] = []
    for question in questions:
        degrees = _receiving_degrees(question, direction)
        gold = set(question.gold_indices)
        gold_degree = [degrees[index] for index in gold]
        other_degree = [
            degrees[index]
            for index in range(question.document_count)
            if index not in gold
        ]
        gold_values.extend(gold_degree)
        non_gold_values.extend(other_degree)
        per_question_advantage.append(
            statistics.fmean(gold_degree) - statistics.fmean(other_degree)
        )
        any_gold_positive += any(value > 0 for value in gold_degree)
        all_gold_positive += all(value > 0 for value in gold_degree)
        for left in gold_degree:
            for right in other_degree:
                pairwise_total += 1
                pairwise_wins += left > right
                pairwise_ties += left == right
        rankings.append(
            tuple(
                sorted(
                    range(question.document_count),
                    key=lambda index: (
                        -degrees[index],
                        -question.dense_scores[index],
                        index,
                    ),
                )
            )
        )
    complete_by_budget = {}
    for extra in EXTRA_BUDGETS:
        complete_count = sum(
            set(question.gold_indices).issubset(
                set(ranking[: len(question.gold_indices) + extra])
            )
            for question, ranking in zip(questions, rankings, strict=True)
        )
        complete_by_budget[str(extra)] = {
            "extra_budget": extra,
            "complete_gold_evidence_count": complete_count,
            "complete_gold_evidence_rate": complete_count / len(questions),
        }
    return {
        "question_count": len(questions),
        "receiving_degree_semantics": {
            "outgoing": "literal-title-mention in-degree",
            "incoming": "literal-title-mention out-degree",
            "undirected": "direction-ignored incident degree",
        }[direction],
        "mean_gold_receiving_degree": statistics.fmean(gold_values),
        "mean_non_gold_receiving_degree": statistics.fmean(non_gold_values),
        "mean_per_question_gold_minus_non_gold_degree": statistics.fmean(
            per_question_advantage
        ),
        "gold_vs_non_gold_degree_pairwise_auc": (
            pairwise_wins + 0.5 * pairwise_ties
        )
        / pairwise_total,
        "any_gold_positive_degree_rate": any_gold_positive / len(questions),
        "all_gold_positive_degree_rate": all_gold_positive / len(questions),
        "pure_degree_then_dense_tiebreak_metrics": {
            "question_count": len(questions),
            "by_extra_budget": complete_by_budget,
            "mean_complete_gold_evidence_rate": statistics.fmean(
                row["complete_gold_evidence_rate"]
                for row in complete_by_budget.values()
            ),
        },
    }


def centrality_shortcut_report(
    questions: Sequence[RetrievalQuestion],
) -> dict[str, Any]:
    """Audit whether gold documents are structurally central."""

    grouped = {
        question_type: tuple(
            question
            for question in questions
            if question.question_type == question_type
        )
        for question_type in QUESTION_TYPES
    }
    if any(not subset for subset in grouped.values()):
        raise TwoWikiDegreeDiagnosticError("all question types are required")
    return {
        direction: {
            "overall": _centrality_subset(questions, direction),
            "by_question_type": {
                question_type: _centrality_subset(subset, direction)
                for question_type, subset in grouped.items()
            },
        }
        for direction in ("outgoing", "incoming", "undirected")
    }


def _failed_checks(validation_report: Mapping[str, Any]) -> list[str]:
    gate = validation_report.get("gate")
    checks = gate.get("checks") if isinstance(gate, Mapping) else None
    if not isinstance(checks, Mapping):
        raise TwoWikiDegreeDiagnosticError("validation checks are missing")
    return sorted(
        name
        for name, value in checks.items()
        if not isinstance(value, Mapping) or not value.get("passed")
    )


def build_degree_diagnostic_report(
    train_questions: Sequence[RetrievalQuestion],
    dev_questions: Sequence[RetrievalQuestion],
    *,
    base_report: Mapping[str, Any],
    base_report_sha256: str,
    validation_report: Mapping[str, Any],
    validation_report_sha256: str,
    hotpot_validation_report: Mapping[str, Any],
    hotpot_validation_report_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Determine whether to route methods or pivot to shortcut analysis."""

    failed = _failed_checks(validation_report)
    if failed != ["degree_only_control"]:
        raise TwoWikiDegreeDiagnosticError(
            "diagnostic requires degree-only to be the sole failed check"
        )
    validation_base = validation_report.get("base_report")
    if not isinstance(validation_base, Mapping) or (
        validation_base.get("sha256") != base_report_sha256
    ):
        raise TwoWikiDegreeDiagnosticError("base report hash does not match")
    hotpot_gate = hotpot_validation_report.get("gate")
    if not isinstance(hotpot_gate, Mapping) or not hotpot_gate.get("passed"):
        raise TwoWikiDegreeDiagnosticError("Hotpot robustness gate must pass")

    selection = base_report.get("selection")
    validation_train = validation_report.get("train")
    if not isinstance(selection, Mapping) or not isinstance(
        validation_train, Mapping
    ):
        raise TwoWikiDegreeDiagnosticError("selection metadata is missing")
    tuned_payload = selection.get("selected_2wiki_graph_config")
    transfer_payload = selection.get("hotpot_transfer_config")
    degree_payload = validation_train.get("selected_degree_only_config")
    if not all(
        isinstance(payload, Mapping)
        for payload in (tuned_payload, transfer_payload, degree_payload)
    ):
        raise TwoWikiDegreeDiagnosticError("method configuration is missing")
    tuned_config = GraphMethodConfig(**tuned_payload)
    transfer_config = GraphMethodConfig(**transfer_payload)
    degree_config = GraphMethodConfig(**degree_payload)
    strongest_baseline = selection.get("selected_2wiki_baseline")
    if strongest_baseline not in {"dense", "bm25", "dense_bm25_rrf"}:
        raise TwoWikiDegreeDiagnosticError("strongest baseline is invalid")

    def evaluate_split(questions: Sequence[RetrievalQuestion]) -> dict[str, Any]:
        rankings = {
            "strongest_non_graph": baseline_rankings(
                questions, strongest_baseline
            ),
            "hotpot_transfer_graph": graph_rankings(
                questions, transfer_config
            ),
            "twowiki_tuned_graph": graph_rankings(questions, tuned_config),
            "twowiki_tuned_degree": degree_prior_rankings(
                questions, degree_config
            ),
        }
        return {
            "rankings": rankings,
            "metrics": {
                name: evaluate_twowiki_rankings(questions, value)
                for name, value in rankings.items()
            },
        }

    train = evaluate_split(train_questions)
    dev = evaluate_split(dev_questions)
    base_train = base_report.get("train")
    base_dev = base_report.get("dev")
    validation_dev = validation_report.get("dev")
    if not all(
        isinstance(value, Mapping)
        for value in (base_train, base_dev, validation_dev)
    ):
        raise TwoWikiDegreeDiagnosticError("base split metrics are missing")
    if not _metrics_match(
        train["metrics"]["twowiki_tuned_graph"],
        base_train["selected_2wiki_graph_metrics"],
    ) or not _metrics_match(
        dev["metrics"]["twowiki_tuned_graph"],
        base_dev["selected_2wiki_graph"],
    ):
        raise TwoWikiDegreeDiagnosticError("tuned graph does not reproduce")
    if not _metrics_match(
        train["metrics"]["twowiki_tuned_degree"],
        validation_train["selected_degree_only_metrics"],
    ) or not _metrics_match(
        dev["metrics"]["twowiki_tuned_degree"],
        validation_dev["selected_degree_only_metrics"],
    ):
        raise TwoWikiDegreeDiagnosticError("degree control does not reproduce")

    comparisons = {
        "tuned_graph_vs_degree": compare_metrics(
            dev["metrics"]["twowiki_tuned_graph"],
            dev["metrics"]["twowiki_tuned_degree"],
        ),
        "degree_vs_strongest_non_graph": compare_metrics(
            dev["metrics"]["twowiki_tuned_degree"],
            dev["metrics"]["strongest_non_graph"],
        ),
        "transfer_graph_vs_degree": compare_metrics(
            dev["metrics"]["hotpot_transfer_graph"],
            dev["metrics"]["twowiki_tuned_degree"],
        ),
    }
    complementarity = complementarity_report(
        dev_questions,
        dev["rankings"]["twowiki_tuned_degree"],
        dev["rankings"]["twowiki_tuned_graph"],
        left_name="twowiki_tuned_degree",
        right_name="twowiki_tuned_graph",
    )
    tuned_comparison = comparisons["tuned_graph_vs_degree"]
    tuned_beats_degree = (
        tuned_comparison["macro_mean_complete_delta"]
        >= MATERIAL_METHOD_DELTA
        and min(
            tuned_comparison[
                "macro_complete_delta_by_extra_budget"
            ].values()
        )
        >= -MAX_PER_BUDGET_REGRESSION
    )
    routing_headroom = (
        complementarity["macro_mean_oracle_headroom"]
        >= ROUTER_ORACLE_HEADROOM
        and complementarity["macro_mean_left_only_rate"]
        >= ROUTER_UNIQUE_WIN_RATE
        and complementarity["macro_mean_right_only_rate"]
        >= ROUTER_UNIQUE_WIN_RATE
    )
    if tuned_beats_degree:
        status = "repair_cross_dataset_transfer_then_validate_on_musique"
    elif routing_headroom:
        status = "develop_label_free_selective_router_on_train_only"
    else:
        status = "pivot_to_degree_shortcut_benchmark_audit"

    hotpot_dev = hotpot_validation_report.get("dev")
    if not isinstance(hotpot_dev, Mapping):
        raise TwoWikiDegreeDiagnosticError("Hotpot dev controls are missing")
    hotpot_degree = hotpot_dev.get("degree_only_control")
    if not isinstance(hotpot_degree, Mapping):
        raise TwoWikiDegreeDiagnosticError("Hotpot degree control is missing")

    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "task": "2wiki_receiving_degree_shortcut_diagnostic",
        "scope": "post-validation controlled ten-document context-pool diagnosis",
        "protocol": {
            "declared_after_2wiki_degree_control_failure": True,
            "dev_used_for_method_or_degree_selection": False,
            "gold_labels_used_for_aggregate_diagnosis_only": True,
            "method_materiality_threshold": MATERIAL_METHOD_DELTA,
            "router_oracle_headroom_threshold": ROUTER_ORACLE_HEADROOM,
            "router_unique_win_rate_threshold": ROUTER_UNIQUE_WIN_RATE,
        },
        "artifacts": {
            "base_report_sha256": base_report_sha256,
            "validation_report_sha256": validation_report_sha256,
            "hotpot_validation_report_sha256": (
                hotpot_validation_report_sha256
            ),
        },
        "provenance": dict(provenance),
        "methods": {
            "strongest_non_graph": strongest_baseline,
            "hotpot_transfer_graph": asdict(transfer_config),
            "twowiki_tuned_graph": asdict(tuned_config),
            "twowiki_tuned_degree": asdict(degree_config),
        },
        "cross_dataset_degree_control": {
            "hotpot_selected_graph_minus_degree_dev": hotpot_degree[
                "actual_graph_minus_degree_mean"
            ],
            "twowiki_transfer_graph_minus_degree_dev": comparisons[
                "transfer_graph_vs_degree"
            ]["macro_mean_complete_delta"],
            "twowiki_tuned_graph_minus_degree_dev": comparisons[
                "tuned_graph_vs_degree"
            ]["macro_mean_complete_delta"],
        },
        "train": {
            "method_metrics": train["metrics"],
            "centrality_shortcut": centrality_shortcut_report(
                train_questions
            ),
        },
        "dev": {
            "method_metrics": dev["metrics"],
            "comparisons": comparisons,
            "tuned_graph_degree_complementarity": complementarity,
            "centrality_shortcut": centrality_shortcut_report(dev_questions),
        },
        "decision": {
            "tuned_graph_materially_beats_degree": tuned_beats_degree,
            "material_label_free_routing_headroom": routing_headroom,
            "status": status,
            "interpretation": (
                "This is an exploratory decision diagnostic declared after the "
                "2Wiki degree-control failure. It distinguishes query-conditioned "
                "graph propagation from a dataset-specific receiving-degree "
                "shortcut; it is not confirmatory evidence or full-corpus evaluation."
            ),
        },
        "content_contract": (
            "aggregate metrics, configurations, paths, and hashes only; no question "
            "IDs, questions, answers, titles, sentences, evidence text, or labels"
        ),
    }
