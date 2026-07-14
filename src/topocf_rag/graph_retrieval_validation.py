"""Robustness controls for the graph-aware HotpotQA retrieval kill test."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import math
import random
import statistics
from typing import Any, Mapping, Sequence

from .graph_retrieval import (
    DEFAULT_TOP_KS,
    GraphMethodConfig,
    RetrievalQuestion,
    baseline_rankings,
    evaluate_baseline,
    evaluate_degree_prior,
    evaluate_graph_method,
    graph_candidate_grid,
    graph_rankings,
    select_best_result,
)


VALIDATION_SCHEMA_VERSION = 1
PERMUTATION_REPETITIONS = 200
PERMUTATION_SEED = 20260714
MATCHED_MEAN_DELTA_THRESHOLD = 0.02
DEGREE_CONTROL_MEAN_DELTA_THRESHOLD = 0.02
PERMUTATION_P_THRESHOLD = 0.01
MCNEMAR_P_THRESHOLD = 0.01


class GraphRetrievalValidationError(ValueError):
    """Raised when a robustness-control input violates the frozen protocol."""


def deterministic_node_permutation(
    qid: str,
    document_count: int,
    *,
    replicate: int,
    seed: int = PERMUTATION_SEED,
) -> tuple[int, ...]:
    """Return a question-local non-identity node relabeling."""

    if not isinstance(qid, str) or not qid:
        raise GraphRetrievalValidationError("qid must be a non-empty string")
    if (
        not isinstance(document_count, int)
        or isinstance(document_count, bool)
        or document_count < 2
    ):
        raise GraphRetrievalValidationError("document count must be at least two")
    if not isinstance(replicate, int) or isinstance(replicate, bool) or replicate < 0:
        raise GraphRetrievalValidationError("replicate must be a non-negative integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise GraphRetrievalValidationError("seed must be an integer")
    digest = hashlib.sha256(
        f"{seed}\0{replicate}\0{qid}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    permutation = list(range(document_count))
    rng.shuffle(permutation)
    if permutation == list(range(document_count)):
        permutation = permutation[1:] + permutation[:1]
    return tuple(permutation)


def permute_question_graph(
    question: RetrievalQuestion,
    *,
    replicate: int,
    seed: int = PERMUTATION_SEED,
) -> RetrievalQuestion:
    """Relabel graph nodes while keeping text scores and gold labels fixed."""

    permutation = deterministic_node_permutation(
        question.qid,
        question.document_count,
        replicate=replicate,
        seed=seed,
    )
    permuted_edges = tuple(
        sorted(
            (permutation[source], permutation[target])
            for source, target in question.mention_edges
        )
    )
    return replace(question, mention_edges=permuted_edges)


def _validate_top_ks(top_ks: Sequence[int]) -> tuple[int, ...]:
    if not top_ks or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in top_ks
    ):
        raise GraphRetrievalValidationError(
            "top_ks must contain positive integers"
        )
    normalized = tuple(sorted(set(top_ks)))
    if len(normalized) != len(top_ks):
        raise GraphRetrievalValidationError("top_ks must be unique")
    return normalized


def complete_evidence_flags(
    questions: Sequence[RetrievalQuestion],
    rankings: Sequence[Sequence[int]],
    *,
    top_k: int,
) -> tuple[bool, ...]:
    """Return per-question complete-gold retrieval indicators."""

    if not questions or len(questions) != len(rankings):
        raise GraphRetrievalValidationError(
            "questions and rankings must be non-empty and equally sized"
        )
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise GraphRetrievalValidationError("top_k must be a positive integer")
    flags: list[bool] = []
    for question, ranking_value in zip(questions, rankings, strict=True):
        ranking = tuple(ranking_value)
        if set(ranking) != set(range(question.document_count)):
            raise GraphRetrievalValidationError(
                "ranking must be a full context-index permutation"
            )
        retrieved = set(ranking[: min(top_k, len(ranking))])
        flags.append(set(question.gold_indices).issubset(retrieved))
    return tuple(flags)


def exact_mcnemar_p_value(graph_only_count: int, baseline_only_count: int) -> float:
    """Two-sided exact McNemar/binomial p-value for paired binary outcomes."""

    if (
        not isinstance(graph_only_count, int)
        or isinstance(graph_only_count, bool)
        or graph_only_count < 0
        or not isinstance(baseline_only_count, int)
        or isinstance(baseline_only_count, bool)
        or baseline_only_count < 0
    ):
        raise GraphRetrievalValidationError(
            "discordant counts must be non-negative integers"
        )
    discordant = graph_only_count + baseline_only_count
    if discordant == 0:
        return 1.0
    lower = min(graph_only_count, baseline_only_count)
    tail_numerator = sum(math.comb(discordant, index) for index in range(lower + 1))
    return min(1.0, 2.0 * tail_numerator / (2**discordant))


def paired_transition_report(
    questions: Sequence[RetrievalQuestion],
    baseline: Sequence[Sequence[int]],
    graph: Sequence[Sequence[int]],
    *,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> dict[str, Any]:
    """Report paired wins/losses and exact McNemar tests at each budget."""

    normalized_top_ks = _validate_top_ks(top_ks)
    if len(baseline) != len(graph):
        raise GraphRetrievalValidationError(
            "baseline and graph ranking counts must match"
        )
    by_top_k: dict[str, Any] = {}
    for top_k in normalized_top_ks:
        baseline_flags = complete_evidence_flags(
            questions, baseline, top_k=top_k
        )
        graph_flags = complete_evidence_flags(questions, graph, top_k=top_k)
        both_complete = sum(
            base and candidate
            for base, candidate in zip(baseline_flags, graph_flags, strict=True)
        )
        graph_only = sum(
            not base and candidate
            for base, candidate in zip(baseline_flags, graph_flags, strict=True)
        )
        baseline_only = sum(
            base and not candidate
            for base, candidate in zip(baseline_flags, graph_flags, strict=True)
        )
        neither_complete = len(questions) - both_complete - graph_only - baseline_only
        by_top_k[str(top_k)] = {
            "both_complete_count": both_complete,
            "graph_only_complete_count": graph_only,
            "baseline_only_complete_count": baseline_only,
            "neither_complete_count": neither_complete,
            "net_complete_count_gain": graph_only - baseline_only,
            "net_complete_rate_gain": (graph_only - baseline_only) / len(questions),
            "discordant_count": graph_only + baseline_only,
            "exact_mcnemar_two_sided_p": exact_mcnemar_p_value(
                graph_only, baseline_only
            ),
        }
    return {"question_count": len(questions), "by_top_k": by_top_k}


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise GraphRetrievalValidationError("distribution must not be empty")
    if not 0.0 <= fraction <= 1.0:
        raise GraphRetrievalValidationError("percentile fraction is invalid")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _null_summary(values: Sequence[float], actual: float) -> dict[str, Any]:
    if not values:
        raise GraphRetrievalValidationError("null values must not be empty")
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
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    repetitions: int = PERMUTATION_REPETITIONS,
    seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """Evaluate topology-preserving random node relabelings as a null."""

    normalized_top_ks = _validate_top_ks(top_ks)
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions < 1
    ):
        raise GraphRetrievalValidationError("repetitions must be positive")
    distributions: dict[str, list[float]] = {
        "mean_complete_gold_evidence_rate": [],
        "full_evidence_mrr": [],
    }
    for top_k in normalized_top_ks:
        distributions[f"complete_at_{top_k}"] = []

    for replicate in range(repetitions):
        permuted = tuple(
            permute_question_graph(question, replicate=replicate, seed=seed)
            for question in questions
        )
        metrics = evaluate_graph_method(permuted, config, top_ks=normalized_top_ks)
        distributions["mean_complete_gold_evidence_rate"].append(
            metrics["mean_complete_gold_evidence_rate"]
        )
        distributions["full_evidence_mrr"].append(metrics["full_evidence_mrr"])
        for top_k in normalized_top_ks:
            distributions[f"complete_at_{top_k}"].append(
                metrics["by_top_k"][str(top_k)][
                    "complete_gold_evidence_rate"
                ]
            )

    return {
        "null": "question-local node-label permutation",
        "preserved": (
            "directed graph isomorphism, edge count, in/out degree multiset, "
            "document scores, gold labels, and context budget"
        ),
        "destroyed": "alignment between graph nodes and document/title identities",
        "repetitions": repetitions,
        "seed": seed,
        "mean_complete_gold_evidence_rate": _null_summary(
            distributions["mean_complete_gold_evidence_rate"],
            float(actual_metrics["mean_complete_gold_evidence_rate"]),
        ),
        "full_evidence_mrr": _null_summary(
            distributions["full_evidence_mrr"],
            float(actual_metrics["full_evidence_mrr"]),
        ),
        "complete_gold_evidence_rate_by_top_k": {
            str(top_k): _null_summary(
                distributions[f"complete_at_{top_k}"],
                float(
                    actual_metrics["by_top_k"][str(top_k)][
                        "complete_gold_evidence_rate"
                    ]
                ),
            )
            for top_k in normalized_top_ks
        },
    }


def _best_train_config_per_direction(
    base_report: Mapping[str, Any],
) -> dict[str, GraphMethodConfig]:
    train = base_report.get("train")
    leaderboard = (
        train.get("graph_candidate_leaderboard")
        if isinstance(train, Mapping)
        else None
    )
    if not isinstance(leaderboard, list):
        raise GraphRetrievalValidationError(
            "base report is missing the train graph leaderboard"
        )
    selected: dict[str, GraphMethodConfig] = {}
    for row in leaderboard:
        if not isinstance(row, Mapping) or not isinstance(
            row.get("config"), Mapping
        ):
            raise GraphRetrievalValidationError("graph leaderboard row is invalid")
        config = GraphMethodConfig(**row["config"])
        selected.setdefault(config.direction, config)
    if set(selected) != {"outgoing", "incoming", "undirected"}:
        raise GraphRetrievalValidationError(
            "graph leaderboard does not cover every direction"
        )
    return selected


def _metrics_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare the deterministic metric fields reproduced by this validation."""

    keys = (
        "question_count",
        "by_top_k",
        "mean_complete_gold_evidence_rate",
        "full_evidence_mrr",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def build_validation_report(
    train_questions: Sequence[RetrievalQuestion],
    dev_questions: Sequence[RetrievalQuestion],
    *,
    base_report: Mapping[str, Any],
    base_report_sha256: str,
    provenance: Mapping[str, Any],
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    permutation_repetitions: int = PERMUTATION_REPETITIONS,
    permutation_seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """Build matched, degree-only, permutation, and paired controls."""

    normalized_top_ks = _validate_top_ks(top_ks)
    selection = base_report.get("selection")
    if not isinstance(selection, Mapping):
        raise GraphRetrievalValidationError("base report selection is invalid")
    config_payload = selection.get("selected_graph_config")
    if not isinstance(config_payload, Mapping):
        raise GraphRetrievalValidationError(
            "base report selected graph config is invalid"
        )
    config = GraphMethodConfig(**config_payload)
    matching_baseline = config.seed

    degree_candidates = graph_candidate_grid()
    degree_train_results = [
        (
            candidate.name,
            evaluate_degree_prior(
                train_questions, candidate, top_ks=normalized_top_ks
            ),
        )
        for candidate in degree_candidates
    ]
    selected_degree_name, selected_degree_train_metrics = select_best_result(
        degree_train_results, top_ks=normalized_top_ks
    )
    degree_config_by_name = {
        candidate.name: candidate for candidate in degree_candidates
    }
    selected_degree_config = degree_config_by_name[selected_degree_name]

    split_questions = {"train": train_questions, "dev": dev_questions}
    split_reports: dict[str, Any] = {}
    for split_name, questions in split_questions.items():
        actual_graph = evaluate_graph_method(
            questions, config, top_ks=normalized_top_ks
        )
        matching = evaluate_baseline(
            questions, matching_baseline, top_ks=normalized_top_ks
        )
        degree = (
            selected_degree_train_metrics
            if split_name == "train"
            else evaluate_degree_prior(
                questions, selected_degree_config, top_ks=normalized_top_ks
            )
        )
        base_split = base_report.get(split_name)
        expected_graph = (
            base_split.get("selected_graph_metrics")
            if isinstance(base_split, Mapping)
            else None
        )
        if not isinstance(expected_graph, Mapping) or not _metrics_match(
            actual_graph, expected_graph
        ):
            raise GraphRetrievalValidationError(
                f"{split_name} selected graph metrics do not reproduce base report"
            )
        graph_ranking = graph_rankings(questions, config)
        baseline_ranking = baseline_rankings(questions, matching_baseline)
        split_reports[split_name] = {
            "question_count": len(questions),
            "actual_graph_metrics": actual_graph,
            "matched_non_graph_baseline": {
                "name": matching_baseline,
                "metrics": matching,
                "absolute_mean_delta": (
                    actual_graph["mean_complete_gold_evidence_rate"]
                    - matching["mean_complete_gold_evidence_rate"]
                ),
            },
            "degree_only_control": {
                "definition": (
                    "train-selected seed, direction, and weight blended with receiving "
                    "degree; no query-conditioned neighbor message"
                ),
                "selection": "best of the frozen 36 configurations on train only",
                "selected_name": selected_degree_name,
                "selected_config": asdict(selected_degree_config),
                "metrics": degree,
                "actual_graph_minus_degree_mean": (
                    actual_graph["mean_complete_gold_evidence_rate"]
                    - degree["mean_complete_gold_evidence_rate"]
                ),
            },
            "paired_transitions_vs_matched_baseline": paired_transition_report(
                questions,
                baseline_ranking,
                graph_ranking,
                top_ks=normalized_top_ks,
            ),
            "permutation_null": permutation_null_report(
                questions,
                config,
                actual_graph,
                top_ks=normalized_top_ks,
                repetitions=permutation_repetitions,
                seed=permutation_seed,
            ),
        }

    direction_configs = _best_train_config_per_direction(base_report)
    direction_ablation = {
        direction: {
            "selection": "best configuration for this direction on frozen train",
            "config": asdict(direction_config),
            "dev_metrics": evaluate_graph_method(
                dev_questions, direction_config, top_ks=normalized_top_ks
            ),
        }
        for direction, direction_config in sorted(direction_configs.items())
    }

    dev = split_reports["dev"]
    same_seed_delta = dev["matched_non_graph_baseline"]["absolute_mean_delta"]
    degree_delta = dev["degree_only_control"][
        "actual_graph_minus_degree_mean"
    ]
    null_mean = dev["permutation_null"][
        "mean_complete_gold_evidence_rate"
    ]
    paired_by_k = dev["paired_transitions_vs_matched_baseline"]["by_top_k"]
    paired_pass = all(
        row["graph_only_complete_count"] > row["baseline_only_complete_count"]
        and row["exact_mcnemar_two_sided_p"] <= MCNEMAR_P_THRESHOLD
        for row in paired_by_k.values()
    )
    checks = {
        "matched_non_graph_delta": {
            "threshold": MATCHED_MEAN_DELTA_THRESHOLD,
            "value": same_seed_delta,
            "passed": same_seed_delta >= MATCHED_MEAN_DELTA_THRESHOLD,
        },
        "degree_only_delta": {
            "threshold": DEGREE_CONTROL_MEAN_DELTA_THRESHOLD,
            "value": degree_delta,
            "passed": degree_delta >= DEGREE_CONTROL_MEAN_DELTA_THRESHOLD,
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
        "paired_mcnemar_every_k": {
            "p_threshold": MCNEMAR_P_THRESHOLD,
            "passed": paired_pass,
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "task": "hotpot_graph_retrieval_robustness_validation",
        "base_report": {
            "sha256": base_report_sha256,
            "selected_graph_method": selection.get("selected_graph_method"),
            "selected_graph_config": asdict(config),
        },
        "protocol": {
            "top_ks": list(normalized_top_ks),
            "matching_baseline": matching_baseline,
            "permutation_repetitions": permutation_repetitions,
            "permutation_seed": permutation_seed,
            "dev_used_for_additional_model_selection": False,
            "direction_ablation_selection": "separate train-only best per direction",
            "degree_control_selection": "best frozen configuration on train only",
        },
        "provenance": dict(provenance),
        "train": split_reports["train"],
        "dev": split_reports["dev"],
        "direction_ablation": direction_ablation,
        "gate": {
            "name": "hotpot_graph_alignment_robustness_gate_v1",
            "checks": checks,
            "passed": passed,
            "status": (
                "continue_to_2wiki_adapter"
                if passed
                else "stop_and_diagnose_graph_artifact"
            ),
            "interpretation": (
                "Passing shows that retrieval gains require the observed alignment "
                "between document identities and title-mention topology inside the "
                "controlled HotpotQA context pool. It does not establish direction "
                "reasoning, full-corpus performance, or a final TopoCF certificate."
            ),
        },
        "content_contract": (
            "aggregate metrics, configurations, paths, and hashes only; no question "
            "IDs, questions, answers, titles, sentences, or supporting facts"
        ),
    }
