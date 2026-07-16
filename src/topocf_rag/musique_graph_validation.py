"""Post-primary robustness validation for MuSiQue graph retrieval."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
import math
import random
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
from .musique_graph_retrieval import (
    EXTRA_BUDGETS,
    evaluate_musique_rankings,
)
from .musique_splits import (
    HOP_COUNTS,
    TITLE_COLLISION_STRATA,
    TRAIN_PER_CELL,
    all_cells,
    cell_name,
)


MUSIQUE_VALIDATION_SCHEMA_VERSION = 1
EXPECTED_STAGE_E_REPORT_SHA256 = (
    "b248dd17a432f305730671169b99fde68ef6a4171c2079c8cc85d5db7c75d79a"
)
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 20260716
CONFIG_BOOTSTRAP_REPETITIONS = 500
CONFIG_BOOTSTRAP_SEED = 20260717
CI_LOW = 0.025
CI_HIGH = 0.975


class MuSiQueGraphValidationError(ValueError):
    """Raised when a Stage F validation invariant is violated."""


OutcomeRows = dict[str, dict[str, tuple[Any, ...]]]


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise MuSiQueGraphValidationError("distribution must not be empty")
    if not 0.0 <= fraction <= 1.0:
        raise MuSiQueGraphValidationError("percentile must be in [0, 1]")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution_summary(
    values: Sequence[float], actual: float
) -> dict[str, Any]:
    return {
        "actual": actual,
        "bootstrap_mean": statistics.fmean(values),
        "bootstrap_std": statistics.pstdev(values),
        "percentile_95_low": _percentile(values, CI_LOW),
        "percentile_95_high": _percentile(values, CI_HIGH),
        "strictly_positive_replicate_count": sum(value > 0.0 for value in values),
        "strictly_positive_replicate_rate": (
            sum(value > 0.0 for value in values) / len(values)
        ),
    }


def _validate_question_sets(
    train_questions: Sequence[RetrievalQuestion],
    dev_questions: Sequence[RetrievalQuestion],
) -> None:
    for split, questions in (("train", train_questions), ("dev", dev_questions)):
        if not questions:
            raise MuSiQueGraphValidationError(f"{split} questions are empty")
        counts = Counter(question.question_type for question in questions)
        if set(counts) != set(all_cells()):
            raise MuSiQueGraphValidationError(
                f"{split} must contain all nine frozen cells"
            )
        if len({question.qid for question in questions}) != len(questions):
            raise MuSiQueGraphValidationError(
                f"{split} question IDs are not unique"
            )
        if split == "train" and any(
            counts[cell] != TRAIN_PER_CELL for cell in all_cells()
        ):
            raise MuSiQueGraphValidationError(
                "train must contain 200 questions in every frozen cell"
            )
    if {question.qid for question in train_questions}.intersection(
        question.qid for question in dev_questions
    ):
        raise MuSiQueGraphValidationError("train and dev IDs overlap")


def _ranking_outcomes(
    questions: Sequence[RetrievalQuestion],
    rankings: Sequence[Sequence[int]],
) -> OutcomeRows:
    if not questions or len(questions) != len(rankings):
        raise MuSiQueGraphValidationError("ranking inputs are invalid")
    complete: dict[str, list[tuple[int, ...]]] = {
        cell: [] for cell in all_cells()
    }
    reciprocal: dict[str, list[float]] = {cell: [] for cell in all_cells()}
    for question, ranking_value in zip(questions, rankings, strict=True):
        cell = question.question_type
        if cell not in complete:
            raise MuSiQueGraphValidationError("question cell is invalid")
        ranking = tuple(ranking_value)
        if set(ranking) != set(range(question.document_count)):
            raise MuSiQueGraphValidationError(
                "ranking must be a full paragraph-index permutation"
            )
        gold = set(question.gold_indices)
        rank_by_index = {
            index: rank for rank, index in enumerate(ranking, start=1)
        }
        reciprocal[cell].append(
            1.0 / max(rank_by_index[index] for index in gold)
        )
        complete[cell].append(
            tuple(
                int(
                    gold.issubset(
                        set(ranking[: len(gold) + extra_budget])
                    )
                )
                for extra_budget in EXTRA_BUDGETS
            )
        )
    if any(not complete[cell] for cell in all_cells()):
        raise MuSiQueGraphValidationError("one or more cells are empty")
    return {
        cell: {
            "complete": tuple(complete[cell]),
            "reciprocal_full_rank": tuple(reciprocal[cell]),
        }
        for cell in all_cells()
    }


def _outcome_summary(
    outcomes: OutcomeRows,
    cells: Sequence[str],
    *,
    sampled_indices: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    if not cells or any(cell not in outcomes for cell in cells):
        raise MuSiQueGraphValidationError("summary cells are invalid")
    by_cell: dict[str, Any] = {}
    for cell in cells:
        complete_rows = outcomes[cell]["complete"]
        reciprocal_rows = outcomes[cell]["reciprocal_full_rank"]
        indices = (
            tuple(range(len(complete_rows)))
            if sampled_indices is None
            else tuple(sampled_indices[cell])
        )
        if not indices:
            raise MuSiQueGraphValidationError("summary sample is empty")
        budget_rates = {
            str(extra): statistics.fmean(
                complete_rows[index][budget_index] for index in indices
            )
            for budget_index, extra in enumerate(EXTRA_BUDGETS)
        }
        by_cell[cell] = {
            "question_count": len(indices),
            "complete_gold_evidence_rate_by_extra_budget": budget_rates,
            "mean_complete_gold_evidence_rate": statistics.fmean(
                budget_rates.values()
            ),
            "full_evidence_mrr": statistics.fmean(
                reciprocal_rows[index] for index in indices
            ),
        }
    return {
        "cell_count": len(cells),
        "question_count": sum(row["question_count"] for row in by_cell.values()),
        "by_cell": by_cell,
        "macro": {
            "complete_gold_evidence_rate_by_extra_budget": {
                str(extra): statistics.fmean(
                    by_cell[cell][
                        "complete_gold_evidence_rate_by_extra_budget"
                    ][str(extra)]
                    for cell in cells
                )
                for extra in EXTRA_BUDGETS
            },
            "mean_complete_gold_evidence_rate": statistics.fmean(
                by_cell[cell]["mean_complete_gold_evidence_rate"]
                for cell in cells
            ),
            "full_evidence_mrr": statistics.fmean(
                by_cell[cell]["full_evidence_mrr"] for cell in cells
            ),
        },
    }


def _selection_key(summary: Mapping[str, Any]) -> tuple[float, ...]:
    macro = summary["macro"]
    by_budget = macro["complete_gold_evidence_rate_by_extra_budget"]
    return (
        float(macro["mean_complete_gold_evidence_rate"]),
        *(float(by_budget[str(extra)]) for extra in reversed(EXTRA_BUDGETS)),
        float(macro["full_evidence_mrr"]),
    )


def _select_outcome(
    named_outcomes: Mapping[str, OutcomeRows],
    cells: Sequence[str],
    *,
    sampled_indices: Mapping[str, Sequence[int]] | None = None,
) -> tuple[str, dict[str, Any]]:
    if not named_outcomes:
        raise MuSiQueGraphValidationError("candidate outcomes are empty")
    best_name: str | None = None
    best_summary: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    for name, outcomes in named_outcomes.items():
        summary = _outcome_summary(
            outcomes, cells, sampled_indices=sampled_indices
        )
        key = _selection_key(summary)
        if best_key is None or key > best_key:
            best_name, best_summary, best_key = name, summary, key
    assert best_name is not None and best_summary is not None
    return best_name, best_summary


def _metric_groups() -> dict[str, tuple[str, ...]]:
    groups = {"overall": all_cells()}
    groups.update(
        {
            f"hop:{hop}": tuple(
                cell_name(hop, status) for status in TITLE_COLLISION_STRATA
            )
            for hop in HOP_COUNTS
        }
    )
    groups.update(
        {
            f"collision:{status}": tuple(
                cell_name(hop, status) for hop in HOP_COUNTS
            )
            for status in TITLE_COLLISION_STRATA
        }
    )
    groups.update({f"cell:{cell}": (cell,) for cell in all_cells()})
    return groups


def paired_stratified_bootstrap(
    candidate: OutcomeRows,
    control: OutcomeRows,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap paired complete-retrieval deltas within every frozen cell."""

    if repetitions < 1:
        raise MuSiQueGraphValidationError(
            "bootstrap repetitions must be positive"
        )
    for cell in all_cells():
        if len(candidate[cell]["complete"]) != len(control[cell]["complete"]):
            raise MuSiQueGraphValidationError(
                "paired control cell sizes do not match"
            )
    groups = _metric_groups()
    actual_cell_budget: dict[str, tuple[float, ...]] = {}
    for cell in all_cells():
        left = candidate[cell]["complete"]
        right = control[cell]["complete"]
        actual_cell_budget[cell] = tuple(
            statistics.fmean(
                left[index][budget_index] - right[index][budget_index]
                for index in range(len(left))
            )
            for budget_index in range(len(EXTRA_BUDGETS))
        )

    distributions = {
        group: {
            "mean": [],
            **{str(extra): [] for extra in EXTRA_BUDGETS},
        }
        for group in groups
    }
    rng = random.Random(seed)
    for _replicate in range(repetitions):
        replicate_cell_budget: dict[str, tuple[float, ...]] = {}
        for cell in all_cells():
            left = candidate[cell]["complete"]
            right = control[cell]["complete"]
            sampled = tuple(rng.randrange(len(left)) for _ in range(len(left)))
            replicate_cell_budget[cell] = tuple(
                statistics.fmean(
                    left[index][budget_index] - right[index][budget_index]
                    for index in sampled
                )
                for budget_index in range(len(EXTRA_BUDGETS))
            )
        for group, cells in groups.items():
            budget_values = tuple(
                statistics.fmean(
                    replicate_cell_budget[cell][budget_index] for cell in cells
                )
                for budget_index in range(len(EXTRA_BUDGETS))
            )
            distributions[group]["mean"].append(
                statistics.fmean(budget_values)
            )
            for budget_index, extra in enumerate(EXTRA_BUDGETS):
                distributions[group][str(extra)].append(
                    budget_values[budget_index]
                )

    def summarize(group: str) -> dict[str, Any]:
        cells = groups[group]
        actual_budgets = tuple(
            statistics.fmean(
                actual_cell_budget[cell][budget_index] for cell in cells
            )
            for budget_index in range(len(EXTRA_BUDGETS))
        )
        return {
            "cell_count": len(cells),
            "mean_across_extra_budgets": _distribution_summary(
                distributions[group]["mean"],
                statistics.fmean(actual_budgets),
            ),
            "by_extra_budget": {
                str(extra): _distribution_summary(
                    distributions[group][str(extra)],
                    actual_budgets[budget_index],
                )
                for budget_index, extra in enumerate(EXTRA_BUDGETS)
            },
        }

    return {
        "protocol": {
            "paired_within_question": True,
            "resampling": "with replacement independently within each of nine cells",
            "aggregate": "equal-weight macro across cells, then mean across budgets",
            "confidence_interval": "percentile 95%",
            "repetitions": repetitions,
            "seed": seed,
        },
        "overall": summarize("overall"),
        "by_hop": {str(hop): summarize(f"hop:{hop}") for hop in HOP_COUNTS},
        "by_title_collision": {
            status: summarize(f"collision:{status}")
            for status in TITLE_COLLISION_STRATA
        },
        "by_cell": {cell: summarize(f"cell:{cell}") for cell in all_cells()},
    }


def configuration_bootstrap_stability(
    candidate_outcomes: Mapping[str, OutcomeRows],
    config_by_name: Mapping[str, GraphMethodConfig],
    *,
    primary_name: str,
    repetitions: int = CONFIG_BOOTSTRAP_REPETITIONS,
    seed: int = CONFIG_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Repeat train-only selection under nine-cell stratified bootstrap."""

    if repetitions < 1:
        raise MuSiQueGraphValidationError(
            "configuration bootstrap repetitions must be positive"
        )
    if primary_name not in candidate_outcomes:
        raise MuSiQueGraphValidationError("primary configuration is missing")
    rng = random.Random(seed)
    selected = Counter()
    family = Counter()
    seed_histogram = Counter()
    direction = Counter()
    exemplar = next(iter(candidate_outcomes.values()))
    for _replicate in range(repetitions):
        sampled = {
            cell: tuple(
                rng.randrange(len(exemplar[cell]["complete"]))
                for _ in range(len(exemplar[cell]["complete"]))
            )
            for cell in all_cells()
        }
        name, _summary = _select_outcome(
            candidate_outcomes, all_cells(), sampled_indices=sampled
        )
        config = config_by_name[name]
        selected[name] += 1
        family[config.family] += 1
        seed_histogram[config.seed] += 1
        direction[config.direction] += 1
    primary_count = selected[primary_name]
    return {
        "repetitions": repetitions,
        "seed": seed,
        "resampling": "with replacement independently within every balanced train cell",
        "selection_uses_dev": False,
        "primary_configuration": primary_name,
        "primary_configuration_selected_count": primary_count,
        "primary_configuration_selected_rate": primary_count / repetitions,
        "selected_configuration_histogram": dict(
            sorted(selected.items(), key=lambda item: (-item[1], item[0]))
        ),
        "family_histogram": dict(sorted(family.items())),
        "seed_histogram": dict(sorted(seed_histogram.items())),
        "direction_histogram": dict(sorted(direction.items())),
    }


def _comparison_from_summaries(
    candidate: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_macro = candidate["macro"]
    control_macro = control["macro"]
    by_budget = {
        str(extra): (
            candidate_macro["complete_gold_evidence_rate_by_extra_budget"][
                str(extra)
            ]
            - control_macro["complete_gold_evidence_rate_by_extra_budget"][
                str(extra)
            ]
        )
        for extra in EXTRA_BUDGETS
    }
    return {
        "mean_complete_gold_evidence_rate_delta": (
            candidate_macro["mean_complete_gold_evidence_rate"]
            - control_macro["mean_complete_gold_evidence_rate"]
        ),
        "complete_gold_evidence_rate_delta_by_extra_budget": by_budget,
    }


def leave_one_cell_out_report(
    graph_outcomes: Mapping[str, OutcomeRows],
    degree_outcomes: Mapping[str, OutcomeRows],
    baseline_outcomes: Mapping[str, OutcomeRows],
) -> dict[str, Any]:
    """Select on eight train cells and score the ninth, never touching dev."""

    rows: dict[str, Any] = {}
    for held_out in all_cells():
        selection_cells = tuple(cell for cell in all_cells() if cell != held_out)
        graph_name, _graph_selection = _select_outcome(
            graph_outcomes, selection_cells
        )
        degree_name, _degree_selection = _select_outcome(
            degree_outcomes, selection_cells
        )
        baseline_name, _baseline_selection = _select_outcome(
            baseline_outcomes, selection_cells
        )
        graph_held = _outcome_summary(graph_outcomes[graph_name], (held_out,))
        degree_held = _outcome_summary(
            degree_outcomes[degree_name], (held_out,)
        )
        baseline_held = _outcome_summary(
            baseline_outcomes[baseline_name], (held_out,)
        )
        rows[held_out] = {
            "selection_cell_count": len(selection_cells),
            "selected_graph_method": graph_name,
            "selected_degree_only_method": degree_name,
            "selected_baseline": baseline_name,
            "held_out_question_count": graph_held["question_count"],
            "graph_vs_baseline": _comparison_from_summaries(
                graph_held, baseline_held
            ),
            "graph_vs_degree_only": _comparison_from_summaries(
                graph_held, degree_held
            ),
        }

    def aggregate(comparison_name: str) -> dict[str, Any]:
        mean_values = [
            row[comparison_name]["mean_complete_gold_evidence_rate_delta"]
            for row in rows.values()
        ]
        return {
            "macro_mean_complete_gold_evidence_rate_delta": statistics.fmean(
                mean_values
            ),
            "positive_cell_count": sum(value > 0.0 for value in mean_values),
            "zero_cell_count": sum(value == 0.0 for value in mean_values),
            "negative_cell_count": sum(value < 0.0 for value in mean_values),
            "macro_complete_gold_evidence_rate_delta_by_extra_budget": {
                str(extra): statistics.fmean(
                    row[comparison_name][
                        "complete_gold_evidence_rate_delta_by_extra_budget"
                    ][str(extra)]
                    for row in rows.values()
                )
                for extra in EXTRA_BUDGETS
            },
        }

    return {
        "protocol": {
            "data": "balanced train only",
            "folds": len(all_cells()),
            "selection_cells_per_fold": len(all_cells()) - 1,
            "held_out_cells_per_fold": 1,
            "graph_baseline_and_degree_selected_independently_per_fold": True,
            "dev_used": False,
        },
        "by_held_out_cell": rows,
        "macro_across_held_out_cells": {
            "graph_vs_baseline": aggregate("graph_vs_baseline"),
            "graph_vs_degree_only": aggregate("graph_vs_degree_only"),
        },
    }


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "macro": metrics["macro"],
        "by_hop": metrics["by_hop"],
        "by_title_collision": metrics["by_title_collision"],
    }


def _metrics_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "question_count",
        "question_count_by_cell",
        "extra_budgets",
        "by_cell",
        "macro",
        "by_hop",
        "by_title_collision",
        "micro",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def _macro_comparison(
    candidate: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "absolute_macro_mean_complete_delta": (
            candidate["macro"]["mean_complete_gold_evidence_rate"]
            - control["macro"]["mean_complete_gold_evidence_rate"]
        ),
        "absolute_macro_complete_delta_by_extra_budget": {
            str(extra): (
                candidate["macro"]["by_extra_budget"][str(extra)][
                    "macro_complete_gold_evidence_rate"
                ]
                - control["macro"]["by_extra_budget"][str(extra)][
                    "macro_complete_gold_evidence_rate"
                ]
            )
            for extra in EXTRA_BUDGETS
        },
    }


def _direction_sensitivity(
    train_graph_outcomes: Mapping[str, OutcomeRows],
    config_by_name: Mapping[str, GraphMethodConfig],
    train_questions: Sequence[RetrievalQuestion],
    dev_questions: Sequence[RetrievalQuestion],
    selected_config: GraphMethodConfig,
    dev_baseline_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    selected_per_direction: dict[str, Any] = {}
    for direction in ("outgoing", "incoming", "undirected"):
        candidates = {
            name: outcomes
            for name, outcomes in train_graph_outcomes.items()
            if config_by_name[name].direction == direction
        }
        name, train_summary = _select_outcome(candidates, all_cells())
        config = config_by_name[name]
        dev_metrics = evaluate_musique_rankings(
            dev_questions, graph_rankings(dev_questions, config)
        )
        selected_per_direction[direction] = {
            "selected_on_train": name,
            "config": asdict(config),
            "train_selection_macro": train_summary["macro"],
            "dev_metrics": _compact_metrics(dev_metrics),
            "dev_vs_selected_baseline": _macro_comparison(
                dev_metrics, dev_baseline_metrics
            ),
        }

    fixed_variants: dict[str, Any] = {}
    for direction in ("outgoing", "incoming", "undirected"):
        config = replace(selected_config, direction=direction)
        train_metrics = evaluate_musique_rankings(
            train_questions, graph_rankings(train_questions, config)
        )
        dev_metrics = evaluate_musique_rankings(
            dev_questions, graph_rankings(dev_questions, config)
        )
        fixed_variants[direction] = {
            "config": asdict(config),
            "train_metrics": _compact_metrics(train_metrics),
            "dev_metrics": _compact_metrics(dev_metrics),
            "dev_vs_selected_baseline": _macro_comparison(
                dev_metrics, dev_baseline_metrics
            ),
        }
    return {
        "protocol": {
            "direction_specific_config_selected_on_train_only": True,
            "fixed_primary_config_variants_are_post_primary_sensitivity": True,
            "dev_used_for_selection": False,
        },
        "train_selected_within_each_direction": selected_per_direction,
        "fixed_primary_family_seed_and_weight": fixed_variants,
    }


def build_musique_validation_report(
    train_questions: Sequence[RetrievalQuestion],
    dev_questions: Sequence[RetrievalQuestion],
    *,
    base_report: Mapping[str, Any],
    base_report_sha256: str,
    provenance: Mapping[str, Any],
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    config_bootstrap_repetitions: int = CONFIG_BOOTSTRAP_REPETITIONS,
    config_bootstrap_seed: int = CONFIG_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Reproduce Stage E and run content-free post-primary controls."""

    _validate_question_sets(train_questions, dev_questions)
    if base_report_sha256 != EXPECTED_STAGE_E_REPORT_SHA256:
        raise MuSiQueGraphValidationError("Stage E report SHA256 changed")
    if base_report.get("task") != (
        "musique_occurrence_graph_shortcut_controlled_retrieval"
    ):
        raise MuSiQueGraphValidationError("Stage E task changed")
    base_gate = base_report.get("gate")
    if not isinstance(base_gate, Mapping) or not base_gate.get("passed"):
        raise MuSiQueGraphValidationError("Stage E primary gate did not pass")
    selection = base_report.get("selection")
    base_train = base_report.get("train")
    base_dev = base_report.get("dev")
    if not all(isinstance(row, Mapping) for row in (selection, base_train, base_dev)):
        raise MuSiQueGraphValidationError("Stage E report sections are missing")

    selected_baseline_name = selection.get("selected_baseline")
    if selected_baseline_name not in {"dense", "bm25", "dense_bm25_rrf"}:
        raise MuSiQueGraphValidationError("Stage E baseline is invalid")
    graph_payload = selection.get("selected_graph_config")
    degree_payload = selection.get("selected_degree_only_config")
    if not isinstance(graph_payload, Mapping) or not isinstance(
        degree_payload, Mapping
    ):
        raise MuSiQueGraphValidationError("Stage E configurations are missing")
    selected_graph_config = GraphMethodConfig(**graph_payload)
    selected_degree_config = GraphMethodConfig(**degree_payload)

    candidates = graph_candidate_grid()
    config_by_name = {config.name: config for config in candidates}
    if len(config_by_name) != len(candidates):
        raise MuSiQueGraphValidationError("graph grid contains duplicate names")
    if selected_graph_config.name not in config_by_name:
        raise MuSiQueGraphValidationError("selected graph left the frozen grid")
    if selected_degree_config.name not in config_by_name:
        raise MuSiQueGraphValidationError("selected degree control left the grid")

    train_graph_rankings = {
        config.name: graph_rankings(train_questions, config)
        for config in candidates
    }
    train_degree_rankings = {
        config.name: degree_prior_rankings(train_questions, config)
        for config in candidates
    }
    train_baseline_rankings = {
        name: baseline_rankings(train_questions, name)
        for name in ("dense", "bm25", "dense_bm25_rrf")
    }
    train_graph_outcomes = {
        name: _ranking_outcomes(train_questions, rankings)
        for name, rankings in train_graph_rankings.items()
    }
    train_degree_outcomes = {
        name: _ranking_outcomes(train_questions, rankings)
        for name, rankings in train_degree_rankings.items()
    }
    train_baseline_outcomes = {
        name: _ranking_outcomes(train_questions, rankings)
        for name, rankings in train_baseline_rankings.items()
    }
    reproduced_graph_name, _ = _select_outcome(
        train_graph_outcomes, all_cells()
    )
    reproduced_degree_name, _ = _select_outcome(
        train_degree_outcomes, all_cells()
    )
    reproduced_baseline_name, _ = _select_outcome(
        train_baseline_outcomes, all_cells()
    )

    selected_train_graph = evaluate_musique_rankings(
        train_questions, train_graph_rankings[selected_graph_config.name]
    )
    selected_train_degree = evaluate_musique_rankings(
        train_questions, train_degree_rankings[selected_degree_config.name]
    )
    selected_train_baseline = evaluate_musique_rankings(
        train_questions, train_baseline_rankings[selected_baseline_name]
    )
    selected_dev_graph_rankings = graph_rankings(
        dev_questions, selected_graph_config
    )
    selected_dev_degree_rankings = degree_prior_rankings(
        dev_questions, selected_degree_config
    )
    selected_dev_baseline_rankings = baseline_rankings(
        dev_questions, selected_baseline_name
    )
    selected_dev_graph = evaluate_musique_rankings(
        dev_questions, selected_dev_graph_rankings
    )
    selected_dev_degree = evaluate_musique_rankings(
        dev_questions, selected_dev_degree_rankings
    )
    selected_dev_baseline = evaluate_musique_rankings(
        dev_questions, selected_dev_baseline_rankings
    )

    reproduction_checks = {
        "selected_graph_name": reproduced_graph_name
        == selection.get("selected_graph_method"),
        "selected_degree_name": reproduced_degree_name
        == selection.get("selected_degree_only_method"),
        "selected_baseline_name": reproduced_baseline_name
        == selected_baseline_name,
        "train_graph_metrics": _metrics_match(
            selected_train_graph, base_train["selected_graph_metrics"]
        ),
        "train_degree_metrics": _metrics_match(
            selected_train_degree, base_train["selected_degree_only_metrics"]
        ),
        "train_baseline_metrics": _metrics_match(
            selected_train_baseline,
            base_train["baselines"][selected_baseline_name],
        ),
        "dev_graph_metrics": _metrics_match(
            selected_dev_graph, base_dev["selected_graph"]
        ),
        "dev_degree_metrics": _metrics_match(
            selected_dev_degree, base_dev["selected_degree_only"]
        ),
        "dev_baseline_metrics": _metrics_match(
            selected_dev_baseline,
            base_dev["baselines"][selected_baseline_name],
        ),
    }
    if not all(reproduction_checks.values()):
        raise MuSiQueGraphValidationError(
            "Stage E selections or metrics did not reproduce"
        )

    dev_graph_outcomes = _ranking_outcomes(
        dev_questions, selected_dev_graph_rankings
    )
    dev_degree_outcomes = _ranking_outcomes(
        dev_questions, selected_dev_degree_rankings
    )
    dev_baseline_outcomes = _ranking_outcomes(
        dev_questions, selected_dev_baseline_rankings
    )
    same_config_degree_rankings = degree_prior_rankings(
        dev_questions, selected_graph_config
    )
    same_config_degree_metrics = evaluate_musique_rankings(
        dev_questions, same_config_degree_rankings
    )
    same_config_degree_outcomes = _ranking_outcomes(
        dev_questions, same_config_degree_rankings
    )

    bootstrap_vs_baseline = paired_stratified_bootstrap(
        dev_graph_outcomes,
        dev_baseline_outcomes,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    bootstrap_vs_independent_degree = paired_stratified_bootstrap(
        dev_graph_outcomes,
        dev_degree_outcomes,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    bootstrap_vs_same_config_degree = paired_stratified_bootstrap(
        dev_graph_outcomes,
        same_config_degree_outcomes,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    config_stability = configuration_bootstrap_stability(
        train_graph_outcomes,
        config_by_name,
        primary_name=selected_graph_config.name,
        repetitions=config_bootstrap_repetitions,
        seed=config_bootstrap_seed,
    )
    leave_one_cell_out = leave_one_cell_out_report(
        train_graph_outcomes,
        train_degree_outcomes,
        train_baseline_outcomes,
    )
    direction_sensitivity = _direction_sensitivity(
        train_graph_outcomes,
        config_by_name,
        train_questions,
        dev_questions,
        selected_graph_config,
        selected_dev_baseline,
    )

    checks = {
        "stage_e_exact_reproduction": {
            "passed": all(reproduction_checks.values()),
            "checks": reproduction_checks,
        },
        "stratified_bootstrap_vs_selected_baseline": {
            "percentile_95_low": bootstrap_vs_baseline["overall"][
                "mean_across_extra_budgets"
            ]["percentile_95_low"],
            "required_lower_bound": 0.0,
            "passed": bootstrap_vs_baseline["overall"][
                "mean_across_extra_budgets"
            ]["percentile_95_low"]
            > 0.0,
        },
        "stratified_bootstrap_vs_independent_degree_only": {
            "percentile_95_low": bootstrap_vs_independent_degree["overall"][
                "mean_across_extra_budgets"
            ]["percentile_95_low"],
            "required_lower_bound": 0.0,
            "passed": bootstrap_vs_independent_degree["overall"][
                "mean_across_extra_budgets"
            ]["percentile_95_low"]
            > 0.0,
        },
        "stratified_bootstrap_vs_same_config_degree_only": {
            "percentile_95_low": bootstrap_vs_same_config_degree["overall"][
                "mean_across_extra_budgets"
            ]["percentile_95_low"],
            "required_lower_bound": 0.0,
            "passed": bootstrap_vs_same_config_degree["overall"][
                "mean_across_extra_budgets"
            ]["percentile_95_low"]
            > 0.0,
        },
    }
    passed = all(row["passed"] for row in checks.values())
    cell_deltas = {
        cell: bootstrap_vs_baseline["by_cell"][cell][
            "mean_across_extra_budgets"
        ]["actual"]
        for cell in all_cells()
    }
    return {
        "schema_version": MUSIQUE_VALIDATION_SCHEMA_VERSION,
        "task": "musique_graph_retrieval_post_primary_robustness_validation",
        "base_report": {
            "path": provenance.get("base_report_path"),
            "sha256": base_report_sha256,
            "primary_gate_passed": True,
        },
        "protocol": {
            "declared_after_stage_e_primary_result": True,
            "not_preregistered_evidence": True,
            "dev_used_for_additional_selection": False,
            "primary_uncertainty_unit": "question paired within nine frozen cells",
            "bootstrap_repetitions": bootstrap_repetitions,
            "bootstrap_seed": bootstrap_seed,
            "configuration_bootstrap_repetitions": config_bootstrap_repetitions,
            "configuration_bootstrap_seed": config_bootstrap_seed,
            "same_config_degree_control": (
                "degree prior uses the exact selected graph family, seed, "
                "direction, and weight/restart setting"
            ),
        },
        "provenance": dict(provenance),
        "stage_e_reproduction": {
            "checks": reproduction_checks,
            "selected_baseline": selected_baseline_name,
            "selected_graph_method": selected_graph_config.name,
            "selected_graph_config": asdict(selected_graph_config),
            "selected_independent_degree_method": selected_degree_config.name,
            "selected_independent_degree_config": asdict(selected_degree_config),
        },
        "train": {
            "configuration_bootstrap_stability": config_stability,
            "leave_one_cell_out": leave_one_cell_out,
        },
        "dev": {
            "selected_graph": _compact_metrics(selected_dev_graph),
            "selected_baseline": _compact_metrics(selected_dev_baseline),
            "independent_degree_only": _compact_metrics(selected_dev_degree),
            "same_config_degree_only": {
                "config": asdict(selected_graph_config),
                "metrics": _compact_metrics(same_config_degree_metrics),
                "selected_graph_vs_same_config_degree": _macro_comparison(
                    selected_dev_graph, same_config_degree_metrics
                ),
            },
            "paired_stratified_bootstrap": {
                "selected_graph_vs_selected_baseline": bootstrap_vs_baseline,
                "selected_graph_vs_independent_degree_only": (
                    bootstrap_vs_independent_degree
                ),
                "selected_graph_vs_same_config_degree_only": (
                    bootstrap_vs_same_config_degree
                ),
            },
            "direction_sensitivity": direction_sensitivity,
            "heterogeneity_summary": {
                "graph_vs_baseline_mean_delta_by_cell": cell_deltas,
                "positive_cell_count": sum(
                    value > 0.0 for value in cell_deltas.values()
                ),
                "zero_cell_count": sum(value == 0.0 for value in cell_deltas.values()),
                "negative_cell_count": sum(
                    value < 0.0 for value in cell_deltas.values()
                ),
            },
        },
        "gate": {
            "name": "musique_post_primary_robustness_gate_v1",
            "checks": checks,
            "passed": passed,
            "status": (
                "retain_narrow_query_conditioned_alignment_claim"
                if passed
                else "downgrade_stage_e_gain_to_exploratory"
            ),
            "interpretation": (
                "This post-primary gate asks whether the nine-cell macro gain "
                "has a positive paired-bootstrap lower bound against the dense "
                "baseline and two degree-only controls. Configuration stability, "
                "leave-one-cell-out, and subgroup estimates are diagnostics, not "
                "additional model-selection gates."
            ),
        },
        "claim_boundary": {
            "authorized_if_gate_passes": (
                "Within MuSiQue's frozen 20-paragraph candidate pools, the "
                "train-selected occurrence-aware title graph provides a small "
                "query-conditioned retrieval-alignment gain on the predeclared "
                "nine-cell macro metric."
            ),
            "directed_topology_reasoning_authorized": False,
            "directed_topology_reasoning_reason": (
                "the primary train-selected propagation direction is undirected"
            ),
            "uniform_gain_across_hops_and_collision_strata_authorized": False,
            "full_corpus_graphrag_claim_authorized": False,
            "answer_generation_gain_claim_authorized": False,
            "topology_counterfactual_verifier_claim_authorized": False,
        },
        "content_contract": (
            "aggregate metrics, configurations, paths, and hashes only; no IDs, "
            "questions, answers, titles, paragraph text, decomposition text, or "
            "supporting labels"
        ),
    }
