"""Aggregate-only population audit for locally grounded T3 rewires."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import statistics
from typing import Any

from .graph import validate_hotpot_example
from .pairs import generate_question_pairs


class AllObservedPopulationInvariantError(ValueError):
    """Raised when a population audit violates its content-safe contract."""


def audit_all_observed_population(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count all-observed T3 endpoint operations without retaining IDs/text."""

    record_count = 0
    bridge_count = 0
    question_count = 0
    pair_count = 0
    context_lengths: Counter[int] = Counter()
    pair_counts: Counter[int] = Counter()
    positive_nonobserved_edge_count = 0
    negative_nonobserved_edge_count = 0

    for record in records:
        validate_hotpot_example(record)
        record_count += 1
        if record.get("type") != "bridge":
            continue
        bridge_count += 1
        context = record.get("context")
        if not isinstance(context, list) or not context:
            raise AllObservedPopulationInvariantError(
                "validated context must be a non-empty list"
            )
        context_lengths[len(context)] += 1
        scores = [0.0] * len(context)
        pairs = tuple(
            pair
            for pair in generate_question_pairs(
                record,
                scores,
                retrieval_top_k=len(context),
            )
            if pair.stratum == "all_observed_rewire"
            and pair.variant == "t3_all_observed"
        )
        pair_counts[len(pairs)] += 1
        if pairs:
            question_count += 1
            pair_count += len(pairs)
        for pair in pairs:
            positive_nonobserved_edge_count += sum(
                not edge.observed for edge in pair.positive_edges
            )
            negative_nonobserved_edge_count += sum(
                not edge.observed for edge in pair.negative_edges
            )

    if record_count < 1:
        raise AllObservedPopulationInvariantError("source contains no records")
    if sum(pair_counts.values()) != bridge_count:
        raise RuntimeError("pair-count histogram does not cover bridge questions")
    nonzero_counts = [
        count
        for count, frequency in pair_counts.items()
        for _ in range(frequency)
        if count > 0
    ]
    return {
        "record_count": record_count,
        "bridge_question_count": bridge_count,
        "all_observed_question_count": question_count,
        "all_observed_question_rate": (
            question_count / bridge_count if bridge_count else None
        ),
        "all_observed_pair_count": pair_count,
        "all_observed_pairs_per_eligible_question": {
            "mean": statistics.fmean(nonzero_counts) if nonzero_counts else None,
            "median": statistics.median(nonzero_counts) if nonzero_counts else None,
            "minimum": min(nonzero_counts) if nonzero_counts else None,
            "maximum": max(nonzero_counts) if nonzero_counts else None,
        },
        "all_observed_pair_count_per_bridge_question_histogram": {
            str(count): frequency for count, frequency in sorted(pair_counts.items())
        },
        "context_length_histogram": {
            str(count): frequency
            for count, frequency in sorted(context_lengths.items())
        },
        "integrity": {
            "positive_nonobserved_edge_count": positive_nonobserved_edge_count,
            "negative_nonobserved_edge_count": negative_nonobserved_edge_count,
            "all_candidate_edges_observed": (
                positive_nonobserved_edge_count == 0
                and negative_nonobserved_edge_count == 0
            ),
        },
    }


def population_gate(
    splits: Mapping[str, Mapping[str, Any]],
    *,
    minimum_train_question_count: int,
    minimum_dev_question_count: int,
) -> dict[str, Any]:
    """Apply the frozen population-only feasibility threshold."""

    if minimum_train_question_count < 1 or minimum_dev_question_count < 1:
        raise AllObservedPopulationInvariantError(
            "population thresholds must be positive"
        )
    train = splits.get("train")
    dev = splits.get("dev_distractor")
    if not isinstance(train, Mapping) or not isinstance(dev, Mapping):
        raise AllObservedPopulationInvariantError(
            "train and dev_distractor summaries are required"
        )
    checks = {
        "train_question_count": {
            "minimum": minimum_train_question_count,
            "value": train.get("all_observed_question_count"),
            "passed": (
                isinstance(train.get("all_observed_question_count"), int)
                and train["all_observed_question_count"]
                >= minimum_train_question_count
            ),
        },
        "dev_question_count": {
            "minimum": minimum_dev_question_count,
            "value": dev.get("all_observed_question_count"),
            "passed": (
                isinstance(dev.get("all_observed_question_count"), int)
                and dev["all_observed_question_count"]
                >= minimum_dev_question_count
            ),
        },
        "all_candidate_edges_observed": {
            "passed": (
                train.get("integrity", {}).get("all_candidate_edges_observed")
                is True
                and dev.get("integrity", {}).get("all_candidate_edges_observed")
                is True
            )
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "checks": checks,
        "passed": passed,
        "status": (
            "authorize_split_design_only"
            if passed
            else "stop_and_change_dataset_or_graph_construction"
        ),
    }
