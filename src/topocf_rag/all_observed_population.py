"""Aggregate-only population audit for locally grounded T3 rewires."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import statistics
from typing import Any

from .graph import HotpotInvariantError, validate_hotpot_example
from .hotpot import validate_hotpot_record
from .pairs import generate_question_pairs


class AllObservedPopulationInvariantError(ValueError):
    """Raised when a population audit violates its content-safe contract."""


def _graph_ineligibility_reason(error: HotpotInvariantError) -> str:
    """Map content-free graph validation failures to stable aggregate labels."""

    message = str(error)
    if message == "context normalized titles must be unique":
        return "duplicate_normalized_context_titles"
    if "references a missing context title" in message:
        return "supporting_title_missing_from_context"
    if "sentence_index is out of range" in message:
        return "supporting_fact_sentence_index_out_of_range"
    if message == "supporting_facts must not be empty":
        return "empty_supporting_facts"
    if message == "context must contain at least one document":
        return "empty_context"
    if message == "question must be a non-empty string":
        return "empty_question"
    return "other_graph_invariant_failure"


def audit_all_observed_population(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count all-observed T3 endpoint operations without retaining IDs/text."""

    record_count = 0
    bridge_count = 0
    graph_eligible_count = 0
    graph_ineligible_reasons: Counter[str] = Counter()
    question_count = 0
    pair_count = 0
    context_lengths: Counter[int] = Counter()
    pair_counts: Counter[int] = Counter()
    positive_nonobserved_edge_count = 0
    negative_nonobserved_edge_count = 0

    for record in records:
        # The source-level contract applies to every official record, while the
        # graph contract applies only after the frozen bridge filter.  Applying
        # the latter first incorrectly lets an out-of-scope comparison record
        # abort the bridge-only census.
        validate_hotpot_record(record)
        record_count += 1
        if record.get("type") != "bridge":
            continue
        bridge_count += 1
        try:
            validate_hotpot_example(record)
        except HotpotInvariantError as error:
            graph_ineligible_reasons[_graph_ineligibility_reason(error)] += 1
            continue
        graph_eligible_count += 1
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
    graph_ineligible_count = sum(graph_ineligible_reasons.values())
    if graph_eligible_count + graph_ineligible_count != bridge_count:
        raise RuntimeError("graph eligibility counts do not cover bridge questions")
    if sum(pair_counts.values()) != graph_eligible_count:
        raise RuntimeError(
            "pair-count histogram does not cover graph-eligible bridge questions"
        )
    nonzero_counts = [
        count
        for count, frequency in pair_counts.items()
        for _ in range(frequency)
        if count > 0
    ]
    return {
        "record_count": record_count,
        "bridge_question_count": bridge_count,
        "graph_eligible_bridge_question_count": graph_eligible_count,
        "graph_ineligible_bridge_question_count": graph_ineligible_count,
        "graph_ineligible_reason_histogram": dict(
            sorted(graph_ineligible_reasons.items())
        ),
        "all_observed_question_count": question_count,
        "all_observed_question_rate": (
            question_count / bridge_count if bridge_count else None
        ),
        "all_observed_question_rate_among_graph_eligible": (
            question_count / graph_eligible_count
            if graph_eligible_count
            else None
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
            "all_bridge_questions_accounted_for": (
                graph_eligible_count + graph_ineligible_count == bridge_count
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
