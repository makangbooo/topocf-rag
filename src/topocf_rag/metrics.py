"""Question-macro ranking metrics with explicit, label-neutral tie handling."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real


def _validate_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_score(value: float, field_name: str) -> None:
    # NumPy floating/integer scalars implement Real and are common outputs of
    # embedding scorers.  Accept them while continuing to reject booleans.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """One uniquely identified candidate in a question-local ranking."""

    question_id: str
    candidate_id: str
    is_positive: bool
    score: float

    def __post_init__(self) -> None:
        _validate_identifier(self.question_id, "question_id")
        _validate_identifier(self.candidate_id, "candidate_id")
        if not isinstance(self.is_positive, bool):
            raise TypeError("is_positive must be bool")
        _validate_score(self.score, "score")
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True, slots=True)
class ScoredPair:
    """One explicit positive/negative comparison for pairwise accuracy."""

    question_id: str
    pair_id: str
    positive_score: float
    negative_score: float

    def __post_init__(self) -> None:
        _validate_identifier(self.question_id, "question_id")
        _validate_identifier(self.pair_id, "pair_id")
        _validate_score(self.positive_score, "positive_score")
        _validate_score(self.negative_score, "negative_score")
        object.__setattr__(self, "positive_score", float(self.positive_score))
        object.__setattr__(self, "negative_score", float(self.negative_score))


@dataclass(frozen=True, slots=True)
class MetricReport:
    """A macro metric together with auditable denominators.

    ``item_count`` is the number of unique pairs for pairwise accuracy, unique
    candidates for ranking metrics, or positive/negative comparisons for
    AUROC.  ``tie_count`` is pair ties, AUROC comparison ties, or tied score
    groups respectively.
    """

    value: float | None
    eligible_question_count: int
    excluded_question_count: int
    item_count: int
    tie_count: int
    duplicate_count: int
    exclusions: tuple[tuple[str, int], ...]

    @property
    def total_question_count(self) -> int:
        return self.eligible_question_count + self.excluded_question_count


@dataclass(frozen=True, slots=True)
class RankingReport:
    auroc: MetricReport
    expected_mrr: MetricReport
    expected_recall_at_1: MetricReport
    expected_recall_at_3: MetricReport
    expected_recall_at_5: MetricReport


@dataclass(frozen=True, slots=True)
class ScoreMarginReport:
    """Question-macro summaries of ``positive_score - negative_score``.

    Every question first contributes one mean and one median over its unique
    pairs.  The four metric fields then summarize those per-question values;
    no question receives extra weight merely because it has more pairs.
    """

    mean_of_question_mean_margins: float | None
    median_of_question_mean_margins: float | None
    mean_of_question_median_margins: float | None
    median_of_question_median_margins: float | None
    eligible_question_count: int
    excluded_question_count: int
    pair_count: int
    tie_count: int
    duplicate_count: int
    exclusions: tuple[tuple[str, int], ...]

    @property
    def total_question_count(self) -> int:
        return self.eligible_question_count + self.excluded_question_count


def _question_universe(
    observed_question_ids: set[str],
    expected_question_ids: Iterable[str] | None,
) -> tuple[str, ...]:
    if expected_question_ids is None:
        return tuple(sorted(observed_question_ids))
    expected = tuple(expected_question_ids)
    if any(not isinstance(question_id, str) or not question_id for question_id in expected):
        raise ValueError("expected question IDs must be non-empty strings")
    if len(expected) != len(set(expected)):
        raise ValueError("expected question IDs must be unique")
    unexpected = observed_question_ids.difference(expected)
    if unexpected:
        raise ValueError("observations contain a question outside the expected universe")
    return tuple(sorted(expected))


def _exclusions(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((reason, count) for reason, count in counter.items() if count))


def _deduplicate_pairs(
    pairs: Iterable[ScoredPair],
) -> tuple[dict[str, list[ScoredPair]], int]:
    unique: dict[tuple[str, str], ScoredPair] = {}
    duplicate_count = 0
    for pair in pairs:
        if not isinstance(pair, ScoredPair):
            raise TypeError("pairs must contain ScoredPair values")
        key = (pair.question_id, pair.pair_id)
        previous = unique.get(key)
        if previous is None:
            unique[key] = pair
        elif previous == pair:
            duplicate_count += 1
        else:
            raise ValueError("a duplicate pair ID has conflicting scores")

    grouped: dict[str, list[ScoredPair]] = defaultdict(list)
    for pair in unique.values():
        grouped[pair.question_id].append(pair)
    for question_pairs in grouped.values():
        question_pairs.sort(key=lambda pair: pair.pair_id)
    return dict(grouped), duplicate_count


def _deduplicate_candidates(
    candidates: Iterable[ScoredCandidate],
) -> tuple[dict[str, list[ScoredCandidate]], int]:
    unique: dict[tuple[str, str], ScoredCandidate] = {}
    duplicate_count = 0
    for candidate in candidates:
        if not isinstance(candidate, ScoredCandidate):
            raise TypeError("candidates must contain ScoredCandidate values")
        key = (candidate.question_id, candidate.candidate_id)
        previous = unique.get(key)
        if previous is None:
            unique[key] = candidate
        elif previous == candidate:
            duplicate_count += 1
        else:
            raise ValueError("a duplicate candidate ID has a conflicting label or score")

    grouped: dict[str, list[ScoredCandidate]] = defaultdict(list)
    for candidate in unique.values():
        grouped[candidate.question_id].append(candidate)
    for question_candidates in grouped.values():
        question_candidates.sort(key=lambda candidate: candidate.candidate_id)
    return dict(grouped), duplicate_count


def macro_pairwise_accuracy(
    pairs: Iterable[ScoredPair],
    *,
    expected_question_ids: Iterable[str] | None = None,
) -> MetricReport:
    """Compute win/tie/loss accuracy within question and then macro-average."""

    grouped, duplicate_count = _deduplicate_pairs(pairs)
    questions = _question_universe(set(grouped), expected_question_ids)
    question_values: list[float] = []
    exclusions: Counter[str] = Counter()
    tie_count = 0
    pair_count = 0

    for question_id in questions:
        question_pairs = grouped.get(question_id, [])
        if not question_pairs:
            exclusions["no_pairs"] += 1
            continue
        outcomes: list[float] = []
        for pair in question_pairs:
            pair_count += 1
            if pair.positive_score > pair.negative_score:
                outcomes.append(1.0)
            elif pair.positive_score == pair.negative_score:
                outcomes.append(0.5)
                tie_count += 1
            else:
                outcomes.append(0.0)
        question_values.append(sum(outcomes) / len(outcomes))

    return MetricReport(
        value=(sum(question_values) / len(question_values) if question_values else None),
        eligible_question_count=len(question_values),
        excluded_question_count=sum(exclusions.values()),
        item_count=pair_count,
        tie_count=tie_count,
        duplicate_count=duplicate_count,
        exclusions=_exclusions(exclusions),
    )


def score_margin_report(
    pairs: Iterable[ScoredPair],
    *,
    expected_question_ids: Iterable[str] | None = None,
) -> ScoreMarginReport:
    """Summarize score margins after aggregating unique pairs per question."""

    grouped, duplicate_count = _deduplicate_pairs(pairs)
    questions = _question_universe(set(grouped), expected_question_ids)
    question_mean_margins: list[float] = []
    question_median_margins: list[float] = []
    exclusions: Counter[str] = Counter()
    pair_count = 0
    tie_count = 0

    for question_id in questions:
        question_pairs = grouped.get(question_id, [])
        if not question_pairs:
            exclusions["no_pairs"] += 1
            continue

        margins: list[float] = []
        for pair in question_pairs:
            margin = pair.positive_score - pair.negative_score
            if not math.isfinite(margin):
                raise ValueError("positive/negative score difference must be finite")
            margins.append(margin)
            pair_count += 1
            tie_count += int(margin == 0.0)
        question_mean_margins.append(statistics.fmean(margins))
        question_median_margins.append(float(statistics.median(margins)))

    if question_mean_margins:
        mean_of_means = statistics.fmean(question_mean_margins)
        median_of_means = float(statistics.median(question_mean_margins))
        mean_of_medians = statistics.fmean(question_median_margins)
        median_of_medians = float(statistics.median(question_median_margins))
    else:
        mean_of_means = None
        median_of_means = None
        mean_of_medians = None
        median_of_medians = None

    return ScoreMarginReport(
        mean_of_question_mean_margins=mean_of_means,
        median_of_question_mean_margins=median_of_means,
        mean_of_question_median_margins=mean_of_medians,
        median_of_question_median_margins=median_of_medians,
        eligible_question_count=len(question_mean_margins),
        excluded_question_count=sum(exclusions.values()),
        pair_count=pair_count,
        tie_count=tie_count,
        duplicate_count=duplicate_count,
        exclusions=_exclusions(exclusions),
    )


def macro_auroc(
    candidates: Iterable[ScoredCandidate],
    *,
    expected_question_ids: Iterable[str] | None = None,
) -> MetricReport:
    """Compute AUROC independently within each eligible question.

    The implementation is the positive/negative concordance definition, so a
    tied comparison contributes exactly 0.5 and raw scores are never pooled
    across questions.
    """

    grouped, duplicate_count = _deduplicate_candidates(candidates)
    questions = _question_universe(set(grouped), expected_question_ids)
    question_values: list[float] = []
    exclusions: Counter[str] = Counter()
    comparison_count = 0
    tie_count = 0

    for question_id in questions:
        question_candidates = grouped.get(question_id, [])
        if not question_candidates:
            exclusions["no_candidates"] += 1
            continue
        positives = [candidate.score for candidate in question_candidates if candidate.is_positive]
        negatives = [candidate.score for candidate in question_candidates if not candidate.is_positive]
        if not positives or not negatives:
            exclusions["single_class"] += 1
            continue

        concordance = 0.0
        for positive_score in positives:
            for negative_score in negatives:
                comparison_count += 1
                if positive_score > negative_score:
                    concordance += 1.0
                elif positive_score == negative_score:
                    concordance += 0.5
                    tie_count += 1
        question_values.append(concordance / (len(positives) * len(negatives)))

    return MetricReport(
        value=(sum(question_values) / len(question_values) if question_values else None),
        eligible_question_count=len(question_values),
        excluded_question_count=sum(exclusions.values()),
        item_count=comparison_count,
        tie_count=tie_count,
        duplicate_count=duplicate_count,
        exclusions=_exclusions(exclusions),
    )


def _score_groups(
    candidates: Sequence[ScoredCandidate],
) -> list[tuple[float, int, int]]:
    """Return descending ``(score, group_size, positive_count)`` groups."""

    counts: dict[float, list[int]] = {}
    for candidate in candidates:
        group = counts.setdefault(candidate.score, [0, 0])
        group[0] += 1
        group[1] += int(candidate.is_positive)
    return [
        (score, counts[score][0], counts[score][1])
        for score in sorted(counts, reverse=True)
    ]


def _expected_reciprocal_rank(
    groups: Sequence[tuple[float, int, int]],
) -> float:
    candidates_before = 0
    for _score, group_size, positive_count in groups:
        if positive_count == 0:
            candidates_before += group_size
            continue

        # Under a uniformly random order within this tie group, the first
        # positive is at local position j with this negative-hypergeometric
        # probability.  Summing analytically avoids arbitrary ID tie-breaks.
        denominator = math.comb(group_size, positive_count)
        expectation = 0.0
        latest_first_position = group_size - positive_count + 1
        for local_position in range(1, latest_first_position + 1):
            arrangements = math.comb(
                group_size - local_position, positive_count - 1
            )
            probability = arrangements / denominator
            expectation += probability / (candidates_before + local_position)
        return expectation
    raise ValueError("expected reciprocal rank requires a positive candidate")


def expected_macro_mrr(
    candidates: Iterable[ScoredCandidate],
    *,
    expected_question_ids: Iterable[str] | None = None,
) -> MetricReport:
    """Macro MRR, analytically averaged over uniform permutations of ties."""

    grouped, duplicate_count = _deduplicate_candidates(candidates)
    questions = _question_universe(set(grouped), expected_question_ids)
    question_values: list[float] = []
    exclusions: Counter[str] = Counter()
    candidate_count = 0
    tie_group_count = 0

    for question_id in questions:
        question_candidates = grouped.get(question_id, [])
        if not question_candidates:
            exclusions["no_candidates"] += 1
            continue
        if not any(candidate.is_positive for candidate in question_candidates):
            exclusions["no_positive"] += 1
            continue
        groups = _score_groups(question_candidates)
        candidate_count += len(question_candidates)
        tie_group_count += sum(group_size > 1 for _score, group_size, _positive in groups)
        question_values.append(_expected_reciprocal_rank(groups))

    return MetricReport(
        value=(sum(question_values) / len(question_values) if question_values else None),
        eligible_question_count=len(question_values),
        excluded_question_count=sum(exclusions.values()),
        item_count=candidate_count,
        tie_count=tie_group_count,
        duplicate_count=duplicate_count,
        exclusions=_exclusions(exclusions),
    )


def expected_macro_recall_at_k(
    candidates: Iterable[ScoredCandidate],
    k: int,
    *,
    expected_question_ids: Iterable[str] | None = None,
) -> MetricReport:
    """Macro Recall@K with fractional expectation at a boundary tie group."""

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    grouped, duplicate_count = _deduplicate_candidates(candidates)
    questions = _question_universe(set(grouped), expected_question_ids)
    question_values: list[float] = []
    exclusions: Counter[str] = Counter()
    candidate_count = 0
    tie_group_count = 0

    for question_id in questions:
        question_candidates = grouped.get(question_id, [])
        if not question_candidates:
            exclusions["no_candidates"] += 1
            continue
        total_positives = sum(candidate.is_positive for candidate in question_candidates)
        if total_positives == 0:
            exclusions["no_positive"] += 1
            continue

        groups = _score_groups(question_candidates)
        candidate_count += len(question_candidates)
        tie_group_count += sum(group_size > 1 for _score, group_size, _positive in groups)
        remaining_slots = min(k, len(question_candidates))
        expected_retrieved_positives = 0.0
        for _score, group_size, positive_count in groups:
            if remaining_slots <= 0:
                break
            if group_size <= remaining_slots:
                expected_retrieved_positives += positive_count
                remaining_slots -= group_size
            else:
                expected_retrieved_positives += (
                    positive_count * remaining_slots / group_size
                )
                remaining_slots = 0
        question_values.append(expected_retrieved_positives / total_positives)

    return MetricReport(
        value=(sum(question_values) / len(question_values) if question_values else None),
        eligible_question_count=len(question_values),
        excluded_question_count=sum(exclusions.values()),
        item_count=candidate_count,
        tie_count=tie_group_count,
        duplicate_count=duplicate_count,
        exclusions=_exclusions(exclusions),
    )


def evaluate_ranking(
    candidates: Iterable[ScoredCandidate],
    *,
    expected_question_ids: Iterable[str] | None = None,
) -> RankingReport:
    """Evaluate the preregistered AUROC, MRR, and Recall@1/3/5 bundle."""

    materialized = tuple(candidates)
    expected = tuple(expected_question_ids) if expected_question_ids is not None else None
    return RankingReport(
        auroc=macro_auroc(materialized, expected_question_ids=expected),
        expected_mrr=expected_macro_mrr(materialized, expected_question_ids=expected),
        expected_recall_at_1=expected_macro_recall_at_k(
            materialized, 1, expected_question_ids=expected
        ),
        expected_recall_at_3=expected_macro_recall_at_k(
            materialized, 3, expected_question_ids=expected
        ),
        expected_recall_at_5=expected_macro_recall_at_k(
            materialized, 5, expected_question_ids=expected
        ),
    )
