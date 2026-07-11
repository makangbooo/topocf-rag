import itertools

import pytest

from topocf_rag.metrics import (
    ScoredCandidate,
    ScoredPair,
    evaluate_ranking,
    expected_macro_mrr,
    expected_macro_recall_at_k,
    macro_auroc,
    macro_pairwise_accuracy,
    score_margin_report,
)


def candidate(question: str, name: str, positive: bool, score: float) -> ScoredCandidate:
    return ScoredCandidate(question, name, positive, score)


def test_pairwise_accuracy_is_question_macro_not_pair_micro() -> None:
    pairs = [ScoredPair("many", f"win-{index}", 1.0, 0.0) for index in range(10)]
    pairs.append(ScoredPair("one", "loss", 0.0, 1.0))
    report = macro_pairwise_accuracy(pairs)
    assert report.value == pytest.approx(0.5)
    assert report.eligible_question_count == 2
    assert report.item_count == 11


def test_pairwise_ties_and_missing_expected_questions_are_reported() -> None:
    pair = ScoredPair("q1", "pair", 0.25, 0.25)
    report = macro_pairwise_accuracy(
        [pair, pair], expected_question_ids=["q1", "q2"]
    )
    assert report.value == 0.5
    assert report.tie_count == 1
    assert report.duplicate_count == 1
    assert report.excluded_question_count == 1
    assert report.exclusions == (("no_pairs", 1),)


def test_score_margins_aggregate_within_question_before_cross_question() -> None:
    pairs = [
        ScoredPair("many", "zero-a", 0.0, 0.0),
        ScoredPair("many", "zero-b", 1.0, 1.0),
        ScoredPair("many", "large", 9.0, 0.0),
        ScoredPair("one", "negative", 0.0, 1.0),
    ]
    report = score_margin_report(pairs)

    # Per-question means are [3, -1], while per-question medians are [0, -1].
    # A micro mean over all four pairs would be 2 and must not be reported.
    assert report.mean_of_question_mean_margins == pytest.approx(1.0)
    assert report.median_of_question_mean_margins == pytest.approx(1.0)
    assert report.mean_of_question_median_margins == pytest.approx(-0.5)
    assert report.median_of_question_median_margins == pytest.approx(-0.5)
    assert report.eligible_question_count == 2
    assert report.pair_count == 4
    assert report.tie_count == 2


def test_score_margin_input_order_and_duplicate_pairs_do_not_change_values() -> None:
    pairs = [
        ScoredPair("q1", "a", 0.8, 0.2),
        ScoredPair("q1", "b", 0.1, 0.4),
        ScoredPair("q2", "a", 0.5, 0.5),
    ]
    expected = score_margin_report(pairs, expected_question_ids=["q1", "q2", "q3"])
    reordered = score_margin_report(
        list(reversed(pairs)), expected_question_ids=["q3", "q2", "q1"]
    )
    duplicated = score_margin_report(
        [*pairs, pairs[0]], expected_question_ids=["q1", "q2", "q3"]
    )

    assert reordered == expected
    assert duplicated.mean_of_question_mean_margins == expected.mean_of_question_mean_margins
    assert duplicated.median_of_question_mean_margins == expected.median_of_question_mean_margins
    assert duplicated.mean_of_question_median_margins == expected.mean_of_question_median_margins
    assert duplicated.median_of_question_median_margins == expected.median_of_question_median_margins
    assert duplicated.pair_count == expected.pair_count
    assert duplicated.tie_count == expected.tie_count
    assert duplicated.duplicate_count == 1
    assert expected.exclusions == (("no_pairs", 1),)


def test_question_macro_auroc_deduplicates_and_excludes_single_class() -> None:
    rows = [
        candidate("win", "p", True, 1.0),
        candidate("win", "n", False, 0.0),
        candidate("tie", "p", True, 0.5),
        candidate("tie", "n", False, 0.5),
        candidate("single", "p", True, 0.9),
    ]
    rows.append(rows[0])
    report = macro_auroc(rows)
    assert report.value == pytest.approx(0.75)
    assert report.eligible_question_count == 2
    assert report.excluded_question_count == 1
    assert report.exclusions == (("single_class", 1),)
    assert report.item_count == 2
    assert report.tie_count == 1
    assert report.duplicate_count == 1


def test_all_ties_use_uniform_expected_mrr_and_fractional_recall() -> None:
    rows = [
        candidate("q", "positive", True, 0.0),
        candidate("q", "negative", False, 0.0),
    ]
    # The positive is first or second with equal probability: E[RR] = .75.
    assert expected_macro_mrr(rows).value == pytest.approx(0.75)
    assert expected_macro_recall_at_k(rows, 1).value == pytest.approx(0.5)
    assert expected_macro_recall_at_k(rows, 3).value == pytest.approx(1.0)


def test_recall_fractionally_allocates_a_boundary_tie_group() -> None:
    rows = [
        candidate("q", "top-positive", True, 2.0),
        candidate("q", "tie-positive", True, 1.0),
        candidate("q", "tie-negative-a", False, 1.0),
        candidate("q", "tie-negative-b", False, 1.0),
        candidate("q", "bottom", False, 0.0),
    ]
    # At K=2 one of the three tied candidates is selected in expectation.
    # Expected positives retrieved = 1 + 1/3, divided by two positives.
    report = expected_macro_recall_at_k(rows, 2)
    assert report.value == pytest.approx(2 / 3)
    assert report.tie_count == 1


def test_candidate_input_order_and_exact_duplicates_do_not_change_metrics() -> None:
    rows = [
        candidate("q1", "p", True, 0.7),
        candidate("q1", "n", False, 0.2),
        candidate("q2", "p", True, 0.1),
        candidate("q2", "n", False, 0.4),
    ]
    expected = evaluate_ranking(rows)
    for permutation in itertools.permutations(rows):
        assert evaluate_ranking(permutation) == expected

    duplicated = evaluate_ranking([*rows, rows[0]])
    assert duplicated.auroc.value == expected.auroc.value
    assert duplicated.expected_mrr.value == expected.expected_mrr.value
    assert duplicated.auroc.duplicate_count == 1


def test_conflicting_duplicate_candidate_is_rejected() -> None:
    rows = [
        candidate("q", "same", True, 1.0),
        candidate("q", "same", False, 1.0),
    ]
    with pytest.raises(ValueError, match="conflicting"):
        macro_auroc(rows)


def test_ranking_bundle_reports_recall_at_1_3_5_and_exclusions() -> None:
    rows = [
        candidate("q1", "p", True, 1.0),
        candidate("q1", "n", False, 0.0),
        candidate("q2", "n", False, 0.0),
    ]
    report = evaluate_ranking(rows, expected_question_ids=["q1", "q2", "q3"])
    assert report.auroc.eligible_question_count == 1
    assert report.auroc.excluded_question_count == 2
    assert report.expected_mrr.exclusions == (
        ("no_candidates", 1),
        ("no_positive", 1),
    )
    assert report.expected_recall_at_1.value == 1.0
    assert report.expected_recall_at_3.value == 1.0
    assert report.expected_recall_at_5.value == 1.0
