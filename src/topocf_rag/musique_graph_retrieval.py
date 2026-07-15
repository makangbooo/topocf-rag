"""Occurrence-aware graph retrieval and shortcut controls for MuSiQue."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import math
import statistics
from typing import Any

from .baselines import BM25Config, PerQuestionBM25
from .graph_retrieval import (
    GraphMethodConfig,
    RetrievalQuestion,
    baseline_rankings,
    degree_prior_rankings,
    graph_candidate_grid,
    graph_rankings,
)
from .graph_retrieval_validation import permute_question_graph
from .musique_duplicates import diagnose_duplicate_titles
from .musique_splits import (
    HOP_COUNTS,
    TITLE_COLLISION_STRATA,
    TRAIN_PER_CELL,
    all_cells,
    cell_name,
)
from .title_normalization import contains_title_mention


MUSIQUE_GRAPH_REPORT_SCHEMA_VERSION = 1
EXPECTED_STAGE_D_REPORT_SHA256 = (
    "3aa6a337b300c61aedff799e62f33430f70673c836617340074034c6584cb913"
)
EXTRA_BUDGETS = (0, 1, 3)
PERMUTATION_REPETITIONS = 200
PERMUTATION_SEED = 20260715
STRONG_BASELINE_MEAN_DELTA_THRESHOLD = 0.02
DEGREE_CONTROL_MEAN_DELTA_THRESHOLD = 0.02
MAX_PER_BUDGET_REGRESSION = 0.01
PERMUTATION_P_THRESHOLD = 0.01
MCNEMAR_P_THRESHOLD = 0.01


class MuSiQueGraphRetrievalError(ValueError):
    """Raised when a MuSiQue graph experiment invariant is violated."""


def _validate_dense_scores(scores: Sequence[Any]) -> tuple[float, ...]:
    if len(scores) != 20:
        raise MuSiQueGraphRetrievalError(
            "dense score count must equal the frozen paragraph count"
        )
    values: list[float] = []
    for score in scores:
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise MuSiQueGraphRetrievalError("dense scores must be numeric")
        value = float(score)
        if not math.isfinite(value):
            raise MuSiQueGraphRetrievalError("dense scores must be finite")
        values.append(value)
    return tuple(values)


def build_occurrence_mention_edges(
    paragraphs: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, int], ...]:
    """Fan each literal title mention out to every matching occurrence."""

    if len(paragraphs) != 20:
        raise MuSiQueGraphRetrievalError(
            "occurrence graph requires exactly 20 paragraphs"
        )
    edges: set[tuple[int, int]] = set()
    for source in paragraphs:
        source_index = source.get("idx")
        source_text = source.get("paragraph_text")
        if not isinstance(source_index, int) or isinstance(source_index, bool):
            raise MuSiQueGraphRetrievalError("paragraph idx must be an integer")
        if not isinstance(source_text, str):
            raise MuSiQueGraphRetrievalError("paragraph text must be a string")
        for target in paragraphs:
            target_index = target.get("idx")
            target_title = target.get("title")
            if source_index == target_index:
                continue
            if not isinstance(target_index, int) or isinstance(
                target_index, bool
            ):
                raise MuSiQueGraphRetrievalError(
                    "paragraph idx must be an integer"
                )
            if not isinstance(target_title, str):
                raise MuSiQueGraphRetrievalError(
                    "paragraph title must be a string"
                )
            if contains_title_mention(source_text, target_title):
                edges.add((source_index, target_index))
    return tuple(sorted(edges))


def prepare_musique_retrieval_question(
    example: Mapping[str, Any], cache_record: Mapping[str, Any]
) -> RetrievalQuestion:
    """Build label-free scores and graph, attaching gold only for metrics."""

    diagnostic = diagnose_duplicate_titles(example)
    if (
        not diagnostic.occurrence_structurally_eligible
        or diagnostic.exact_text_mixed_support_label_group_count
        or diagnostic.paragraph_count != 20
        or diagnostic.hop_count not in HOP_COUNTS
        or diagnostic.status not in TITLE_COLLISION_STRATA
    ):
        raise MuSiQueGraphRetrievalError(
            "question does not satisfy the frozen occurrence-aware contract"
        )
    qid = example.get("id")
    question_text = example.get("question")
    paragraphs = example.get("paragraphs")
    if not isinstance(qid, str) or not qid:
        raise MuSiQueGraphRetrievalError("question ID must be non-empty")
    if not isinstance(question_text, str) or not question_text.strip():
        raise MuSiQueGraphRetrievalError("question text must be non-empty")
    if not isinstance(paragraphs, list):
        raise MuSiQueGraphRetrievalError("paragraphs must be a list")
    expected_indices = list(range(20))
    if cache_record.get("paragraph_indices") != expected_indices:
        raise MuSiQueGraphRetrievalError(
            "cache paragraph indices do not match official occurrences"
        )
    scores = cache_record.get("scores")
    if not isinstance(scores, list):
        raise MuSiQueGraphRetrievalError("cache scores are invalid")
    dense = _validate_dense_scores(scores)

    serialized = {
        str(index): (
            f"{paragraph['title']}\n{paragraph['paragraph_text']}"
        )
        for index, paragraph in enumerate(paragraphs)
    }
    bm25_index = PerQuestionBM25.fit(serialized, config=BM25Config())
    bm25_by_id = bm25_index.score(question_text)
    bm25 = tuple(bm25_by_id[str(index)] for index in range(20))

    decomposition = example.get("question_decomposition")
    assert isinstance(decomposition, list)
    gold_indices = tuple(
        int(step["paragraph_support_idx"]) for step in decomposition
    )
    if len(gold_indices) != diagnostic.hop_count:
        raise MuSiQueGraphRetrievalError("gold count differs from hop count")
    if len(gold_indices) != len(set(gold_indices)):
        raise MuSiQueGraphRetrievalError("gold paragraph indices are not unique")
    return RetrievalQuestion(
        qid=qid,
        dense_scores=dense,
        bm25_scores=bm25,
        gold_indices=gold_indices,
        mention_edges=build_occurrence_mention_edges(paragraphs),
        question_type=cell_name(diagnostic.hop_count, diagnostic.status),
    )


def _validate_rankings(
    questions: Sequence[RetrievalQuestion], rankings: Sequence[Sequence[int]]
) -> tuple[tuple[int, ...], ...]:
    if not questions or len(questions) != len(rankings):
        raise MuSiQueGraphRetrievalError(
            "questions and rankings must be non-empty and equally sized"
        )
    normalized: list[tuple[int, ...]] = []
    for question, ranking_value in zip(questions, rankings, strict=True):
        ranking = tuple(ranking_value)
        if set(ranking) != set(range(question.document_count)):
            raise MuSiQueGraphRetrievalError(
                "ranking must be a full paragraph-index permutation"
            )
        if len(question.gold_indices) not in HOP_COUNTS:
            raise MuSiQueGraphRetrievalError("gold count is not a frozen hop")
        normalized.append(ranking)
    return tuple(normalized)


def _cell_metrics(
    questions: Sequence[RetrievalQuestion],
    rankings: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if not questions or len(questions) != len(rankings):
        raise MuSiQueGraphRetrievalError("cell inputs are invalid")
    gold_counts = {len(question.gold_indices) for question in questions}
    if len(gold_counts) != 1:
        raise MuSiQueGraphRetrievalError("one cell must have one gold count")
    gold_count = next(iter(gold_counts))
    complete = {extra: 0 for extra in EXTRA_BUDGETS}
    support = {extra: 0 for extra in EXTRA_BUDGETS}
    no_gold = {extra: 0 for extra in EXTRA_BUDGETS}
    reciprocal_full: list[float] = []
    for question, ranking in zip(questions, rankings, strict=True):
        rank_by_index = {
            index: rank for rank, index in enumerate(ranking, start=1)
        }
        gold = set(question.gold_indices)
        reciprocal_full.append(
            1.0 / max(rank_by_index[index] for index in gold)
        )
        for extra in EXTRA_BUDGETS:
            retrieved = set(ranking[: gold_count + extra])
            found = len(gold.intersection(retrieved))
            complete[extra] += found == gold_count
            support[extra] += found
            no_gold[extra] += found == 0
    count = len(questions)
    by_budget = {
        str(extra): {
            "extra_budget": extra,
            "top_k": gold_count + extra,
            "complete_gold_evidence_count": complete[extra],
            "complete_gold_evidence_rate": complete[extra] / count,
            "support_document_recall": support[extra] / (count * gold_count),
            "no_gold_count": no_gold[extra],
            "no_gold_rate": no_gold[extra] / count,
        }
        for extra in EXTRA_BUDGETS
    }
    return {
        "question_count": count,
        "gold_document_count": gold_count,
        "by_extra_budget": by_budget,
        "mean_complete_gold_evidence_rate": sum(
            row["complete_gold_evidence_rate"] for row in by_budget.values()
        )
        / len(EXTRA_BUDGETS),
        "full_evidence_mrr": statistics.fmean(reciprocal_full),
    }


def _macro_rows(
    by_cell: Mapping[str, Mapping[str, Any]], cells: Sequence[str]
) -> dict[str, Any]:
    rows = [by_cell[cell] for cell in cells]
    return {
        "cell_count": len(rows),
        "by_extra_budget": {
            str(extra): {
                "extra_budget": extra,
                "macro_complete_gold_evidence_rate": statistics.fmean(
                    row["by_extra_budget"][str(extra)][
                        "complete_gold_evidence_rate"
                    ]
                    for row in rows
                ),
                "macro_support_document_recall": statistics.fmean(
                    row["by_extra_budget"][str(extra)][
                        "support_document_recall"
                    ]
                    for row in rows
                ),
            }
            for extra in EXTRA_BUDGETS
        },
        "mean_complete_gold_evidence_rate": statistics.fmean(
            row["mean_complete_gold_evidence_rate"] for row in rows
        ),
        "full_evidence_mrr": statistics.fmean(
            row["full_evidence_mrr"] for row in rows
        ),
    }


def evaluate_musique_rankings(
    questions: Sequence[RetrievalQuestion], rankings: Sequence[Sequence[int]]
) -> dict[str, Any]:
    """Return nine-cell macro metrics plus hop/collision and micro views."""

    normalized = _validate_rankings(questions, rankings)
    grouped_questions: defaultdict[str, list[RetrievalQuestion]] = defaultdict(
        list
    )
    grouped_rankings: defaultdict[str, list[tuple[int, ...]]] = defaultdict(list)
    for question, ranking in zip(questions, normalized, strict=True):
        if question.question_type not in all_cells():
            raise MuSiQueGraphRetrievalError("question cell is invalid")
        grouped_questions[question.question_type].append(question)
        grouped_rankings[question.question_type].append(ranking)
    if set(grouped_questions) != set(all_cells()):
        raise MuSiQueGraphRetrievalError("evaluation must contain all nine cells")
    by_cell = {
        cell: _cell_metrics(grouped_questions[cell], grouped_rankings[cell])
        for cell in all_cells()
    }

    micro_complete = {extra: 0 for extra in EXTRA_BUDGETS}
    micro_support = {extra: 0 for extra in EXTRA_BUDGETS}
    total_gold = 0
    micro_full: list[float] = []
    for question, ranking in zip(questions, normalized, strict=True):
        gold = set(question.gold_indices)
        total_gold += len(gold)
        rank_by_index = {
            index: rank for rank, index in enumerate(ranking, start=1)
        }
        micro_full.append(1.0 / max(rank_by_index[index] for index in gold))
        for extra in EXTRA_BUDGETS:
            retrieved = set(ranking[: len(gold) + extra])
            found = len(gold.intersection(retrieved))
            micro_complete[extra] += found == len(gold)
            micro_support[extra] += found
    count = len(questions)
    return {
        "question_count": count,
        "question_count_by_cell": {
            cell: len(grouped_questions[cell]) for cell in all_cells()
        },
        "extra_budgets": list(EXTRA_BUDGETS),
        "by_cell": by_cell,
        "macro": _macro_rows(by_cell, all_cells()),
        "by_hop": {
            str(hop): _macro_rows(
                by_cell,
                [cell_name(hop, status) for status in TITLE_COLLISION_STRATA],
            )
            for hop in HOP_COUNTS
        },
        "by_title_collision": {
            status: _macro_rows(
                by_cell,
                [cell_name(hop, status) for hop in HOP_COUNTS],
            )
            for status in TITLE_COLLISION_STRATA
        },
        "micro": {
            "by_extra_budget": {
                str(extra): {
                    "complete_gold_evidence_rate": (
                        micro_complete[extra] / count
                    ),
                    "support_document_recall": (
                        micro_support[extra] / total_gold
                    ),
                }
                for extra in EXTRA_BUDGETS
            },
            "mean_complete_gold_evidence_rate": statistics.fmean(
                micro_complete[extra] / count for extra in EXTRA_BUDGETS
            ),
            "full_evidence_mrr": statistics.fmean(micro_full),
        },
    }


def evaluate_baseline(
    questions: Sequence[RetrievalQuestion], method: str
) -> dict[str, Any]:
    return evaluate_musique_rankings(
        questions, baseline_rankings(questions, method)
    )


def evaluate_graph(
    questions: Sequence[RetrievalQuestion], config: GraphMethodConfig
) -> dict[str, Any]:
    return evaluate_musique_rankings(questions, graph_rankings(questions, config))


def evaluate_degree_prior(
    questions: Sequence[RetrievalQuestion], config: GraphMethodConfig
) -> dict[str, Any]:
    return evaluate_musique_rankings(
        questions, degree_prior_rankings(questions, config)
    )


def _selection_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    macro = metrics["macro"]
    return (
        float(macro["mean_complete_gold_evidence_rate"]),
        *(
            float(
                macro["by_extra_budget"][str(extra)][
                    "macro_complete_gold_evidence_rate"
                ]
            )
            for extra in reversed(EXTRA_BUDGETS)
        ),
        float(macro["full_evidence_mrr"]),
    )


def select_best_result(
    named_results: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[str, Mapping[str, Any]]:
    if not named_results:
        raise MuSiQueGraphRetrievalError("named results must not be empty")
    best_name, best_metrics = named_results[0]
    best_key = _selection_key(best_metrics)
    for name, metrics in named_results[1:]:
        key = _selection_key(metrics)
        if key > best_key:
            best_name, best_metrics, best_key = name, metrics, key
    return best_name, best_metrics


def _comparison(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    macro_delta = (
        candidate["macro"]["mean_complete_gold_evidence_rate"]
        - baseline["macro"]["mean_complete_gold_evidence_rate"]
    )
    return {
        "absolute_macro_mean_complete_delta": macro_delta,
        "absolute_macro_complete_delta_by_extra_budget": {
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
        "absolute_mean_complete_delta_by_cell": {
            cell: (
                candidate["by_cell"][cell][
                    "mean_complete_gold_evidence_rate"
                ]
                - baseline["by_cell"][cell][
                    "mean_complete_gold_evidence_rate"
                ]
            )
            for cell in all_cells()
        },
    }


def _exact_mcnemar_p_value(graph_only: int, baseline_only: int) -> float:
    discordant = graph_only + baseline_only
    if discordant == 0:
        return 1.0
    lower = min(graph_only, baseline_only)
    numerator = sum(math.comb(discordant, index) for index in range(lower + 1))
    return min(1.0, 2.0 * numerator / (2**discordant))


def _paired_row(
    questions: Sequence[RetrievalQuestion],
    baseline: Sequence[Sequence[int]],
    graph: Sequence[Sequence[int]],
    *,
    extra_budget: int,
) -> dict[str, Any]:
    counts = Counter()
    for question, base_rank, graph_rank in zip(
        questions, baseline, graph, strict=True
    ):
        top_k = len(question.gold_indices) + extra_budget
        gold = set(question.gold_indices)
        base_complete = gold.issubset(set(base_rank[:top_k]))
        graph_complete = gold.issubset(set(graph_rank[:top_k]))
        if base_complete and graph_complete:
            counts["both"] += 1
        elif graph_complete:
            counts["graph_only"] += 1
        elif base_complete:
            counts["baseline_only"] += 1
        else:
            counts["neither"] += 1
    count = len(questions)
    return {
        "question_count": count,
        "both_complete_count": counts["both"],
        "graph_only_complete_count": counts["graph_only"],
        "baseline_only_complete_count": counts["baseline_only"],
        "neither_complete_count": counts["neither"],
        "net_complete_count_gain": (
            counts["graph_only"] - counts["baseline_only"]
        ),
        "net_complete_rate_gain": (
            counts["graph_only"] - counts["baseline_only"]
        )
        / count,
        "exact_mcnemar_two_sided_p": _exact_mcnemar_p_value(
            counts["graph_only"], counts["baseline_only"]
        ),
    }


def paired_transition_report(
    questions: Sequence[RetrievalQuestion],
    baseline: Sequence[Sequence[int]],
    graph: Sequence[Sequence[int]],
) -> dict[str, Any]:
    indices_by_cell = {
        cell: [
            index
            for index, question in enumerate(questions)
            if question.question_type == cell
        ]
        for cell in all_cells()
    }

    def subset(values: Sequence[Any], indices: Sequence[int]) -> tuple[Any, ...]:
        return tuple(values[index] for index in indices)

    return {
        "overall": {
            str(extra): _paired_row(
                questions, baseline, graph, extra_budget=extra
            )
            for extra in EXTRA_BUDGETS
        },
        "by_cell": {
            cell: {
                str(extra): _paired_row(
                    subset(questions, indices),
                    subset(baseline, indices),
                    subset(graph, indices),
                    extra_budget=extra,
                )
                for extra in EXTRA_BUDGETS
            }
            for cell, indices in indices_by_cell.items()
        },
    }


def _graph_summary(questions: Sequence[RetrievalQuestion]) -> dict[str, Any]:
    if not questions:
        raise MuSiQueGraphRetrievalError("graph diagnostic subset is empty")
    edge_count = 0
    no_edge_count = 0
    any_gold_edge = 0
    connected_gold = 0
    for question in questions:
        edges = set(question.mention_edges)
        edge_count += len(edges)
        no_edge_count += not edges
        gold = set(question.gold_indices)
        adjacency = {index: set() for index in gold}
        for source, target in edges:
            if source in gold and target in gold:
                adjacency[source].add(target)
                adjacency[target].add(source)
        any_gold_edge += any(adjacency.values())
        visited: set[int] = set()
        frontier = [next(iter(gold))]
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            frontier.extend(adjacency[current].difference(visited))
        connected_gold += visited == gold
    count = len(questions)
    return {
        "question_count": count,
        "mean_title_mention_edge_count": edge_count / count,
        "no_title_mention_edge_count": no_edge_count,
        "no_title_mention_edge_rate": no_edge_count / count,
        "any_gold_to_gold_edge_count": any_gold_edge,
        "any_gold_to_gold_edge_rate": any_gold_edge / count,
        "direction_ignored_gold_subgraph_connected_count": connected_gold,
        "direction_ignored_gold_subgraph_connected_rate": connected_gold / count,
    }


def graph_diagnostics(
    questions: Sequence[RetrievalQuestion],
) -> dict[str, Any]:
    return {
        "overall": _graph_summary(questions),
        "by_cell": {
            cell: _graph_summary(
                tuple(
                    question
                    for question in questions
                    if question.question_type == cell
                )
            )
            for cell in all_cells()
        },
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _null_summary(values: Sequence[float], actual: float) -> dict[str, Any]:
    exceed = sum(value >= actual for value in values)
    return {
        "actual": actual,
        "null_mean": statistics.fmean(values),
        "null_std": statistics.pstdev(values),
        "null_min": min(values),
        "null_p50": _percentile(values, 0.5),
        "null_p95": _percentile(values, 0.95),
        "null_max": max(values),
        "actual_minus_null_mean": actual - statistics.fmean(values),
        "actual_minus_null_max": actual - max(values),
        "null_at_least_actual_count": exceed,
        "empirical_one_sided_p": (exceed + 1) / (len(values) + 1),
    }


def permutation_null_report(
    questions: Sequence[RetrievalQuestion],
    config: GraphMethodConfig,
    actual_metrics: Mapping[str, Any],
    *,
    repetitions: int = PERMUTATION_REPETITIONS,
    seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    if repetitions < 1:
        raise MuSiQueGraphRetrievalError(
            "permutation repetitions must be positive"
        )
    distributions = {
        "macro_mean": [],
        "macro_mrr": [],
        **{f"extra_{extra}": [] for extra in EXTRA_BUDGETS},
    }
    for replicate in range(repetitions):
        permuted = tuple(
            permute_question_graph(question, replicate=replicate, seed=seed)
            for question in questions
        )
        metrics = evaluate_musique_rankings(
            permuted, graph_rankings(permuted, config)
        )
        distributions["macro_mean"].append(
            metrics["macro"]["mean_complete_gold_evidence_rate"]
        )
        distributions["macro_mrr"].append(
            metrics["macro"]["full_evidence_mrr"]
        )
        for extra in EXTRA_BUDGETS:
            distributions[f"extra_{extra}"].append(
                metrics["macro"]["by_extra_budget"][str(extra)][
                    "macro_complete_gold_evidence_rate"
                ]
            )
    return {
        "null": "question-local node-label permutation",
        "preserved": (
            "directed graph isomorphism, edge count, degree sequence, paragraph "
            "scores, gold labels, hop-collision cells, and context budget"
        ),
        "destroyed": (
            "alignment between graph nodes and paragraph/title identities"
        ),
        "repetitions": repetitions,
        "seed": seed,
        "macro_mean_complete_gold_evidence_rate": _null_summary(
            distributions["macro_mean"],
            actual_metrics["macro"]["mean_complete_gold_evidence_rate"],
        ),
        "macro_full_evidence_mrr": _null_summary(
            distributions["macro_mrr"],
            actual_metrics["macro"]["full_evidence_mrr"],
        ),
        "macro_complete_rate_by_extra_budget": {
            str(extra): _null_summary(
                distributions[f"extra_{extra}"],
                actual_metrics["macro"]["by_extra_budget"][str(extra)][
                    "macro_complete_gold_evidence_rate"
                ],
            )
            for extra in EXTRA_BUDGETS
        },
    }


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "macro": metrics["macro"],
        "micro": metrics["micro"],
        "by_hop": metrics["by_hop"],
        "by_title_collision": metrics["by_title_collision"],
    }


def build_musique_graph_report(
    train_questions: Sequence[RetrievalQuestion],
    dev_questions: Sequence[RetrievalQuestion],
    *,
    provenance: Mapping[str, Any],
    hotpot_transfer_config: GraphMethodConfig,
    hotpot_authorization: Mapping[str, Any],
    permutation_repetitions: int = PERMUTATION_REPETITIONS,
    permutation_seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """Select on balanced train and evaluate once on full official dev."""

    for split, questions in (("train", train_questions), ("dev", dev_questions)):
        counts = Counter(question.question_type for question in questions)
        if set(counts) != set(all_cells()):
            raise MuSiQueGraphRetrievalError(
                f"{split} does not contain every frozen cell"
            )
        if len({question.qid for question in questions}) != len(questions):
            raise MuSiQueGraphRetrievalError(
                f"{split} question IDs are not unique"
            )
        if split == "train" and any(
            counts[cell] != TRAIN_PER_CELL for cell in all_cells()
        ):
            raise MuSiQueGraphRetrievalError(
                "train is not balanced at 200 questions per cell"
            )
    if {question.qid for question in train_questions}.intersection(
        question.qid for question in dev_questions
    ):
        raise MuSiQueGraphRetrievalError("train and dev IDs overlap")

    baseline_names = ("dense", "bm25", "dense_bm25_rrf")
    train_baselines = {
        name: evaluate_baseline(train_questions, name)
        for name in baseline_names
    }
    dev_baselines = {
        name: evaluate_baseline(dev_questions, name) for name in baseline_names
    }
    selected_baseline_name, selected_train_baseline = select_best_result(
        list(train_baselines.items())
    )

    candidates = graph_candidate_grid()
    config_by_name = {config.name: config for config in candidates}
    train_graph_results = [
        (config.name, evaluate_graph(train_questions, config))
        for config in candidates
    ]
    selected_graph_name, selected_train_graph = select_best_result(
        train_graph_results
    )
    selected_graph_config = config_by_name[selected_graph_name]
    selected_dev_graph = evaluate_graph(dev_questions, selected_graph_config)

    train_degree_results = [
        (config.name, evaluate_degree_prior(train_questions, config))
        for config in candidates
    ]
    selected_degree_name, selected_train_degree = select_best_result(
        train_degree_results
    )
    selected_degree_config = config_by_name[selected_degree_name]
    selected_dev_degree = evaluate_degree_prior(
        dev_questions, selected_degree_config
    )

    transfer_train = evaluate_graph(train_questions, hotpot_transfer_config)
    transfer_dev = evaluate_graph(dev_questions, hotpot_transfer_config)
    selected_dev_baseline = dev_baselines[selected_baseline_name]
    graph_vs_baseline = _comparison(
        selected_dev_graph, selected_dev_baseline
    )
    graph_vs_degree = _comparison(selected_dev_graph, selected_dev_degree)
    transfer_vs_baseline = _comparison(
        transfer_dev, dev_baselines[hotpot_transfer_config.seed]
    )
    baseline_ranking = baseline_rankings(
        dev_questions, selected_baseline_name
    )
    selected_graph_ranking = graph_rankings(
        dev_questions, selected_graph_config
    )
    paired = paired_transition_report(
        dev_questions, baseline_ranking, selected_graph_ranking
    )
    null = permutation_null_report(
        dev_questions,
        selected_graph_config,
        selected_dev_graph,
        repetitions=permutation_repetitions,
        seed=permutation_seed,
    )

    baseline_deltas = graph_vs_baseline[
        "absolute_macro_complete_delta_by_extra_budget"
    ]
    paired_passed = all(
        row["graph_only_complete_count"]
        > row["baseline_only_complete_count"]
        and row["exact_mcnemar_two_sided_p"] <= MCNEMAR_P_THRESHOLD
        for row in paired["overall"].values()
    )
    null_mean = null["macro_mean_complete_gold_evidence_rate"]
    checks = {
        "strongest_non_graph_baseline": {
            "baseline": selected_baseline_name,
            "required_mean_delta": STRONG_BASELINE_MEAN_DELTA_THRESHOLD,
            "maximum_per_budget_regression": MAX_PER_BUDGET_REGRESSION,
            "observed_mean_delta": graph_vs_baseline[
                "absolute_macro_mean_complete_delta"
            ],
            "observed_per_budget_deltas": baseline_deltas,
            "passed": (
                graph_vs_baseline["absolute_macro_mean_complete_delta"]
                >= STRONG_BASELINE_MEAN_DELTA_THRESHOLD
                and min(baseline_deltas.values())
                >= -MAX_PER_BUDGET_REGRESSION
            ),
        },
        "degree_only_control": {
            "required_mean_delta": DEGREE_CONTROL_MEAN_DELTA_THRESHOLD,
            "maximum_per_budget_regression": MAX_PER_BUDGET_REGRESSION,
            "observed_mean_delta": graph_vs_degree[
                "absolute_macro_mean_complete_delta"
            ],
            "observed_per_budget_deltas": graph_vs_degree[
                "absolute_macro_complete_delta_by_extra_budget"
            ],
            "passed": (
                graph_vs_degree["absolute_macro_mean_complete_delta"]
                >= DEGREE_CONTROL_MEAN_DELTA_THRESHOLD
                and min(
                    graph_vs_degree[
                        "absolute_macro_complete_delta_by_extra_budget"
                    ].values()
                )
                >= -MAX_PER_BUDGET_REGRESSION
            ),
        },
        "node_permutation_null": {
            "empirical_p_threshold": PERMUTATION_P_THRESHOLD,
            "empirical_p": null_mean["empirical_one_sided_p"],
            "actual_minus_null_max": null_mean["actual_minus_null_max"],
            "passed": (
                null_mean["empirical_one_sided_p"]
                <= PERMUTATION_P_THRESHOLD
                and null_mean["actual_minus_null_max"] > 0.0
            ),
        },
        "paired_mcnemar_every_budget": {
            "p_threshold": MCNEMAR_P_THRESHOLD,
            "passed": paired_passed,
        },
    }
    passed = all(row["passed"] for row in checks.values())
    graph_leaderboard = sorted(
        train_graph_results, key=lambda item: _selection_key(item[1]), reverse=True
    )
    degree_leaderboard = sorted(
        train_degree_results,
        key=lambda item: _selection_key(item[1]),
        reverse=True,
    )
    return {
        "schema_version": MUSIQUE_GRAPH_REPORT_SCHEMA_VERSION,
        "task": "musique_occurrence_graph_shortcut_controlled_retrieval",
        "scope": "official_answerable_exact20_frozen_train_and_dev_pools",
        "protocol": {
            "extra_budgets": list(EXTRA_BUDGETS),
            "top_k_rule": "gold_document_count plus extra budget",
            "primary_aggregate": "macro average across nine hop-collision cells",
            "train_only_hyperparameter_selection": True,
            "dev_used_for_selection": False,
            "graph_candidate_count": len(candidates),
            "degree_candidate_count": len(candidates),
            "stable_tie_break": "ascending official paragraph idx",
            "graph_nodes": "official paragraph occurrences keyed by idx",
            "graph_edges": (
                "literal directed body-to-title mentions with deterministic "
                "fanout to every same-title occurrence"
            ),
            "duplicate_titles_merged": False,
            "gold_labels_used_for": "aggregate evaluation and frozen strata only",
            "selection_order": (
                "nine-cell macro mean complete rate; extra budgets 3,1,0; "
                "macro full-evidence MRR; declared grid order"
            ),
            "permutation_repetitions": permutation_repetitions,
            "permutation_seed": permutation_seed,
        },
        "provenance": dict(provenance),
        "hotpot_transfer_authorization": dict(hotpot_authorization),
        "selection": {
            "selected_baseline": selected_baseline_name,
            "selected_graph_method": selected_graph_name,
            "selected_graph_config": asdict(selected_graph_config),
            "selected_degree_only_method": selected_degree_name,
            "selected_degree_only_config": asdict(selected_degree_config),
            "hotpot_transfer_config": asdict(hotpot_transfer_config),
        },
        "train": {
            "question_count": len(train_questions),
            "graph_diagnostics": graph_diagnostics(train_questions),
            "baselines": train_baselines,
            "selected_baseline_metrics": selected_train_baseline,
            "selected_graph_metrics": selected_train_graph,
            "selected_degree_only_metrics": selected_train_degree,
            "hotpot_zero_shot_graph": transfer_train,
            "graph_candidate_leaderboard": [
                {
                    "rank": rank,
                    "name": name,
                    "config": asdict(config_by_name[name]),
                    "metrics": _compact_metrics(metrics),
                }
                for rank, (name, metrics) in enumerate(
                    graph_leaderboard, start=1
                )
            ],
            "degree_candidate_leaderboard": [
                {
                    "rank": rank,
                    "name": name,
                    "config": asdict(config_by_name[name]),
                    "metrics": _compact_metrics(metrics),
                }
                for rank, (name, metrics) in enumerate(
                    degree_leaderboard, start=1
                )
            ],
        },
        "dev": {
            "question_count": len(dev_questions),
            "graph_diagnostics": graph_diagnostics(dev_questions),
            "baselines": dev_baselines,
            "selected_graph": selected_dev_graph,
            "selected_degree_only": selected_dev_degree,
            "hotpot_zero_shot_graph": transfer_dev,
            "selected_graph_vs_selected_baseline": graph_vs_baseline,
            "selected_graph_vs_selected_degree_only": graph_vs_degree,
            "hotpot_transfer_vs_matching_baseline": transfer_vs_baseline,
            "paired_transitions_vs_selected_baseline": paired,
            "node_permutation_null": null,
        },
        "gate": {
            "name": "musique_query_conditioned_graph_alignment_gate_v1",
            "checks": checks,
            "passed": passed,
            "status": (
                "support_query_conditioned_graph_claim"
                if passed
                else "do_not_claim_query_conditioned_graph_gain"
            ),
            "interpretation": (
                "Passing requires the train-selected graph to beat both the "
                "strongest non-graph method and an independently train-selected "
                "degree-only control, survive a node-identity permutation null, "
                "and show significant paired gains at every frozen budget."
            ),
        },
        "content_contract": (
            "aggregate metrics, configurations, paths, and hashes only; no IDs, "
            "questions, answers, titles, paragraph text, decomposition text, "
            "or supporting labels"
        ),
    }
