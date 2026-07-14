"""Balanced multi-gold graph-retrieval evaluation for 2WikiMultiHopQA."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import math
from typing import Any

from .baselines import BM25Config, PerQuestionBM25
from .graph import build_title_mention_edges, document_nodes
from .graph_retrieval import (
    GraphMethodConfig,
    RetrievalQuestion,
    baseline_rankings,
    graph_candidate_grid,
    graph_rankings,
)
from .twowiki import (
    EXPECTED_GOLD_DOCUMENT_COUNTS,
    QUESTION_TYPES,
    TwoWikiInvariantError,
    assess_twowiki_eligibility,
)


REPORT_SCHEMA_VERSION = 1
EXTRA_BUDGETS = (0, 1, 3)
MEAN_COMPLETE_RATE_GATE_DELTA = 0.02
MAX_PER_BUDGET_REGRESSION = 0.01
FROZEN_HOTPOT_TRANSFER_CONFIG = GraphMethodConfig(
    family="one_hop_max",
    seed="dense_bm25_rrf",
    direction="undirected",
    graph_weight=0.75,
)


class TwoWikiGraphRetrievalError(ValueError):
    """Raised when a 2Wiki retrieval evaluation invariant is violated."""


def prepare_twowiki_retrieval_question(
    example: Mapping[str, Any], dense_scores: Sequence[float]
) -> RetrievalQuestion:
    """Build label-free scores and graph, attaching gold only for evaluation."""

    assessment = assess_twowiki_eligibility(example)
    if not assessment.eligible:
        raise TwoWikiGraphRetrievalError("2Wiki retrieval question is not eligible")
    question_type = assessment.question_type
    if question_type not in EXPECTED_GOLD_DOCUMENT_COUNTS:
        raise TwoWikiGraphRetrievalError("unsupported 2Wiki question type")
    qid = example.get("_id")
    if not isinstance(qid, str) or not qid:
        raise TwoWikiGraphRetrievalError("question ID must be non-empty")

    documents = document_nodes(example)
    if len(dense_scores) != len(documents):
        raise TwoWikiGraphRetrievalError(
            "dense score count must equal context document count"
        )
    dense: list[float] = []
    for score in dense_scores:
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise TwoWikiGraphRetrievalError("dense scores must be numeric")
        value = float(score)
        if not math.isfinite(value):
            raise TwoWikiGraphRetrievalError("dense scores must be finite")
        dense.append(value)

    serialized = {
        str(document.index): document.serialized_text for document in documents
    }
    bm25_index = PerQuestionBM25.fit(serialized, config=BM25Config())
    bm25_by_id = bm25_index.score(str(example["question"]))
    bm25 = tuple(bm25_by_id[str(index)] for index in range(len(documents)))

    exact_index_by_title = {
        document.title: document.index for document in documents
    }
    if len(exact_index_by_title) != len(documents):
        raise TwoWikiGraphRetrievalError("exact context titles must be unique")
    ordered_gold: list[int] = []
    seen_gold: set[int] = set()
    for title, _sentence_index in example["supporting_facts"]:
        try:
            index = exact_index_by_title[title]
        except KeyError as exc:
            raise TwoWikiGraphRetrievalError(
                "supporting title is missing under exact matching"
            ) from exc
        if index not in seen_gold:
            seen_gold.add(index)
            ordered_gold.append(index)
    expected_count = EXPECTED_GOLD_DOCUMENT_COUNTS[question_type]
    if len(ordered_gold) != expected_count:
        raise TwoWikiGraphRetrievalError(
            "gold document count does not match question-type contract"
        )

    mention_edges = tuple(
        (edge.source_index, edge.target_index)
        for edge in build_title_mention_edges(documents)
    )
    return RetrievalQuestion(
        qid=qid,
        dense_scores=tuple(dense),
        bm25_scores=bm25,
        gold_indices=tuple(ordered_gold),
        mention_edges=mention_edges,
        question_type=question_type,
    )


def _validate_rankings(
    questions: Sequence[RetrievalQuestion], rankings: Sequence[Sequence[int]]
) -> tuple[tuple[int, ...], ...]:
    if not questions or len(questions) != len(rankings):
        raise TwoWikiGraphRetrievalError(
            "questions and rankings must be non-empty and equally sized"
        )
    normalized: list[tuple[int, ...]] = []
    for question, ranking_value in zip(questions, rankings, strict=True):
        ranking = tuple(ranking_value)
        if set(ranking) != set(range(question.document_count)):
            raise TwoWikiGraphRetrievalError(
                "ranking must be a full context-index permutation"
            )
        if question.question_type not in EXPECTED_GOLD_DOCUMENT_COUNTS:
            raise TwoWikiGraphRetrievalError("question type is missing or invalid")
        expected_gold = EXPECTED_GOLD_DOCUMENT_COUNTS[question.question_type]
        if len(question.gold_indices) != expected_gold:
            raise TwoWikiGraphRetrievalError("question gold count is invalid")
        normalized.append(ranking)
    return tuple(normalized)


def _stratum_metrics(
    questions: Sequence[RetrievalQuestion],
    rankings: Sequence[Sequence[int]],
    *,
    question_type: str,
) -> dict[str, Any]:
    gold_count = EXPECTED_GOLD_DOCUMENT_COUNTS[question_type]
    complete = {extra: 0 for extra in EXTRA_BUDGETS}
    at_least_one = {extra: 0 for extra in EXTRA_BUDGETS}
    no_gold = {extra: 0 for extra in EXTRA_BUDGETS}
    support_sum = {extra: 0 for extra in EXTRA_BUDGETS}
    reciprocal_full_ranks: list[float] = []

    for question, ranking_value in zip(questions, rankings, strict=True):
        ranking = tuple(ranking_value)
        rank_by_index = {index: rank for rank, index in enumerate(ranking, start=1)}
        gold = set(question.gold_indices)
        reciprocal_full_ranks.append(
            1.0 / max(rank_by_index[index] for index in gold)
        )
        for extra in EXTRA_BUDGETS:
            top_k = gold_count + extra
            retrieved = set(ranking[:top_k])
            found = len(gold.intersection(retrieved))
            complete[extra] += found == gold_count
            at_least_one[extra] += found >= 1
            no_gold[extra] += found == 0
            support_sum[extra] += found

    count = len(questions)
    by_budget = {
        str(extra): {
            "extra_budget": extra,
            "top_k": gold_count + extra,
            "complete_gold_evidence_count": complete[extra],
            "complete_gold_evidence_rate": complete[extra] / count,
            "support_document_recall": support_sum[extra] / (gold_count * count),
            "at_least_one_gold_count": at_least_one[extra],
            "at_least_one_gold_rate": at_least_one[extra] / count,
            "no_gold_count": no_gold[extra],
            "no_gold_rate": no_gold[extra] / count,
        }
        for extra in EXTRA_BUDGETS
    }
    return {
        "question_count": count,
        "gold_document_count": gold_count,
        "top_k_by_extra_budget": {
            str(extra): gold_count + extra for extra in EXTRA_BUDGETS
        },
        "by_extra_budget": by_budget,
        "mean_complete_gold_evidence_rate": sum(
            by_budget[str(extra)]["complete_gold_evidence_rate"]
            for extra in EXTRA_BUDGETS
        )
        / len(EXTRA_BUDGETS),
        "full_evidence_mrr": sum(reciprocal_full_ranks) / count,
    }


def evaluate_twowiki_rankings(
    questions: Sequence[RetrievalQuestion], rankings: Sequence[Sequence[int]]
) -> dict[str, Any]:
    """Evaluate type-specific K and return type-macro and sample-micro metrics."""

    normalized_rankings = _validate_rankings(questions, rankings)
    grouped_questions: defaultdict[str, list[RetrievalQuestion]] = defaultdict(list)
    grouped_rankings: defaultdict[str, list[tuple[int, ...]]] = defaultdict(list)
    for question, ranking in zip(questions, normalized_rankings, strict=True):
        assert question.question_type is not None
        grouped_questions[question.question_type].append(question)
        grouped_rankings[question.question_type].append(ranking)
    if set(grouped_questions) != set(QUESTION_TYPES):
        raise TwoWikiGraphRetrievalError(
            "evaluation must contain all four frozen question types"
        )

    by_type = {
        question_type: _stratum_metrics(
            grouped_questions[question_type],
            grouped_rankings[question_type],
            question_type=question_type,
        )
        for question_type in QUESTION_TYPES
    }
    macro_by_budget = {
        str(extra): {
            "extra_budget": extra,
            "macro_complete_gold_evidence_rate": sum(
                by_type[question_type]["by_extra_budget"][str(extra)][
                    "complete_gold_evidence_rate"
                ]
                for question_type in QUESTION_TYPES
            )
            / len(QUESTION_TYPES),
            "macro_support_document_recall": sum(
                by_type[question_type]["by_extra_budget"][str(extra)][
                    "support_document_recall"
                ]
                for question_type in QUESTION_TYPES
            )
            / len(QUESTION_TYPES),
        }
        for extra in EXTRA_BUDGETS
    }

    micro_complete = {extra: 0 for extra in EXTRA_BUDGETS}
    micro_support = {extra: 0 for extra in EXTRA_BUDGETS}
    total_gold_documents = 0
    micro_reciprocal_full: list[float] = []
    for question, ranking in zip(questions, normalized_rankings, strict=True):
        gold = set(question.gold_indices)
        total_gold_documents += len(gold)
        rank_by_index = {index: rank for rank, index in enumerate(ranking, start=1)}
        micro_reciprocal_full.append(
            1.0 / max(rank_by_index[index] for index in gold)
        )
        for extra in EXTRA_BUDGETS:
            retrieved = set(ranking[: len(gold) + extra])
            found = len(gold.intersection(retrieved))
            micro_complete[extra] += found == len(gold)
            micro_support[extra] += found

    count = len(questions)
    return {
        "question_count": count,
        "question_count_by_type": {
            question_type: len(grouped_questions[question_type])
            for question_type in QUESTION_TYPES
        },
        "extra_budgets": list(EXTRA_BUDGETS),
        "by_question_type": by_type,
        "macro": {
            "by_extra_budget": macro_by_budget,
            "mean_complete_gold_evidence_rate": sum(
                macro_by_budget[str(extra)][
                    "macro_complete_gold_evidence_rate"
                ]
                for extra in EXTRA_BUDGETS
            )
            / len(EXTRA_BUDGETS),
            "full_evidence_mrr": sum(
                by_type[question_type]["full_evidence_mrr"]
                for question_type in QUESTION_TYPES
            )
            / len(QUESTION_TYPES),
        },
        "micro": {
            "by_extra_budget": {
                str(extra): {
                    "complete_gold_evidence_rate": micro_complete[extra] / count,
                    "support_document_recall": (
                        micro_support[extra] / total_gold_documents
                    ),
                }
                for extra in EXTRA_BUDGETS
            },
            "mean_complete_gold_evidence_rate": sum(
                micro_complete[extra] / count for extra in EXTRA_BUDGETS
            )
            / len(EXTRA_BUDGETS),
            "full_evidence_mrr": sum(micro_reciprocal_full) / count,
        },
    }


def evaluate_twowiki_baseline(
    questions: Sequence[RetrievalQuestion], method: str
) -> dict[str, Any]:
    return evaluate_twowiki_rankings(
        questions, baseline_rankings(questions, method)
    )


def evaluate_twowiki_graph(
    questions: Sequence[RetrievalQuestion], config: GraphMethodConfig
) -> dict[str, Any]:
    return evaluate_twowiki_rankings(questions, graph_rankings(questions, config))


def _selection_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    macro = metrics["macro"]
    by_budget = macro["by_extra_budget"]
    return (
        float(macro["mean_complete_gold_evidence_rate"]),
        *(
            float(by_budget[str(extra)]["macro_complete_gold_evidence_rate"])
            for extra in reversed(EXTRA_BUDGETS)
        ),
        float(macro["full_evidence_mrr"]),
    )


def select_best_twowiki_result(
    named_results: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[str, Mapping[str, Any]]:
    if not named_results:
        raise TwoWikiGraphRetrievalError("named results must not be empty")
    best_name, best_metrics = named_results[0]
    best_key = _selection_key(best_metrics)
    for name, metrics in named_results[1:]:
        key = _selection_key(metrics)
        if key > best_key:
            best_name, best_metrics, best_key = name, metrics, key
    return best_name, best_metrics


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
    for question, baseline_ranking, graph_ranking in zip(
        questions, baseline, graph, strict=True
    ):
        top_k = len(question.gold_indices) + extra_budget
        gold = set(question.gold_indices)
        base_complete = gold.issubset(set(baseline_ranking[:top_k]))
        graph_complete = gold.issubset(set(graph_ranking[:top_k]))
        if base_complete and graph_complete:
            counts["both"] += 1
        elif graph_complete:
            counts["graph_only"] += 1
        elif base_complete:
            counts["baseline_only"] += 1
        else:
            counts["neither"] += 1
    return {
        "question_count": len(questions),
        "both_complete_count": counts["both"],
        "graph_only_complete_count": counts["graph_only"],
        "baseline_only_complete_count": counts["baseline_only"],
        "neither_complete_count": counts["neither"],
        "net_complete_count_gain": counts["graph_only"] - counts["baseline_only"],
        "net_complete_rate_gain": (
            counts["graph_only"] - counts["baseline_only"]
        )
        / len(questions),
        "exact_mcnemar_two_sided_p": _exact_mcnemar_p_value(
            counts["graph_only"], counts["baseline_only"]
        ),
    }


def paired_transition_report(
    questions: Sequence[RetrievalQuestion],
    baseline: Sequence[Sequence[int]],
    graph: Sequence[Sequence[int]],
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = {kind: [] for kind in QUESTION_TYPES}
    for index, question in enumerate(questions):
        assert question.question_type is not None
        grouped[question.question_type].append(index)

    def subset(values: Sequence[Sequence[int]], indices: Sequence[int]):
        return tuple(values[index] for index in indices)

    return {
        "overall": {
            str(extra): _paired_row(
                questions, baseline, graph, extra_budget=extra
            )
            for extra in EXTRA_BUDGETS
        },
        "by_question_type": {
            question_type: {
                str(extra): _paired_row(
                    tuple(questions[index] for index in indices),
                    subset(baseline, indices),
                    subset(graph, indices),
                    extra_budget=extra,
                )
                for extra in EXTRA_BUDGETS
            }
            for question_type, indices in grouped.items()
        },
    }


def graph_diagnostics(questions: Sequence[RetrievalQuestion]) -> dict[str, Any]:
    """Describe gold-subgraph connectivity without influencing retrieval scores."""

    def summarize(subset: Sequence[RetrievalQuestion]) -> dict[str, Any]:
        edge_count = 0
        any_gold_edge = 0
        connected_gold = 0
        for question in subset:
            edge_count += len(question.mention_edges)
            gold = set(question.gold_indices)
            adjacency = {index: set() for index in gold}
            for source, target in question.mention_edges:
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
        count = len(subset)
        return {
            "question_count": count,
            "mean_title_mention_edge_count": edge_count / count,
            "any_gold_to_gold_mention_edge_count": any_gold_edge,
            "any_gold_to_gold_mention_edge_rate": any_gold_edge / count,
            "direction_ignored_gold_subgraph_connected_count": connected_gold,
            "direction_ignored_gold_subgraph_connected_rate": connected_gold / count,
        }

    by_type = {
        question_type: tuple(
            question
            for question in questions
            if question.question_type == question_type
        )
        for question_type in QUESTION_TYPES
    }
    return {
        "overall": summarize(questions),
        "by_question_type": {
            question_type: summarize(subset)
            for question_type, subset in by_type.items()
        },
    }


def _comparison(
    graph_metrics: Mapping[str, Any], baseline_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    macro_delta = (
        graph_metrics["macro"]["mean_complete_gold_evidence_rate"]
        - baseline_metrics["macro"]["mean_complete_gold_evidence_rate"]
    )
    by_budget = {
        str(extra): (
            graph_metrics["macro"]["by_extra_budget"][str(extra)][
                "macro_complete_gold_evidence_rate"
            ]
            - baseline_metrics["macro"]["by_extra_budget"][str(extra)][
                "macro_complete_gold_evidence_rate"
            ]
        )
        for extra in EXTRA_BUDGETS
    }
    by_type = {
        question_type: {
            str(extra): (
                graph_metrics["by_question_type"][question_type][
                    "by_extra_budget"
                ][str(extra)]["complete_gold_evidence_rate"]
                - baseline_metrics["by_question_type"][question_type][
                    "by_extra_budget"
                ][str(extra)]["complete_gold_evidence_rate"]
            )
            for extra in EXTRA_BUDGETS
        }
        for question_type in QUESTION_TYPES
    }
    return {
        "absolute_macro_mean_complete_delta": macro_delta,
        "absolute_macro_complete_delta_by_extra_budget": by_budget,
        "absolute_complete_delta_by_question_type_and_extra_budget": by_type,
    }


def build_twowiki_graph_retrieval_report(
    train_questions: Sequence[RetrievalQuestion],
    dev_questions: Sequence[RetrievalQuestion],
    *,
    provenance: Mapping[str, Any],
    hotpot_transfer_config: GraphMethodConfig,
    hotpot_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate frozen Hotpot transfer and train-selected 2Wiki reproduction."""

    for split_name, questions in (("train", train_questions), ("dev", dev_questions)):
        type_counts = Counter(question.question_type for question in questions)
        if set(type_counts) != set(QUESTION_TYPES) or len(set(type_counts.values())) != 1:
            raise TwoWikiGraphRetrievalError(
                f"{split_name} questions must be balanced across all four types"
            )
        if len({question.qid for question in questions}) != len(questions):
            raise TwoWikiGraphRetrievalError(
                f"{split_name} question IDs must be unique"
            )
    if set(question.qid for question in train_questions).intersection(
        question.qid for question in dev_questions
    ):
        raise TwoWikiGraphRetrievalError("train and dev question IDs overlap")

    baseline_names = ("dense", "bm25", "dense_bm25_rrf")
    train_baselines = {
        name: evaluate_twowiki_baseline(train_questions, name)
        for name in baseline_names
    }
    dev_baselines = {
        name: evaluate_twowiki_baseline(dev_questions, name)
        for name in baseline_names
    }
    selected_baseline_name, selected_train_baseline = select_best_twowiki_result(
        list(train_baselines.items())
    )

    candidates = graph_candidate_grid()
    config_by_name = {config.name: config for config in candidates}
    train_graph_results = [
        (config.name, evaluate_twowiki_graph(train_questions, config))
        for config in candidates
    ]
    selected_graph_name, selected_train_graph = select_best_twowiki_result(
        train_graph_results
    )
    selected_graph_config = config_by_name[selected_graph_name]
    selected_dev_graph = evaluate_twowiki_graph(
        dev_questions, selected_graph_config
    )

    transfer_train = evaluate_twowiki_graph(
        train_questions, hotpot_transfer_config
    )
    transfer_dev = evaluate_twowiki_graph(dev_questions, hotpot_transfer_config)
    matching_baseline_name = hotpot_transfer_config.seed
    transfer_comparison = _comparison(
        transfer_dev, dev_baselines[matching_baseline_name]
    )
    tuned_comparison = _comparison(
        selected_dev_graph, dev_baselines[selected_baseline_name]
    )

    transfer_baseline_rankings = baseline_rankings(
        dev_questions, matching_baseline_name
    )
    transfer_graph_rankings = graph_rankings(
        dev_questions, hotpot_transfer_config
    )
    mean_delta = transfer_comparison["absolute_macro_mean_complete_delta"]
    per_budget_delta = transfer_comparison[
        "absolute_macro_complete_delta_by_extra_budget"
    ]
    gate_passed = (
        mean_delta >= MEAN_COMPLETE_RATE_GATE_DELTA
        and min(per_budget_delta.values()) >= -MAX_PER_BUDGET_REGRESSION
    )
    leaderboard = sorted(
        train_graph_results, key=lambda item: _selection_key(item[1]), reverse=True
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "task": "2wiki_balanced_controlled_pool_graph_retrieval_reproduction",
        "scope": "official_ten_document_context_pool_clean_balanced_sample",
        "protocol": {
            "extra_budgets": list(EXTRA_BUDGETS),
            "top_k_rule": "gold_document_count + extra_budget",
            "top_k_by_question_type": {
                question_type: {
                    str(extra): EXPECTED_GOLD_DOCUMENT_COUNTS[question_type] + extra
                    for extra in EXTRA_BUDGETS
                }
                for question_type in QUESTION_TYPES
            },
            "primary_aggregate": "macro average across four question types",
            "hotpot_transfer_uses_2wiki_selection": False,
            "within_2wiki_selection": "frozen 36 configurations on train only",
            "dev_used_for_selection": False,
            "stable_tie_break": "ascending official context index",
            "graph_edges": "literal directed cross-document title mentions",
            "gold_labels_used_for": "aggregate evaluation only",
        },
        "provenance": dict(provenance),
        "hotpot_transfer_authorization": dict(hotpot_authorization),
        "selection": {
            "hotpot_transfer_config": asdict(hotpot_transfer_config),
            "matching_transfer_baseline": matching_baseline_name,
            "selected_2wiki_baseline": selected_baseline_name,
            "selected_2wiki_graph_method": selected_graph_name,
            "selected_2wiki_graph_config": asdict(selected_graph_config),
        },
        "train": {
            "question_count": len(train_questions),
            "graph_diagnostics": graph_diagnostics(train_questions),
            "baselines": train_baselines,
            "hotpot_zero_shot_graph": transfer_train,
            "selected_2wiki_baseline_metrics": selected_train_baseline,
            "selected_2wiki_graph_metrics": selected_train_graph,
            "graph_candidate_leaderboard": [
                {
                    "rank": rank,
                    "name": name,
                    "config": asdict(config_by_name[name]),
                    "metrics": metrics,
                }
                for rank, (name, metrics) in enumerate(leaderboard, start=1)
            ],
        },
        "dev": {
            "question_count": len(dev_questions),
            "graph_diagnostics": graph_diagnostics(dev_questions),
            "baselines": dev_baselines,
            "hotpot_zero_shot_graph": transfer_dev,
            "selected_2wiki_graph": selected_dev_graph,
            "zero_shot_vs_matching_baseline": {
                **transfer_comparison,
                "paired_transitions": paired_transition_report(
                    dev_questions,
                    transfer_baseline_rankings,
                    transfer_graph_rankings,
                ),
            },
            "tuned_graph_vs_train_selected_baseline": tuned_comparison,
        },
        "gate": {
            "name": "2wiki_hotpot_config_zero_shot_reproduction_gate_v1",
            "matching_baseline": matching_baseline_name,
            "required_absolute_macro_mean_delta": MEAN_COMPLETE_RATE_GATE_DELTA,
            "maximum_allowed_macro_per_budget_regression": (
                MAX_PER_BUDGET_REGRESSION
            ),
            "observed_absolute_macro_mean_delta": mean_delta,
            "observed_absolute_macro_per_budget_deltas": per_budget_delta,
            "passed": gate_passed,
            "status": (
                "continue_to_musique_cross_dataset"
                if gate_passed
                else "diagnose_2wiki_transfer_before_musique"
            ),
            "interpretation": (
                "Passing means the train-selected Hotpot graph configuration "
                "transfers to a clean balanced 2Wiki controlled context pool. "
                "It does not establish full-corpus GraphRAG performance."
            ),
        },
        "content_contract": (
            "aggregate metrics, configurations, paths, and hashes only; no question "
            "IDs, questions, answers, titles, sentences, evidence text, or labels"
        ),
    }
