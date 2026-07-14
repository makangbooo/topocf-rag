"""Leakage-safe graph-aware retrieval diagnostics for the HotpotQA kill test.

The graph is built only from literal cross-document title mentions.  Gold
supporting facts are used exclusively for evaluation; they never influence a
retrieval score, graph edge, hyperparameter, or tie break.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from .baselines import BM25Config, PerQuestionBM25
from .graph import (
    build_title_mention_edges,
    document_nodes,
    supporting_title_order,
    validate_hotpot_example,
)
from .title_normalization import normalize_title


REPORT_SCHEMA_VERSION = 1
DEFAULT_TOP_KS = (2, 3, 5)
RRF_CONSTANT = 60
MEAN_COMPLETE_RATE_GATE_DELTA = 0.02
MAX_PER_K_REGRESSION = 0.01


class GraphRetrievalInvariantError(ValueError):
    """Raised when a graph-retrieval input violates the frozen protocol."""


@dataclass(frozen=True, slots=True)
class RetrievalQuestion:
    """One in-memory retrieval instance; it is never serialized publicly."""

    qid: str
    dense_scores: tuple[float, ...]
    bm25_scores: tuple[float, ...]
    gold_indices: tuple[int, int]
    mention_edges: tuple[tuple[int, int], ...]

    @property
    def document_count(self) -> int:
        return len(self.dense_scores)


@dataclass(frozen=True, slots=True)
class GraphMethodConfig:
    """One train-selected, label-free graph scoring configuration."""

    family: str
    seed: str
    direction: str
    graph_weight: float | None = None
    restart_probability: float | None = None

    def __post_init__(self) -> None:
        if self.family not in {"one_hop_max", "personalized_pagerank"}:
            raise GraphRetrievalInvariantError("unsupported graph method family")
        if self.seed not in {"dense", "dense_bm25_rrf"}:
            raise GraphRetrievalInvariantError("unsupported graph seed")
        if self.direction not in {"outgoing", "incoming", "undirected"}:
            raise GraphRetrievalInvariantError("unsupported graph direction")
        if self.family == "one_hop_max":
            if (
                self.graph_weight is None
                or not math.isfinite(self.graph_weight)
                or not 0.0 < self.graph_weight < 1.0
            ):
                raise GraphRetrievalInvariantError(
                    "one-hop graph weight must be finite and between zero and one"
                )
            if self.restart_probability is not None:
                raise GraphRetrievalInvariantError(
                    "one-hop methods must not define a restart probability"
                )
        else:
            if (
                self.restart_probability is None
                or not math.isfinite(self.restart_probability)
                or not 0.0 < self.restart_probability < 1.0
            ):
                raise GraphRetrievalInvariantError(
                    "PageRank restart probability must be finite and between zero and one"
                )
            if self.graph_weight is not None:
                raise GraphRetrievalInvariantError(
                    "PageRank methods must not define a graph weight"
                )

    @property
    def name(self) -> str:
        if self.family == "one_hop_max":
            suffix = f"w{self.graph_weight:.2f}"
        else:
            suffix = f"restart{self.restart_probability:.2f}"
        return f"{self.family}__{self.seed}__{self.direction}__{suffix}"


def _validate_scores(scores: Sequence[float], *, field: str) -> tuple[float, ...]:
    if not scores:
        raise GraphRetrievalInvariantError(f"{field} must not be empty")
    numeric: list[float] = []
    for score in scores:
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise GraphRetrievalInvariantError(f"{field} must contain numbers")
        value = float(score)
        if not math.isfinite(value):
            raise GraphRetrievalInvariantError(f"{field} must contain finite values")
        numeric.append(value)
    return tuple(numeric)


def stable_rank(scores: Sequence[float]) -> tuple[int, ...]:
    """Rank descending by score and then ascending by context index."""

    numeric = _validate_scores(scores, field="scores")
    return tuple(sorted(range(len(numeric)), key=lambda index: (-numeric[index], index)))


def _normalize_positive(scores: Sequence[float]) -> tuple[float, ...]:
    numeric = _validate_scores(scores, field="positive scores")
    if any(score < 0.0 for score in numeric):
        raise GraphRetrievalInvariantError("positive scores must be non-negative")
    total = sum(numeric)
    if total == 0.0:
        uniform = 1.0 / len(numeric)
        return tuple(uniform for _ in numeric)
    return tuple(score / total for score in numeric)


def reciprocal_rank_scores(
    scores: Sequence[float], *, constant: int = RRF_CONSTANT
) -> tuple[float, ...]:
    """Convert arbitrary scores into a positive, scale-free rank distribution."""

    if not isinstance(constant, int) or isinstance(constant, bool) or constant < 1:
        raise GraphRetrievalInvariantError("RRF constant must be a positive integer")
    ranking = stable_rank(scores)
    values = [0.0] * len(ranking)
    for rank, index in enumerate(ranking, start=1):
        values[index] = 1.0 / (constant + rank)
    return _normalize_positive(values)


def rrf_fusion_scores(
    dense_scores: Sequence[float],
    bm25_scores: Sequence[float],
    *,
    constant: int = RRF_CONSTANT,
) -> tuple[float, ...]:
    """Fuse dense and lexical rankings without using either score scale."""

    if len(dense_scores) != len(bm25_scores):
        raise GraphRetrievalInvariantError("RRF score vectors must have equal length")
    dense = reciprocal_rank_scores(dense_scores, constant=constant)
    lexical = reciprocal_rank_scores(bm25_scores, constant=constant)
    return _normalize_positive(
        [dense[index] + lexical[index] for index in range(len(dense))]
    )


def prepare_retrieval_question(
    example: Mapping[str, Any], dense_scores: Sequence[float]
) -> RetrievalQuestion:
    """Build label-free graph and scorers, then attach gold indices for metrics."""

    validate_hotpot_example(example)
    if example.get("type") != "bridge":
        raise GraphRetrievalInvariantError("kill test requires bridge questions")
    qid = example.get("_id")
    if not isinstance(qid, str) or not qid:
        raise GraphRetrievalInvariantError("question ID must be non-empty")

    dense = _validate_scores(dense_scores, field="dense scores")
    documents = document_nodes(example)
    if len(dense) != len(documents):
        raise GraphRetrievalInvariantError(
            "dense score count must equal context document count"
        )

    serialized = {
        str(document.index): document.serialized_text for document in documents
    }
    bm25_index = PerQuestionBM25.fit(serialized, config=BM25Config())
    bm25_by_id = bm25_index.score(str(example["question"]))
    bm25 = tuple(bm25_by_id[str(index)] for index in range(len(documents)))

    support_titles = supporting_title_order(example)
    if len(support_titles) != 2:
        raise GraphRetrievalInvariantError(
            "bridge kill test requires exactly two supporting documents"
        )
    index_by_title = {
        normalize_title(document.title): document.index for document in documents
    }
    try:
        gold = tuple(index_by_title[title] for title in support_titles)
    except KeyError as error:
        raise GraphRetrievalInvariantError(
            "supporting document is missing from official context"
        ) from error
    if len(set(gold)) != 2:
        raise GraphRetrievalInvariantError(
            "supporting documents must map to distinct context indices"
        )

    mention_edges = tuple(
        (edge.source_index, edge.target_index)
        for edge in build_title_mention_edges(documents)
    )
    return RetrievalQuestion(
        qid=qid,
        dense_scores=dense,
        bm25_scores=bm25,
        gold_indices=(gold[0], gold[1]),
        mention_edges=mention_edges,
    )


def _seed_scores(question: RetrievalQuestion, seed: str) -> tuple[float, ...]:
    if seed == "dense":
        return reciprocal_rank_scores(question.dense_scores)
    if seed == "dense_bm25_rrf":
        return rrf_fusion_scores(question.dense_scores, question.bm25_scores)
    raise GraphRetrievalInvariantError("unsupported graph seed")


def _propagation_arcs(
    edges: Sequence[tuple[int, int]], direction: str
) -> tuple[tuple[int, int], ...]:
    if direction == "outgoing":
        arcs = set(edges)
    elif direction == "incoming":
        arcs = {(target, source) for source, target in edges}
    elif direction == "undirected":
        arcs = set(edges).union((target, source) for source, target in edges)
    else:
        raise GraphRetrievalInvariantError("unsupported graph direction")
    return tuple(sorted(arcs))


def one_hop_max_scores(
    seed_scores: Sequence[float],
    mention_edges: Sequence[tuple[int, int]],
    *,
    direction: str,
    graph_weight: float,
) -> tuple[float, ...]:
    """Boost a node from the strongest one-hop predecessor under a fixed mode."""

    seed = _normalize_positive(seed_scores)
    if not math.isfinite(graph_weight) or not 0.0 < graph_weight < 1.0:
        raise GraphRetrievalInvariantError(
            "graph weight must be finite and between zero and one"
        )
    propagated = list(seed)
    for source, target in _propagation_arcs(mention_edges, direction):
        if not 0 <= source < len(seed) or not 0 <= target < len(seed):
            raise GraphRetrievalInvariantError("mention edge index is out of range")
        propagated[target] = max(propagated[target], seed[source])
    return tuple(
        (1.0 - graph_weight) * seed[index]
        + graph_weight * propagated[index]
        for index in range(len(seed))
    )


def personalized_pagerank_scores(
    seed_scores: Sequence[float],
    mention_edges: Sequence[tuple[int, int]],
    *,
    direction: str,
    restart_probability: float,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> tuple[float, ...]:
    """Run deterministic personalized PageRank over the title-mention graph."""

    seed = _normalize_positive(seed_scores)
    if (
        not math.isfinite(restart_probability)
        or not 0.0 < restart_probability < 1.0
    ):
        raise GraphRetrievalInvariantError(
            "restart probability must be finite and between zero and one"
        )
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise GraphRetrievalInvariantError("tolerance must be finite and positive")
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 1
    ):
        raise GraphRetrievalInvariantError("max iterations must be positive")

    adjacency: list[list[int]] = [[] for _ in seed]
    for source, target in _propagation_arcs(mention_edges, direction):
        if not 0 <= source < len(seed) or not 0 <= target < len(seed):
            raise GraphRetrievalInvariantError("mention edge index is out of range")
        adjacency[source].append(target)

    probability = list(seed)
    walk_weight = 1.0 - restart_probability
    for _ in range(max_iterations):
        updated = [restart_probability * value for value in seed]
        dangling_mass = 0.0
        for source, targets in enumerate(adjacency):
            if not targets:
                dangling_mass += walk_weight * probability[source]
                continue
            share = walk_weight * probability[source] / len(targets)
            for target in targets:
                updated[target] += share
        if dangling_mass:
            for index, value in enumerate(seed):
                updated[index] += dangling_mass * value
        delta = sum(
            abs(updated[index] - probability[index])
            for index in range(len(probability))
        )
        probability = updated
        if delta <= tolerance:
            break
    return _normalize_positive(probability)


def graph_method_scores(
    question: RetrievalQuestion, config: GraphMethodConfig
) -> tuple[float, ...]:
    seed = _seed_scores(question, config.seed)
    if config.family == "one_hop_max":
        assert config.graph_weight is not None
        return one_hop_max_scores(
            seed,
            question.mention_edges,
            direction=config.direction,
            graph_weight=config.graph_weight,
        )
    assert config.restart_probability is not None
    return personalized_pagerank_scores(
        seed,
        question.mention_edges,
        direction=config.direction,
        restart_probability=config.restart_probability,
    )


def graph_candidate_grid() -> tuple[GraphMethodConfig, ...]:
    """Return the fully predeclared train-only model-selection grid."""

    candidates: list[GraphMethodConfig] = []
    for seed in ("dense", "dense_bm25_rrf"):
        for direction in ("outgoing", "incoming", "undirected"):
            for graph_weight in (0.25, 0.5, 0.75):
                candidates.append(
                    GraphMethodConfig(
                        family="one_hop_max",
                        seed=seed,
                        direction=direction,
                        graph_weight=graph_weight,
                    )
                )
            for restart_probability in (0.2, 0.5, 0.8):
                candidates.append(
                    GraphMethodConfig(
                        family="personalized_pagerank",
                        seed=seed,
                        direction=direction,
                        restart_probability=restart_probability,
                    )
                )
    return tuple(candidates)


def _validate_top_ks(top_ks: Sequence[int]) -> tuple[int, ...]:
    if not top_ks or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in top_ks
    ):
        raise GraphRetrievalInvariantError("top_ks must contain positive integers")
    normalized = tuple(sorted(set(top_ks)))
    if len(normalized) != len(top_ks):
        raise GraphRetrievalInvariantError("top_ks must be unique")
    return normalized


def evaluate_rankings(
    questions: Sequence[RetrievalQuestion],
    rankings: Sequence[Sequence[int]],
    *,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> dict[str, Any]:
    """Aggregate question-macro evidence retrieval metrics."""

    normalized_top_ks = _validate_top_ks(top_ks)
    if not questions or len(questions) != len(rankings):
        raise GraphRetrievalInvariantError(
            "questions and rankings must be non-empty and equally sized"
        )

    complete_counts = {top_k: 0 for top_k in normalized_top_ks}
    at_least_one_counts = {top_k: 0 for top_k in normalized_top_ks}
    no_gold_counts = {top_k: 0 for top_k in normalized_top_ks}
    support_sums = {top_k: 0 for top_k in normalized_top_ks}
    reciprocal_full_evidence_ranks: list[float] = []

    for question, ranking_value in zip(questions, rankings, strict=True):
        ranking = tuple(ranking_value)
        if set(ranking) != set(range(question.document_count)):
            raise GraphRetrievalInvariantError(
                "ranking must be a permutation of context document indices"
            )
        rank_by_index = {index: rank for rank, index in enumerate(ranking, start=1)}
        full_evidence_rank = max(
            rank_by_index[question.gold_indices[0]],
            rank_by_index[question.gold_indices[1]],
        )
        reciprocal_full_evidence_ranks.append(1.0 / full_evidence_rank)
        gold = set(question.gold_indices)
        for top_k in normalized_top_ks:
            retrieved = set(ranking[: min(top_k, len(ranking))])
            found = len(gold.intersection(retrieved))
            support_sums[top_k] += found
            complete_counts[top_k] += found == 2
            at_least_one_counts[top_k] += found >= 1
            no_gold_counts[top_k] += found == 0

    count = len(questions)
    by_top_k: dict[str, Any] = {}
    for top_k in normalized_top_ks:
        by_top_k[str(top_k)] = {
            "complete_gold_evidence_count": complete_counts[top_k],
            "complete_gold_evidence_rate": complete_counts[top_k] / count,
            "at_least_one_gold_count": at_least_one_counts[top_k],
            "at_least_one_gold_rate": at_least_one_counts[top_k] / count,
            "no_gold_count": no_gold_counts[top_k],
            "no_gold_rate": no_gold_counts[top_k] / count,
            "support_document_recall": support_sums[top_k] / (2 * count),
        }
    return {
        "question_count": count,
        "by_top_k": by_top_k,
        "mean_complete_gold_evidence_rate": sum(
            by_top_k[str(top_k)]["complete_gold_evidence_rate"]
            for top_k in normalized_top_ks
        )
        / len(normalized_top_ks),
        "full_evidence_mrr": sum(reciprocal_full_evidence_ranks) / count,
    }


def _baseline_scores(question: RetrievalQuestion, method: str) -> tuple[float, ...]:
    if method == "dense":
        return question.dense_scores
    if method == "bm25":
        return question.bm25_scores
    if method == "dense_bm25_rrf":
        return rrf_fusion_scores(question.dense_scores, question.bm25_scores)
    raise GraphRetrievalInvariantError("unsupported baseline method")


def evaluate_baseline(
    questions: Sequence[RetrievalQuestion],
    method: str,
    *,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> dict[str, Any]:
    return evaluate_rankings(
        questions,
        [stable_rank(_baseline_scores(question, method)) for question in questions],
        top_ks=top_ks,
    )


def evaluate_graph_method(
    questions: Sequence[RetrievalQuestion],
    config: GraphMethodConfig,
    *,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> dict[str, Any]:
    return evaluate_rankings(
        questions,
        [stable_rank(graph_method_scores(question, config)) for question in questions],
        top_ks=top_ks,
    )


def _selection_key(metrics: Mapping[str, Any], top_ks: Sequence[int]) -> tuple[float, ...]:
    normalized_top_ks = _validate_top_ks(top_ks)
    by_top_k = metrics["by_top_k"]
    return (
        float(metrics["mean_complete_gold_evidence_rate"]),
        *(float(by_top_k[str(top_k)]["complete_gold_evidence_rate"])
          for top_k in reversed(normalized_top_ks)),
        float(metrics["full_evidence_mrr"]),
    )


def select_best_result(
    named_results: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> tuple[str, Mapping[str, Any]]:
    """Select by train metrics only; explicit order resolves exact ties."""

    if not named_results:
        raise GraphRetrievalInvariantError("named results must not be empty")
    best_name, best_metrics = named_results[0]
    best_key = _selection_key(best_metrics, top_ks)
    for name, metrics in named_results[1:]:
        key = _selection_key(metrics, top_ks)
        if key > best_key:
            best_name, best_metrics, best_key = name, metrics, key
    return best_name, best_metrics


def graph_diagnostics(
    questions: Sequence[RetrievalQuestion],
    *,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> dict[str, Any]:
    """Report graph coverage and dense-failure recovery opportunity."""

    normalized_top_ks = _validate_top_ks(top_ks)
    if not questions:
        raise GraphRetrievalInvariantError("questions must not be empty")
    any_direction_count = 0
    directed_count = 0
    edge_count = 0
    opportunities = {
        top_k: {"partial_failure_count": 0, "gold_edge_recoverable_count": 0}
        for top_k in normalized_top_ks
    }
    for question in questions:
        edges = set(question.mention_edges)
        edge_count += len(edges)
        first, second = question.gold_indices
        forward = (first, second) in edges
        reverse = (second, first) in edges
        directed_count += forward
        any_direction_count += forward or reverse
        ranking = stable_rank(question.dense_scores)
        gold = {first, second}
        for top_k in normalized_top_ks:
            retrieved = set(ranking[:top_k])
            found = gold.intersection(retrieved)
            if len(found) != 1:
                continue
            opportunities[top_k]["partial_failure_count"] += 1
            present = next(iter(found))
            missing = next(iter(gold.difference(found)))
            if (present, missing) in edges or (missing, present) in edges:
                opportunities[top_k]["gold_edge_recoverable_count"] += 1

    count = len(questions)
    by_top_k: dict[str, Any] = {}
    for top_k in normalized_top_ks:
        partial = opportunities[top_k]["partial_failure_count"]
        recoverable = opportunities[top_k]["gold_edge_recoverable_count"]
        by_top_k[str(top_k)] = {
            "dense_partial_failure_count": partial,
            "gold_edge_recoverable_count": recoverable,
            "recoverable_rate_among_dense_partial_failures": (
                recoverable / partial if partial else None
            ),
        }
    return {
        "question_count": count,
        "mean_title_mention_edge_count": edge_count / count,
        "gold_pair_ordered_direction_count": directed_count,
        "gold_pair_ordered_direction_rate": directed_count / count,
        "gold_pair_any_direction_count": any_direction_count,
        "gold_pair_any_direction_rate": any_direction_count / count,
        "dense_recovery_opportunity_by_top_k": by_top_k,
    }


def _gate_decision(
    baseline_name: str,
    baseline_metrics: Mapping[str, Any],
    graph_metrics: Mapping[str, Any],
    *,
    top_ks: Sequence[int],
) -> dict[str, Any]:
    normalized_top_ks = _validate_top_ks(top_ks)
    mean_delta = (
        float(graph_metrics["mean_complete_gold_evidence_rate"])
        - float(baseline_metrics["mean_complete_gold_evidence_rate"])
    )
    per_k_delta = {
        str(top_k): (
            float(
                graph_metrics["by_top_k"][str(top_k)][
                    "complete_gold_evidence_rate"
                ]
            )
            - float(
                baseline_metrics["by_top_k"][str(top_k)][
                    "complete_gold_evidence_rate"
                ]
            )
        )
        for top_k in normalized_top_ks
    }
    passed = (
        mean_delta >= MEAN_COMPLETE_RATE_GATE_DELTA
        and min(per_k_delta.values()) >= -MAX_PER_K_REGRESSION
    )
    return {
        "name": "hotpot_title_graph_retrieval_feasibility_gate_v1",
        "baseline_selected_on_train": baseline_name,
        "primary_metric": "mean_complete_gold_evidence_rate_over_k_2_3_5",
        "required_absolute_mean_delta": MEAN_COMPLETE_RATE_GATE_DELTA,
        "maximum_allowed_absolute_per_k_regression": MAX_PER_K_REGRESSION,
        "observed_absolute_mean_delta": mean_delta,
        "observed_absolute_per_k_deltas": per_k_delta,
        "passed": passed,
        "status": (
            "continue_to_2wiki_and_musique"
            if passed
            else "stop_and_redesign_graph_retrieval"
        ),
        "interpretation": (
            "This gate tests only whether a literal title-mention graph improves "
            "retrieval inside the official HotpotQA ten-document context pool. "
            "It is not a full-corpus or final TopoCF-RAG result."
        ),
    }


def build_kill_test_report(
    train_questions: Sequence[RetrievalQuestion],
    dev_questions: Sequence[RetrievalQuestion],
    *,
    provenance: Mapping[str, Any],
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> dict[str, Any]:
    """Tune on train, evaluate the selected graph method once on dev."""

    normalized_top_ks = _validate_top_ks(top_ks)
    if not train_questions or not dev_questions:
        raise GraphRetrievalInvariantError("train and dev questions must not be empty")
    train_qids = {question.qid for question in train_questions}
    dev_qids = {question.qid for question in dev_questions}
    if len(train_qids) != len(train_questions) or len(dev_qids) != len(dev_questions):
        raise GraphRetrievalInvariantError("question IDs must be unique within a split")
    if train_qids.intersection(dev_qids):
        raise GraphRetrievalInvariantError("train and dev question IDs must be disjoint")

    baseline_names = ("dense", "bm25", "dense_bm25_rrf")
    train_baselines = {
        name: evaluate_baseline(train_questions, name, top_ks=normalized_top_ks)
        for name in baseline_names
    }
    dev_baselines = {
        name: evaluate_baseline(dev_questions, name, top_ks=normalized_top_ks)
        for name in baseline_names
    }
    selected_baseline_name, selected_train_baseline = select_best_result(
        list(train_baselines.items()), top_ks=normalized_top_ks
    )

    candidates = graph_candidate_grid()
    train_graph_results: list[tuple[str, Mapping[str, Any]]] = []
    config_by_name: dict[str, GraphMethodConfig] = {}
    for config in candidates:
        config_by_name[config.name] = config
        train_graph_results.append(
            (
                config.name,
                evaluate_graph_method(
                    train_questions, config, top_ks=normalized_top_ks
                ),
            )
        )
    selected_graph_name, selected_train_graph = select_best_result(
        train_graph_results, top_ks=normalized_top_ks
    )
    selected_config = config_by_name[selected_graph_name]
    selected_dev_graph = evaluate_graph_method(
        dev_questions, selected_config, top_ks=normalized_top_ks
    )
    selected_dev_baseline = dev_baselines[selected_baseline_name]

    leaderboard = sorted(
        train_graph_results,
        key=lambda item: _selection_key(item[1], normalized_top_ks),
        reverse=True,
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "task": "label_free_graph_aware_retrieval_kill_test",
        "scope": "hotpotqa_official_ten_document_context_pool_bridge_questions",
        "protocol": {
            "top_ks": list(normalized_top_ks),
            "train_only_hyperparameter_selection": True,
            "dev_used_for_model_selection": False,
            "stable_tie_break": "ascending official context index",
            "graph_nodes": "official context documents",
            "graph_edges": "literal directed cross-document title mentions",
            "gold_labels_used_for": "aggregate evaluation only",
            "rrf_constant": RRF_CONSTANT,
            "graph_candidate_count": len(candidates),
            "selection_order": (
                "mean complete evidence rate; complete rate at k=5,3,2; "
                "full-evidence MRR; declared grid order"
            ),
        },
        "provenance": dict(provenance),
        "selection": {
            "selected_baseline": selected_baseline_name,
            "selected_graph_method": selected_graph_name,
            "selected_graph_config": asdict(selected_config),
        },
        "train": {
            "question_count": len(train_questions),
            "graph_diagnostics": graph_diagnostics(
                train_questions, top_ks=normalized_top_ks
            ),
            "baselines": train_baselines,
            "selected_baseline_metrics": selected_train_baseline,
            "selected_graph_metrics": selected_train_graph,
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
            "graph_diagnostics": graph_diagnostics(
                dev_questions, top_ks=normalized_top_ks
            ),
            "baselines": dev_baselines,
            "selected_baseline_metrics": selected_dev_baseline,
            "selected_graph_metrics": selected_dev_graph,
        },
        "gate": _gate_decision(
            selected_baseline_name,
            selected_dev_baseline,
            selected_dev_graph,
            top_ks=normalized_top_ks,
        ),
        "content_contract": (
            "aggregate metrics, method configurations, and provenance hashes only; "
            "no question IDs, questions, answers, titles, sentences, or supporting facts"
        ),
    }
