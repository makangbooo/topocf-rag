"""Leakage-safe inputs and aggregate evaluation for TopoCF text baselines.

The flat baseline sees the entire neutral topology serialization in one
single-tower reranker call.  The independent-edge baseline sees one typed edge
at a time and averages local logits; it never receives the other candidate
edges or a constrained assignment matrix.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
import statistics
from typing import Any, Final, Literal

from .method_data import (
    CounterfactualEvaluationExample,
    CounterfactualTrainingExample,
)
from .metrics import (
    ScoredCandidate,
    ScoredPair,
    macro_auroc,
    macro_pairwise_accuracy,
    score_margin_report,
)
from .serialization import SerializedTopology, serialize_evidence_topology
from .topology import EvidenceDocument, EvidenceTopology, TypedEdge


BaselineName = Literal["flat_cross_encoder", "independent_edge"]

BASELINE_NAMES: Final[tuple[BaselineName, ...]] = (
    "flat_cross_encoder",
    "independent_edge",
)
RERANKER_INPUT_VERSION: Final[str] = "topocf-qwen3-reranker-input-v1"
FLAT_RERANKER_INSTRUCTION: Final[str] = (
    "Determine whether the candidate typed evidence topology forms a complete, "
    "correctly connected multi-hop evidence chain that supports answering the "
    "question."
)
EDGE_RERANKER_INSTRUCTION: Final[str] = (
    "Determine whether this single directed typed edge is locally grounded by "
    "the shown endpoint document evidence and is relevant to answering the "
    "question."
)
RERANKER_PREFIX: Final[str] = (
    '<|im_start|>system\n'
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
RERANKER_SUFFIX: Final[str] = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


class MethodBaselineInvariantError(ValueError):
    """Raised when a baseline input or score violates the frozen protocol."""


@dataclass(frozen=True, slots=True)
class CandidateInputs:
    """One candidate represented by one flat or four local-edge inputs."""

    baseline: BaselineName
    inputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BaselineTrainingPair:
    pair_id: str
    qid: str
    positive: CandidateInputs
    negative: CandidateInputs


@dataclass(frozen=True, slots=True)
class BaselineEvaluationPair:
    pair_id: str
    qid: str
    positive_orbit: tuple[CandidateInputs, ...]
    negative_orbit: tuple[CandidateInputs, ...]


def _require_baseline(value: str) -> BaselineName:
    if value not in BASELINE_NAMES:
        raise MethodBaselineInvariantError(
            f"baseline must be one of {', '.join(BASELINE_NAMES)}"
        )
    return value  # type: ignore[return-value]


def resolve_replication_epoch(
    config: Mapping[str, Any],
    *,
    baseline: str,
    seed: int,
    learning_rate: float,
    replicate_selected: bool,
    smoke_only: bool = False,
    audit_only: bool = False,
) -> int | None:
    """Lock non-selection seeds to the already selected LR and epoch.

    Seed ``20260718`` is the sole hyperparameter-selection run.  Later seeds
    must reproduce its frozen learning rate and epoch instead of selecting a
    fresh checkpoint on the same inner-validation questions.
    """

    resolved_baseline = _require_baseline(baseline)
    if not isinstance(replicate_selected, bool):
        raise TypeError("replicate_selected must be bool")
    if smoke_only or audit_only:
        if replicate_selected:
            raise MethodBaselineInvariantError(
                "selected-config replication cannot be combined with smoke or audit"
            )
        return None
    selected = config.get("selected_configurations")
    if not isinstance(selected, Mapping):
        raise MethodBaselineInvariantError(
            "formal seed replication is locked until selected configurations are frozen"
        )
    selection_seed = selected.get("selection_seed")
    replication_seeds = selected.get("replication_seeds")
    if not isinstance(selection_seed, int) or isinstance(selection_seed, bool):
        raise MethodBaselineInvariantError("selection seed is invalid")
    if not isinstance(replication_seeds, list) or any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in replication_seeds
    ):
        raise MethodBaselineInvariantError("replication seeds are invalid")
    if seed == selection_seed:
        if replicate_selected:
            raise MethodBaselineInvariantError(
                "the selection seed cannot be rerun as an independent replication"
            )
        return None
    if seed not in replication_seeds:
        raise MethodBaselineInvariantError("seed is not a frozen replication seed")
    if not replicate_selected:
        raise MethodBaselineInvariantError(
            "non-selection seeds require --replicate-selected so validation cannot "
            "select another epoch"
        )
    by_baseline = selected.get("by_baseline")
    if not isinstance(by_baseline, Mapping):
        raise MethodBaselineInvariantError("selected baseline map is invalid")
    baseline_selection = by_baseline.get(resolved_baseline)
    if not isinstance(baseline_selection, Mapping):
        raise MethodBaselineInvariantError(
            f"no selected configuration for {resolved_baseline}"
        )
    expected_learning_rate = baseline_selection.get("learning_rate")
    epoch = baseline_selection.get("selected_epoch")
    if (
        not isinstance(expected_learning_rate, (int, float))
        or isinstance(expected_learning_rate, bool)
        or not math.isfinite(float(expected_learning_rate))
        or not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or epoch < 1
    ):
        raise MethodBaselineInvariantError("selected baseline settings are invalid")
    if not math.isclose(
        learning_rate,
        float(expected_learning_rate),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise MethodBaselineInvariantError(
            f"{resolved_baseline} replication requires learning_rate="
            f"{expected_learning_rate}"
        )
    return epoch


def format_reranker_input(
    question: str,
    document: str,
    *,
    instruction: str = FLAT_RERANKER_INSTRUCTION,
) -> str:
    """Format only the user portion expected between Qwen's prefix/suffix."""

    for value, field in (
        (instruction, "instruction"),
        (question, "question"),
        (document, "document"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise MethodBaselineInvariantError(f"{field} must be non-empty")
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {question}\n"
        f"<Document>: {document}"
    )


def reranker_token_ids(
    tokenizer: Any,
    formatted_input: str,
    *,
    max_length: int,
) -> tuple[int, ...]:
    """Tokenize without truncation and fail if the complete prompt is too long."""

    if not callable(tokenizer):
        raise TypeError("tokenizer must be callable")
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise MethodBaselineInvariantError("max_length must be positive")
    if not isinstance(formatted_input, str) or not formatted_input.strip():
        raise MethodBaselineInvariantError("formatted input must be non-empty")
    prefix_ids = tokenizer.encode(RERANKER_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False)
    encoded = tokenizer(
        formatted_input,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise MethodBaselineInvariantError("tokenizer returned no input_ids")
    content_ids = encoded["input_ids"]
    if not isinstance(content_ids, list) or any(
        not isinstance(token_id, int) or isinstance(token_id, bool)
        for token_id in content_ids
    ):
        raise MethodBaselineInvariantError("tokenizer input_ids must be integers")
    complete = tuple(prefix_ids) + tuple(content_ids) + tuple(suffix_ids)
    if len(complete) > max_length:
        raise MethodBaselineInvariantError(
            f"reranker input has {len(complete)} tokens and exceeds max_length="
            f"{max_length}; truncation is forbidden"
        )
    return complete


def _document_block(document: EvidenceDocument) -> str:
    lines = [
        f"[{document.alias}]",
        "title " + json.dumps(document.title, ensure_ascii=False, separators=(",", ":")),
    ]
    lines.extend(
        "sentence "
        + str(index)
        + " "
        + json.dumps(sentence, ensure_ascii=False, separators=(",", ":"))
        for index, sentence in enumerate(document.sentences)
    )
    return "\n".join(lines)


def render_local_edge(topology: EvidenceTopology, edge: TypedEdge) -> str:
    """Render one edge without observed/provenance or other candidate edges."""

    if not isinstance(topology, EvidenceTopology):
        raise TypeError("topology must be EvidenceTopology")
    if not isinstance(edge, TypedEdge) or edge not in topology.edges:
        raise MethodBaselineInvariantError("edge must belong to topology")
    documents = {document.alias: document for document in topology.documents}
    lines = ["[typed_edge]", f"{edge.relation} {edge.source} -> {edge.target}"]
    if edge.source != "q":
        lines.extend(("[source_document]", _document_block(documents[edge.source])))
    lines.extend(("[target_document]", _document_block(documents[edge.target])))
    return "\n".join(lines) + "\n"


def candidate_inputs(
    question: str,
    topology: EvidenceTopology,
    serialization: SerializedTopology,
    *,
    baseline: str,
) -> CandidateInputs:
    """Create one model-visible candidate under the selected baseline."""

    resolved = _require_baseline(baseline)
    if topology.question != question:
        raise MethodBaselineInvariantError("question and topology disagree")
    reconstructed = serialize_evidence_topology(topology)
    if reconstructed != serialization:
        raise MethodBaselineInvariantError("topology and serialization disagree")
    if resolved == "flat_cross_encoder":
        inputs = (
            format_reranker_input(
                question,
                serialization.text,
                instruction=FLAT_RERANKER_INSTRUCTION,
            ),
        )
    else:
        inputs = tuple(
            format_reranker_input(
                question,
                render_local_edge(topology, edge),
                instruction=EDGE_RERANKER_INSTRUCTION,
            )
            for edge in topology.edges
        )
        if len(inputs) != 4:
            raise MethodBaselineInvariantError(
                "primary T3 independent-edge candidate must contain four edges"
            )
    return CandidateInputs(baseline=resolved, inputs=inputs)


def build_baseline_training_pairs(
    examples: Sequence[CounterfactualTrainingExample],
    *,
    baseline: str,
) -> tuple[BaselineTrainingPair, ...]:
    resolved = _require_baseline(baseline)
    result = []
    seen: set[str] = set()
    for example in examples:
        if example.pair_id in seen:
            raise MethodBaselineInvariantError("training pair IDs must be unique")
        seen.add(example.pair_id)
        positive = candidate_inputs(
            example.question,
            example.positive_topology,
            example.positive,
            baseline=resolved,
        )
        negative = candidate_inputs(
            example.question,
            example.negative_topology,
            example.negative,
            baseline=resolved,
        )
        if len(positive.inputs) != len(negative.inputs):
            raise MethodBaselineInvariantError("paired candidates have unequal views")
        result.append(
            BaselineTrainingPair(
                pair_id=example.pair_id,
                qid=example.qid,
                positive=positive,
                negative=negative,
            )
        )
    return tuple(result)


def build_baseline_evaluation_pairs(
    examples: Sequence[CounterfactualEvaluationExample],
    *,
    baseline: str,
) -> tuple[BaselineEvaluationPair, ...]:
    resolved = _require_baseline(baseline)
    result = []
    seen: set[str] = set()
    for example in examples:
        if example.pair_id in seen:
            raise MethodBaselineInvariantError("evaluation pair IDs must be unique")
        seen.add(example.pair_id)
        if not (
            len(example.positive_topology_orbit)
            == len(example.negative_topology_orbit)
            == len(example.positive_orbit)
            == len(example.negative_orbit)
            == 24
        ):
            raise MethodBaselineInvariantError("evaluation requires an exact S4 orbit")
        positive = tuple(
            candidate_inputs(
                example.question, topology, serialization, baseline=resolved
            )
            for topology, serialization in zip(
                example.positive_topology_orbit,
                example.positive_orbit,
                strict=True,
            )
        )
        negative = tuple(
            candidate_inputs(
                example.question, topology, serialization, baseline=resolved
            )
            for topology, serialization in zip(
                example.negative_topology_orbit,
                example.negative_orbit,
                strict=True,
            )
        )
        result.append(
            BaselineEvaluationPair(
                pair_id=example.pair_id,
                qid=example.qid,
                positive_orbit=positive,
                negative_orbit=negative,
            )
        )
    return tuple(result)


def aggregate_candidate_logits(logits: Sequence[float]) -> float:
    if not logits:
        raise MethodBaselineInvariantError("candidate logits must not be empty")
    values = tuple(float(value) for value in logits)
    if any(not math.isfinite(value) for value in values):
        raise MethodBaselineInvariantError("candidate logits must be finite")
    return statistics.fmean(values)


def aggregate_orbit_scores(scores: Sequence[float]) -> float:
    if len(scores) != 24:
        raise MethodBaselineInvariantError("orbit aggregation requires 24 scores")
    values = tuple(float(value) for value in scores)
    if any(not math.isfinite(value) for value in values):
        raise MethodBaselineInvariantError("orbit scores must be finite")
    return statistics.fmean(values)


def _question_outcomes(pairs: Sequence[ScoredPair]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    identities: set[tuple[str, str]] = set()
    for pair in pairs:
        identity = (pair.question_id, pair.pair_id)
        if identity in identities:
            raise MethodBaselineInvariantError("scored pair IDs must be unique")
        identities.add(identity)
        if pair.positive_score > pair.negative_score:
            value = 1.0
        elif pair.positive_score == pair.negative_score:
            value = 0.5
        else:
            value = 0.0
        grouped[pair.question_id].append(value)
    return {
        qid: statistics.fmean(values) for qid, values in sorted(grouped.items())
    }


def question_bootstrap_pairwise_interval(
    pairs: Sequence[ScoredPair],
    *,
    repetitions: int = 2000,
    seed: int = 20260718,
) -> dict[str, Any]:
    """Percentile interval from resampling question-macro outcomes."""

    if repetitions < 1:
        raise MethodBaselineInvariantError("bootstrap repetitions must be positive")
    outcomes = _question_outcomes(pairs)
    if not outcomes:
        raise MethodBaselineInvariantError("bootstrap requires scored questions")
    values = tuple(outcomes.values())
    generator = random.Random(seed)
    samples = sorted(
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(repetitions)
    )

    def percentile(probability: float) -> float:
        if len(samples) == 1:
            return samples[0]
        position = probability * (len(samples) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return samples[lower]
        fraction = position - lower
        return samples[lower] * (1 - fraction) + samples[upper] * fraction

    return {
        "method": "question-level nonparametric percentile bootstrap",
        "seed": seed,
        "repetitions": repetitions,
        "low": percentile(0.025),
        "high": percentile(0.975),
    }


def baseline_metric_report(
    pairs: Sequence[ScoredPair],
    *,
    expected_question_ids: Sequence[str],
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 20260718,
) -> dict[str, Any]:
    """Return aggregate-only preregistered matched-pair metrics."""

    expected = tuple(expected_question_ids)
    candidates = tuple(
        candidate
        for pair in pairs
        for candidate in (
            ScoredCandidate(
                question_id=pair.question_id,
                candidate_id=pair.pair_id + ":positive",
                is_positive=True,
                score=pair.positive_score,
            ),
            ScoredCandidate(
                question_id=pair.question_id,
                candidate_id=pair.pair_id + ":negative",
                is_positive=False,
                score=pair.negative_score,
            ),
        )
    )
    pairwise = macro_pairwise_accuracy(pairs, expected_question_ids=expected)
    margins = score_margin_report(pairs, expected_question_ids=expected)
    auroc = macro_auroc(candidates, expected_question_ids=expected)
    return {
        "counts": {
            "question_count": len(expected),
            "pair_count": len(pairs),
        },
        "pairwise_accuracy": asdict(pairwise),
        "pairwise_bootstrap_95": question_bootstrap_pairwise_interval(
            pairs,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        ),
        "auroc": asdict(auroc),
        "score_margins": asdict(margins),
    }


def protocol_sha256() -> str:
    payload = {
        "input_version": RERANKER_INPUT_VERSION,
        "flat_instruction": FLAT_RERANKER_INSTRUCTION,
        "independent_edge_instruction": EDGE_RERANKER_INSTRUCTION,
        "prefix": RERANKER_PREFIX,
        "suffix": RERANKER_SUFFIX,
        "baselines": list(BASELINE_NAMES),
        "flat_aggregation": "one joint candidate logit",
        "independent_edge_aggregation": "arithmetic mean of four local edge logits",
        "orbit_aggregation": "arithmetic mean of 24 synchronized S4 candidate logits",
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
