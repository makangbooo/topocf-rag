"""Content-safe Phase 1 baseline reconstruction, scoring, and evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import ijson
import numpy as np

from .baselines import BM25Config, PerQuestionBM25
from .graph import validate_hotpot_example
from .metrics import (
    ScoredCandidate,
    ScoredPair,
    evaluate_ranking,
    macro_pairwise_accuracy,
    score_margin_report,
)
from .serialization import (
    SERIALIZER_VERSION,
    SerializationTokenStats,
    SerializedTopology,
    count_serialization_tokens,
    serialize_evidence_topology,
)
from .title_normalization import contains_title_mention
from .topology import (
    EvidenceDocument,
    EvidenceTopology,
    TypedEdge,
    validate_matched_pair,
)


EVALUATION_SCHEMA_VERSION = 1
CACHE_SCHEMA_VERSION = 1

_STRATUM_RENAMES = {
    "natural_rewire": "all_observed_rewire",
}
_VARIANT_RENAMES = {
    "t3_all_real": "t3_all_observed",
}


class EvaluationInvariantError(ValueError):
    """Raised when a manifest, source record, score, or cache is inconsistent."""


class DenseTextEmbedder(Protocol):
    def encode(self, texts: Sequence[str]) -> Any:
        """Return one dense vector per input string."""


@dataclass(frozen=True, slots=True)
class PreparedCandidate:
    topology: EvidenceTopology
    serialization: SerializedTopology
    token_stats: SerializationTokenStats
    canonical_signature: str
    edge_reality: str


@dataclass(frozen=True, slots=True)
class PreparedPair:
    pair_id: str
    base_id: str
    qid: str
    stratum: str
    variant: str
    positive: PreparedCandidate
    negative: PreparedCandidate
    canonical_class: str


@dataclass(frozen=True, slots=True)
class PreparedSplit:
    official_split: str
    frozen_question_count: int
    questions: Mapping[str, str]
    pairs: tuple[PreparedPair, ...]


@dataclass(frozen=True, slots=True)
class ScoreBundle:
    bm25_scores: Mapping[tuple[str, str], float]
    dense_scores: Mapping[tuple[str, str], float]
    bm25_fit_question_count: int
    dense_question_embedding_count: int
    dense_candidate_embedding_count: int
    elapsed_seconds: float
    cache_hit: bool = False


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_dump(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_pair_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("pairs"), list):
        raise EvaluationInvariantError("pair manifest must contain a pairs list")
    if payload.get("schema_version") != 1:
        raise EvaluationInvariantError("unsupported pair manifest schema version")
    if not isinstance(payload.get("official_split"), str):
        raise EvaluationInvariantError("pair manifest must name an official split")
    return payload


def manifest_qids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list):
        raise EvaluationInvariantError("pair manifest must contain a pairs list")
    qids: set[str] = set()
    pair_ids: set[str] = set()
    for record in pairs:
        if not isinstance(record, Mapping):
            raise EvaluationInvariantError("pair manifest records must be objects")
        qid = record.get("qid")
        pair_id = record.get("pair_id")
        if not isinstance(qid, str) or not qid:
            raise EvaluationInvariantError("pair record qid must be non-empty")
        if not isinstance(pair_id, str) or not pair_id:
            raise EvaluationInvariantError("pair record ID must be non-empty")
        if pair_id in pair_ids:
            raise EvaluationInvariantError("pair manifest contains duplicate pair IDs")
        pair_ids.add(pair_id)
        qids.add(qid)
    return tuple(sorted(qids))


def stream_selected_source_examples(
    source_path: str | Path, qids: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Stream a HotpotQA array and retain only manifest-referenced records."""

    wanted = set(qids)
    if any(not isinstance(qid, str) or not qid for qid in wanted):
        raise EvaluationInvariantError("wanted qids must be non-empty strings")
    selected: dict[str, dict[str, Any]] = {}
    with Path(source_path).open("rb") as stream:
        for example in ijson.items(stream, "item"):
            qid = example.get("_id") if isinstance(example, Mapping) else None
            if qid not in wanted:
                continue
            if qid in selected:
                raise EvaluationInvariantError("source contains a duplicate selected qid")
            validate_hotpot_example(example)
            selected[qid] = example
            if len(selected) == len(wanted):
                break
    missing = wanted.difference(selected)
    if missing:
        raise EvaluationInvariantError(
            f"source is missing {len(missing)} manifest question(s)"
        )
    return selected


def _aliases(record: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    mapping = record.get("neutral_alias_mapping")
    if not isinstance(mapping, Mapping):
        raise EvaluationInvariantError("pair alias mapping must be an object")
    expected = tuple(f"d{index}" for index in range(4))
    if tuple(mapping) != expected:
        raise EvaluationInvariantError("pair aliases must be ordered d0 through d3")
    result: list[tuple[str, int]] = []
    for alias in expected:
        context_index = mapping[alias]
        if not isinstance(context_index, int) or isinstance(context_index, bool):
            raise EvaluationInvariantError("context indices must be integers")
        result.append((alias, context_index))
    if len({index for _alias, index in result}) != len(result):
        raise EvaluationInvariantError("neutral aliases must map to distinct documents")
    context_indices = record.get("context_indices")
    if context_indices != [index for _alias, index in result]:
        raise EvaluationInvariantError("context indices disagree with alias mapping")
    return tuple(result)


def _documents(
    example: Mapping[str, Any], record: Mapping[str, Any]
) -> tuple[EvidenceDocument, ...]:
    aliases = _aliases(record)
    selections = record.get("sentence_indices")
    if not isinstance(selections, Mapping) or tuple(selections) != tuple(
        alias for alias, _index in aliases
    ):
        raise EvaluationInvariantError("sentence selections must follow alias order")
    context = example["context"]
    documents: list[EvidenceDocument] = []
    for alias, context_index in aliases:
        if not 0 <= context_index < len(context):
            raise EvaluationInvariantError("context index is out of range")
        title, sentences = context[context_index]
        indices = selections[alias]
        if (
            not isinstance(indices, list)
            or not indices
            or any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(sentences)
                for index in indices
            )
            or indices != sorted(set(indices))
        ):
            raise EvaluationInvariantError("selected sentence indices are invalid")
        documents.append(
            EvidenceDocument(
                alias=alias,
                title=title,
                sentences=tuple(sentences[index] for index in indices),
            )
        )
    return tuple(documents)


def _endpoint_order(endpoint: str) -> int:
    if endpoint == "q":
        return -1
    if not endpoint.startswith("d") or not endpoint[1:].isdigit():
        raise EvaluationInvariantError("edge endpoint must use a neutral alias")
    return int(endpoint[1:])


def _edges(member: Any) -> tuple[TypedEdge, ...]:
    if not isinstance(member, Mapping) or not isinstance(member.get("typed_edges"), list):
        raise EvaluationInvariantError("pair member must contain typed edges")
    edges: list[TypedEdge] = []
    for payload in member["typed_edges"]:
        if not isinstance(payload, Mapping):
            raise EvaluationInvariantError("typed edge must be an object")
        edges.append(
            TypedEdge(
                relation=payload.get("relation"),
                source=payload.get("source"),
                target=payload.get("target"),
                observed=payload.get("observed"),
            )
        )
    relation_order = {"retrieval": 0, "title_mention": 1}
    return tuple(
        sorted(
            edges,
            key=lambda edge: (
                relation_order[edge.relation],
                _endpoint_order(edge.source),
                _endpoint_order(edge.target),
            ),
        )
    )


def canonical_signature_text(topology: EvidenceTopology) -> str:
    node_count, edges = topology.canonical_signature
    encoded_edges = "|".join(
        f"{relation}:{source}->{target}" for relation, source, target in edges
    )
    return f"n={node_count}|{encoded_edges}"


def edge_reality(topology: EvidenceTopology) -> str:
    observed_count = sum(edge.observed for edge in topology.edges)
    if observed_count == len(topology.edges):
        return "all_observed"
    if observed_count == 0:
        return "all_counterfactual"
    return f"mixed_{observed_count}_observed_{len(topology.edges) - observed_count}_counterfactual"


def _validate_observed_title_mentions(topology: EvidenceTopology) -> None:
    """Require selected textual evidence for every observed mention edge."""

    documents = {document.alias: document for document in topology.documents}
    for edge in topology.edges:
        if edge.relation != "title_mention" or not edge.observed:
            continue
        source = documents[edge.source]
        target = documents[edge.target]
        if not any(
            contains_title_mention(sentence, target.title)
            for sentence in source.sentences
        ):
            raise EvaluationInvariantError(
                "observed title-mention edge lacks selected sentence evidence"
            )


def _prepared_candidate(
    topology: EvidenceTopology,
    tokenizer: Any,
    *,
    max_length: int,
    token_cache: dict[str, SerializationTokenStats],
    fail_on_truncation: bool,
) -> PreparedCandidate:
    serialization = serialize_evidence_topology(topology)
    token_stats = token_cache.get(serialization.sha256)
    if token_stats is None:
        token_stats = count_serialization_tokens(
            serialization, tokenizer, max_length=max_length
        )
        token_cache[serialization.sha256] = token_stats
    if fail_on_truncation and token_stats.truncated:
        raise EvaluationInvariantError(
            "serialized topology exceeds max_length; truncation is disabled"
        )
    return PreparedCandidate(
        topology=topology,
        serialization=serialization,
        token_stats=token_stats,
        canonical_signature=canonical_signature_text(topology),
        edge_reality=edge_reality(topology),
    )


def prepare_manifest_pairs(
    manifest: Mapping[str, Any],
    examples: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int = 512,
    fail_on_truncation: bool = True,
) -> PreparedSplit:
    """Reconstruct, validate, serialize, and tokenize every manifest pair."""

    qids = manifest_qids(manifest)
    frozen_question_count = manifest.get("frozen_question_count")
    if (
        not isinstance(frozen_question_count, int)
        or isinstance(frozen_question_count, bool)
        or frozen_question_count < len(qids)
    ):
        raise EvaluationInvariantError(
            "manifest frozen question count must cover every pair-eligible qid"
        )
    if set(examples) != set(qids):
        raise EvaluationInvariantError("source example IDs do not match manifest IDs")
    questions: dict[str, str] = {}
    prepared: list[PreparedPair] = []
    token_cache: dict[str, SerializationTokenStats] = {}

    for record in manifest["pairs"]:
        qid = record["qid"]
        example = examples[qid]
        if example.get("_id") != qid:
            raise EvaluationInvariantError("source record ID disagrees with manifest qid")
        question = example.get("question")
        if not isinstance(question, str) or not question.strip():
            raise EvaluationInvariantError("source question must be non-empty")
        previous_question = questions.setdefault(qid, question)
        if previous_question != question:
            raise EvaluationInvariantError("qid maps to conflicting question text")

        documents = _documents(example, record)
        positive_topology = EvidenceTopology(
            question_id=qid,
            question=question,
            documents=documents,
            edges=_edges(record.get("positive")),
        )
        negative_topology = EvidenceTopology(
            question_id=qid,
            question=question,
            documents=documents,
            edges=_edges(record.get("negative")),
        )
        _validate_observed_title_mentions(positive_topology)
        _validate_observed_title_mentions(negative_topology)
        validate_matched_pair(positive_topology, negative_topology)
        positive = _prepared_candidate(
            positive_topology,
            tokenizer,
            max_length=max_length,
            token_cache=token_cache,
            fail_on_truncation=fail_on_truncation,
        )
        negative = _prepared_candidate(
            negative_topology,
            tokenizer,
            max_length=max_length,
            token_cache=token_cache,
            fail_on_truncation=fail_on_truncation,
        )
        validate_matched_pair(
            positive_topology,
            negative_topology,
            positive_token_count=positive.token_stats.untruncated_token_count,
            negative_token_count=negative.token_stats.untruncated_token_count,
        )
        class_payload = [positive.canonical_signature, negative.canonical_signature]
        prepared.append(
            PreparedPair(
                pair_id=record["pair_id"],
                base_id=record["base_id"],
                qid=qid,
                stratum=_STRATUM_RENAMES.get(record["stratum"], record["stratum"]),
                variant=_VARIANT_RENAMES.get(record["variant"], record["variant"]),
                positive=positive,
                negative=negative,
                canonical_class="canonical-" + stable_hash(class_payload)[:16],
            )
        )

    return PreparedSplit(
        official_split=str(manifest["official_split"]),
        frozen_question_count=frozen_question_count,
        questions=questions,
        pairs=tuple(prepared),
    )


def prepare_split_from_files(
    manifest_path: str | Path,
    source_path: str | Path,
    tokenizer: Any,
    *,
    max_length: int = 512,
    fail_on_truncation: bool = True,
) -> tuple[PreparedSplit, dict[str, str]]:
    manifest = load_pair_manifest(manifest_path)
    source_digest = sha256_file(source_path)
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("source_sha256") != source_digest:
        raise EvaluationInvariantError("source SHA256 does not match pair manifest")
    examples = stream_selected_source_examples(source_path, manifest_qids(manifest))
    prepared = prepare_manifest_pairs(
        manifest,
        examples,
        tokenizer,
        max_length=max_length,
        fail_on_truncation=fail_on_truncation,
    )
    return prepared, {
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": source_digest,
    }


def _candidate_inventory(
    prepared: PreparedSplit,
) -> tuple[
    dict[tuple[str, str], PreparedCandidate],
    dict[str, PreparedCandidate],
    dict[tuple[str, str], set[str]],
]:
    by_question: dict[tuple[str, str], PreparedCandidate] = {}
    by_sha: dict[str, PreparedCandidate] = {}
    roles: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pair in prepared.pairs:
        for role, candidate in (("positive", pair.positive), ("negative", pair.negative)):
            sha = candidate.serialization.sha256
            key = (pair.qid, sha)
            previous = by_question.setdefault(key, candidate)
            if previous.serialization.text != candidate.serialization.text:
                raise EvaluationInvariantError("serialization SHA256 collision")
            global_previous = by_sha.setdefault(sha, candidate)
            if global_previous.serialization.text != candidate.serialization.text:
                raise EvaluationInvariantError("serialization SHA256 collision")
            roles[key].add(role)
    return by_question, by_sha, roles


def _normalized_matrix(values: Any, expected_rows: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] < 1:
        raise EvaluationInvariantError("embedder returned an invalid dense matrix shape")
    if not np.isfinite(matrix).all():
        raise EvaluationInvariantError("embedder returned a non-finite vector")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise EvaluationInvariantError("embedder returned a zero-norm vector")
    return matrix / norms


def score_prepared_split(
    prepared: PreparedSplit,
    embedder: DenseTextEmbedder,
    *,
    bm25_config: BM25Config | None = None,
) -> ScoreBundle:
    """Fit one complete BM25 universe per qid and batch dense embeddings."""

    started = time.perf_counter()
    resolved_bm25_config = bm25_config or BM25Config()
    by_question, by_sha, _roles = _candidate_inventory(prepared)
    candidates_by_qid: dict[str, dict[str, str]] = defaultdict(dict)
    for (qid, sha), candidate in by_question.items():
        candidates_by_qid[qid][sha] = candidate.serialization.text

    bm25_scores: dict[tuple[str, str], float] = {}
    for qid in sorted(candidates_by_qid):
        index = PerQuestionBM25.fit(
            candidates_by_qid[qid], config=resolved_bm25_config
        )
        for sha, score in index.score(prepared.questions[qid]).items():
            bm25_scores[(qid, sha)] = float(score)

    qids = tuple(sorted(prepared.questions))
    candidate_shas = tuple(sorted(by_sha))
    question_matrix = _normalized_matrix(
        embedder.encode([prepared.questions[qid] for qid in qids]), len(qids)
    )
    candidate_matrix = _normalized_matrix(
        embedder.encode(
            [by_sha[sha].serialization.text for sha in candidate_shas]
        ),
        len(candidate_shas),
    )
    question_vectors = dict(zip(qids, question_matrix, strict=True))
    candidate_vectors = dict(zip(candidate_shas, candidate_matrix, strict=True))
    dense_scores = {
        (qid, sha): float(question_vectors[qid] @ candidate_vectors[sha])
        for qid, sha in sorted(by_question)
    }
    return ScoreBundle(
        bm25_scores=bm25_scores,
        dense_scores=dense_scores,
        bm25_fit_question_count=len(candidates_by_qid),
        dense_question_embedding_count=len(qids),
        dense_candidate_embedding_count=len(candidate_shas),
        elapsed_seconds=time.perf_counter() - started,
    )


def cache_config(
    *,
    manifest_sha256: str,
    source_sha256: str,
    model_fingerprint_sha256: str,
    tokenizer_config_sha256: str,
    max_length: int,
    bm25_config: BM25Config,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "manifest_sha256": manifest_sha256,
        "source_sha256": source_sha256,
        "model_fingerprint_sha256": model_fingerprint_sha256,
        "tokenizer_config_sha256": tokenizer_config_sha256,
        "serializer_version_sha256": hashlib.sha256(
            SERIALIZER_VERSION.encode("ascii")
        ).hexdigest(),
        "max_length": max_length,
        "bm25_k1": bm25_config.k1,
        "bm25_b": bm25_config.b,
    }
    values["evaluation_config_sha256"] = stable_hash(values)
    return values


def build_score_cache(
    prepared: PreparedSplit,
    scores: ScoreBundle,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a text-free cache containing IDs, hashes, labels, and numbers."""

    inventory, _by_sha, roles = _candidate_inventory(prepared)
    candidates: list[dict[str, Any]] = []
    for qid, sha in sorted(inventory):
        candidate = inventory[(qid, sha)]
        candidates.append(
            {
                "qid": qid,
                "candidate_id": "candidate-" + sha[:24],
                "serialization_sha256": sha,
                "labels": sorted(roles[(qid, sha)]),
                "bm25_score": float(scores.bm25_scores[(qid, sha)]),
                "bge_m3_score": float(scores.dense_scores[(qid, sha)]),
                "untruncated_token_count": candidate.token_stats.untruncated_token_count,
                "actual_token_count": candidate.token_stats.actual_token_count,
                "truncated": candidate.token_stats.truncated,
            }
        )
    pairs = [
        {
            "qid": pair.qid,
            "pair_id": pair.pair_id,
            "positive_candidate_id": "candidate-"
            + pair.positive.serialization.sha256[:24],
            "positive_serialization_sha256": pair.positive.serialization.sha256,
            "positive_label": 1,
            "negative_candidate_id": "candidate-"
            + pair.negative.serialization.sha256[:24],
            "negative_serialization_sha256": pair.negative.serialization.sha256,
            "negative_label": 0,
        }
        for pair in sorted(prepared.pairs, key=lambda item: item.pair_id)
    ]
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "config": dict(config),
        "counts": {
            "frozen_question_count": prepared.frozen_question_count,
            "pair_eligible_question_count": len(prepared.questions),
            "excluded_no_pair_question_count": (
                prepared.frozen_question_count - len(prepared.questions)
            ),
            "candidate_count": len(candidates),
            "pair_count": len(pairs),
        },
        "candidates": candidates,
        "pairs": pairs,
    }


def load_score_cache(
    path: str | Path,
    prepared: PreparedSplit,
    expected_config: Mapping[str, Any],
) -> ScoreBundle | None:
    cache_path = Path(path)
    if not cache_path.is_file():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != CACHE_SCHEMA_VERSION
        or payload.get("config") != dict(expected_config)
        or not isinstance(payload.get("candidates"), list)
    ):
        return None
    inventory, _by_sha, _roles = _candidate_inventory(prepared)
    bm25_scores: dict[tuple[str, str], float] = {}
    dense_scores: dict[tuple[str, str], float] = {}
    for record in payload["candidates"]:
        if not isinstance(record, Mapping):
            return None
        key = (record.get("qid"), record.get("serialization_sha256"))
        if key not in inventory or key in bm25_scores:
            return None
        try:
            bm25 = float(record["bm25_score"])
            dense = float(record["bge_m3_score"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(bm25) or not math.isfinite(dense):
            return None
        bm25_scores[key] = bm25
        dense_scores[key] = dense
    if set(bm25_scores) != set(inventory):
        return None
    return ScoreBundle(
        bm25_scores=bm25_scores,
        dense_scores=dense_scores,
        bm25_fit_question_count=len(prepared.questions),
        dense_question_embedding_count=len(prepared.questions),
        dense_candidate_embedding_count=len(_by_sha),
        elapsed_seconds=0.0,
        cache_hit=True,
    )


def _describe(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": float(statistics.median(values)),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def _baseline_metrics(
    pairs: Sequence[PreparedPair], scores: Mapping[tuple[str, str], float]
) -> dict[str, Any]:
    scored_pairs: list[ScoredPair] = []
    labels: dict[tuple[str, str], bool] = {}
    for pair in pairs:
        positive_key = (pair.qid, pair.positive.serialization.sha256)
        negative_key = (pair.qid, pair.negative.serialization.sha256)
        scored_pairs.append(
            ScoredPair(
                pair.qid,
                pair.pair_id,
                scores[positive_key],
                scores[negative_key],
            )
        )
        for key, label in ((positive_key, True), (negative_key, False)):
            previous = labels.setdefault(key, label)
            if previous != label:
                raise EvaluationInvariantError(
                    "one serialization has conflicting labels within an evaluation slice"
                )
    scored_candidates = [
        ScoredCandidate(qid, sha, label, scores[(qid, sha)])
        for (qid, sha), label in sorted(labels.items())
    ]
    expected_qids = sorted({pair.qid for pair in pairs})
    ranking = evaluate_ranking(
        scored_candidates, expected_question_ids=expected_qids
    )
    return {
        "pairwise_accuracy": asdict(
            macro_pairwise_accuracy(scored_pairs, expected_question_ids=expected_qids)
        ),
        "score_margins": asdict(
            score_margin_report(scored_pairs, expected_question_ids=expected_qids)
        ),
        "auroc": asdict(ranking.auroc),
        "expected_mrr": asdict(ranking.expected_mrr),
        "expected_recall_at_1": asdict(ranking.expected_recall_at_1),
        "expected_recall_at_3": asdict(ranking.expected_recall_at_3),
        "expected_recall_at_5": asdict(ranking.expected_recall_at_5),
    }


def _group_report(pairs: Sequence[PreparedPair], scores: ScoreBundle) -> dict[str, Any]:
    signatures_positive = Counter(pair.positive.canonical_signature for pair in pairs)
    signatures_negative = Counter(pair.negative.canonical_signature for pair in pairs)
    canonical_classes = Counter(pair.canonical_class for pair in pairs)
    edge_realities = Counter(pair.negative.edge_reality for pair in pairs)
    unique_candidates: dict[str, PreparedCandidate] = {}
    member_lengths: list[int] = []
    for pair in pairs:
        for candidate in (pair.positive, pair.negative):
            unique_candidates.setdefault(candidate.serialization.sha256, candidate)
            member_lengths.append(candidate.token_stats.actual_token_count)
    unique_lengths = [
        candidate.token_stats.actual_token_count
        for candidate in unique_candidates.values()
    ]
    untruncated_lengths = [
        candidate.token_stats.untruncated_token_count
        for candidate in unique_candidates.values()
    ]
    return {
        "counts": {
            "pair_count": len(pairs),
            "base_count": len({(pair.qid, pair.base_id) for pair in pairs}),
            "question_count": len({pair.qid for pair in pairs}),
            "unique_serialization_count": len(unique_candidates),
        },
        "token_statistics": {
            "unique_untruncated": _describe(untruncated_lengths),
            "unique_actual": _describe(unique_lengths),
            "pair_member_actual": _describe(member_lengths),
            "truncated_unique_count": sum(
                candidate.token_stats.truncated
                for candidate in unique_candidates.values()
            ),
        },
        "positive_canonical_signature_counts": dict(sorted(signatures_positive.items())),
        "negative_canonical_signature_counts": dict(sorted(signatures_negative.items())),
        "canonical_class_counts": dict(sorted(canonical_classes.items())),
        "negative_edge_reality_counts": dict(sorted(edge_realities.items())),
        "baselines": {
            "bm25": _baseline_metrics(pairs, scores.bm25_scores),
            "bge_m3_dense_cosine": _baseline_metrics(pairs, scores.dense_scores),
        },
    }


def _combined_t2_pairs(pairs: Sequence[PreparedPair]) -> tuple[PreparedPair, ...]:
    """Merge both all-observed T2 sources by their scored topology identity."""

    selected = [
        pair
        for pair in pairs
        if pair.variant == "t2"
        and pair.stratum in {"synthetic_common", "t2_all_observed"}
    ]
    unique: dict[tuple[str, str, str], PreparedPair] = {}
    for pair in sorted(selected, key=lambda item: item.pair_id):
        if pair.negative.edge_reality != "all_observed":
            raise EvaluationInvariantError(
                "combined T2 group contains a non-observed negative topology"
            )
        key = (
            pair.qid,
            pair.positive.serialization.sha256,
            pair.negative.serialization.sha256,
        )
        previous = unique.get(key)
        if previous is None:
            unique[key] = pair
            continue
        if (
            previous.base_id != pair.base_id
            or previous.canonical_class != pair.canonical_class
            or previous.positive.canonical_signature
            != pair.positive.canonical_signature
            or previous.negative.canonical_signature
            != pair.negative.canonical_signature
            or previous.positive.edge_reality != pair.positive.edge_reality
            or previous.negative.edge_reality != pair.negative.edge_reality
        ):
            raise EvaluationInvariantError(
                "duplicate combined T2 topology has conflicting metadata"
            )
    return tuple(unique[key] for key in sorted(unique))


def evaluate_prepared_split(
    prepared: PreparedSplit, scores: ScoreBundle
) -> dict[str, Any]:
    by_slice: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str], list[PreparedPair]] = defaultdict(list)
    for pair in prepared.pairs:
        grouped[(pair.stratum, pair.variant)].append(pair)
    for (stratum, variant), pairs in sorted(grouped.items()):
        by_slice.setdefault(stratum, {})[variant] = _group_report(pairs, scores)

    canonical_summary: dict[str, dict[str, Any]] = {}
    pairs_by_canonical_class: dict[str, list[PreparedPair]] = defaultdict(list)
    pairs_by_edge_reality: dict[str, list[PreparedPair]] = defaultdict(list)
    for pair in prepared.pairs:
        pairs_by_canonical_class[pair.canonical_class].append(pair)
        pairs_by_edge_reality[pair.negative.edge_reality].append(pair)
        entry = canonical_summary.setdefault(
            pair.canonical_class,
            {
                "pair_count": 0,
                "positive_signature": pair.positive.canonical_signature,
                "negative_signature": pair.negative.canonical_signature,
            },
        )
        entry["pair_count"] += 1
    edge_reality_summary = Counter(pair.negative.edge_reality for pair in prepared.pairs)
    combined_t2 = _combined_t2_pairs(prepared.pairs)
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "official_split": prepared.official_split,
        "question_coverage": {
            "frozen_question_count": prepared.frozen_question_count,
            "pair_eligible_question_count": len(prepared.questions),
            "excluded_no_pair_question_count": (
                prepared.frozen_question_count - len(prepared.questions)
            ),
            "pair_eligible_rate": (
                len(prepared.questions) / prepared.frozen_question_count
                if prepared.frozen_question_count
                else None
            ),
        },
        "scoring": {
            "bm25_fit_question_count": scores.bm25_fit_question_count,
            "dense_question_embedding_count": scores.dense_question_embedding_count,
            "dense_candidate_embedding_count": scores.dense_candidate_embedding_count,
            "elapsed_seconds": scores.elapsed_seconds,
            "cache_hit": scores.cache_hit,
        },
        "overall": _group_report(prepared.pairs, scores),
        "by_stratum_variant": by_slice,
        "derived_groups": {
            "t2_all_observed_combined": _group_report(combined_t2, scores)
        },
        "by_canonical_class": {
            class_id: _group_report(class_pairs, scores)
            for class_id, class_pairs in sorted(pairs_by_canonical_class.items())
        },
        "by_edge_reality": {
            reality: _group_report(reality_pairs, scores)
            for reality, reality_pairs in sorted(pairs_by_edge_reality.items())
        },
        "canonical_class_summary": dict(sorted(canonical_summary.items())),
        "negative_edge_reality_summary": dict(sorted(edge_reality_summary.items())),
    }


class BGEM3DenseEmbedder:
    """Lazy FlagEmbedding adapter for exact-serialization dense embeddings."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda:0",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                str(self.model_path), use_fp16=True, devices=self.device
            )
        return self._model

    @property
    def tokenizer(self) -> Any:
        return self._load().tokenizer

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.asarray(encoded["dense_vecs"], dtype=np.float32)
