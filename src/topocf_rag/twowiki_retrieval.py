"""Dense retrieval cache construction for frozen clean 2Wiki splits."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np

from .hotpot import sha256_file
from .twowiki import (
    DATASET_NAME,
    QUESTION_TYPES,
    TwoWikiInvariantError,
    assess_twowiki_eligibility,
    iter_twowiki_records,
    validate_id_manifest,
)


CACHE_SCHEMA_VERSION = 1
SERIALIZATION_VERSION = "title-newline-joined-context-sentences-v1"


class DenseEncoder(Protocol):
    """Minimal interface used by the text-free cache builder."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one L2-normalized row per input text."""

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        """Return truncated token lengths aligned to the inputs."""


class BgeM3DenseEncoder:
    """Lazily load local bge-m3 and expose normalized dense embeddings."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str,
        batch_size: int,
        max_length: int,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any = None
        if batch_size < 1 or max_length < 1:
            raise TwoWikiInvariantError(
                "batch size and maximum length must be positive"
            )

    def _load(self) -> Any:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                str(self.model_path), use_fp16=True, devices=self.device
            )
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise TwoWikiInvariantError("embedding input must not be empty")
        encoded = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = np.asarray(encoded["dense_vecs"], dtype=np.float32)
        if dense.ndim != 2 or dense.shape[0] != len(texts):
            raise TwoWikiInvariantError("bge-m3 returned an invalid dense shape")
        if not np.isfinite(dense).all():
            raise TwoWikiInvariantError("bge-m3 returned non-finite embeddings")
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise TwoWikiInvariantError("bge-m3 returned zero-norm embeddings")
        return dense / norms

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        lengths: list[int] = []
        window = self.batch_size * 4
        for start in range(0, len(texts), window):
            batch = list(texts[start : start + window])
            tokenized = self._load().tokenizer(
                batch,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
                padding=False,
            )
            lengths.extend(len(mask) for mask in tokenized["attention_mask"])
        if len(lengths) != len(texts):
            raise TwoWikiInvariantError("token lengths do not align with inputs")
        return lengths


def load_frozen_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TwoWikiInvariantError("frozen manifest must be a JSON object")
    selection = payload.get("selection")
    sample_size = selection.get("sample_size") if isinstance(selection, Mapping) else None
    split = payload.get("official_split")
    if not isinstance(sample_size, int) or isinstance(sample_size, bool):
        raise TwoWikiInvariantError("frozen manifest sample size is invalid")
    if not isinstance(split, str):
        raise TwoWikiInvariantError("frozen manifest split is invalid")
    validate_id_manifest(payload, expected_split=split, expected_size=sample_size)
    return payload


def load_selected_examples(
    source: str | Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Stream and return only frozen, still-eligible records in manifest order."""

    source_path = Path(source)
    source_metadata = manifest.get("source")
    if not isinstance(source_metadata, Mapping):
        raise TwoWikiInvariantError("manifest source metadata is invalid")
    expected_source_sha256 = source_metadata.get("sha256")
    if expected_source_sha256 != sha256_file(source_path):
        raise TwoWikiInvariantError("source SHA256 does not match frozen manifest")
    ids = manifest.get("ids")
    if not isinstance(ids, list):
        raise TwoWikiInvariantError("manifest IDs are invalid")
    wanted = set(ids)
    selected: dict[str, dict[str, Any]] = {}
    type_counts: Counter[str] = Counter()
    for record in iter_twowiki_records(source_path):
        qid = record.get("_id")
        if qid not in wanted:
            continue
        if qid in selected:
            raise TwoWikiInvariantError("source repeats a frozen question ID")
        assessment = assess_twowiki_eligibility(record)
        if not assessment.eligible:
            raise TwoWikiInvariantError("frozen question is no longer eligible")
        selected[qid] = record
        type_counts[assessment.question_type] += 1
    missing = wanted.difference(selected)
    if missing:
        raise TwoWikiInvariantError(
            f"source is missing {len(missing)} frozen question IDs"
        )
    selection = manifest["selection"]
    expected_counts = selection["selected_by_question_type"]
    if dict(sorted(type_counts.items())) != dict(sorted(expected_counts.items())):
        raise TwoWikiInvariantError("frozen question-type allocation does not reproduce")
    return tuple(selected[qid] for qid in ids)


def serialize_context_document(document: Sequence[Any]) -> str:
    """Use exactly the Hotpot kill-test title/body serialization."""

    title, sentences = document
    return f"{title}\n{' '.join(sentences)}"


def score_examples(
    examples: Sequence[Mapping[str, Any]],
    encoder: DenseEncoder,
) -> tuple[dict[str, dict[str, list[int] | list[float]]], dict[str, Any]]:
    """Score every question against its ten documents without using gold labels."""

    if not examples:
        raise TwoWikiInvariantError("selected examples must not be empty")
    questions = [str(example["question"]) for example in examples]
    documents: list[str] = []
    offsets = [0]
    for example in examples:
        documents.extend(
            serialize_context_document(document) for document in example["context"]
        )
        offsets.append(len(documents))

    started = time.perf_counter()
    query_vectors = encoder.encode(questions)
    document_vectors = encoder.encode(documents)
    token_lengths = encoder.token_lengths(documents)
    elapsed = time.perf_counter() - started
    if query_vectors.ndim != 2 or document_vectors.ndim != 2:
        raise TwoWikiInvariantError("dense embeddings must be matrices")
    if query_vectors.shape[0] != len(questions):
        raise TwoWikiInvariantError("query embeddings do not align with questions")
    if document_vectors.shape[0] != len(documents):
        raise TwoWikiInvariantError("document embeddings do not align with documents")
    if query_vectors.shape[1] != document_vectors.shape[1]:
        raise TwoWikiInvariantError("query and document dimensions differ")
    if len(token_lengths) != len(documents):
        raise TwoWikiInvariantError("token lengths do not align with documents")

    records: dict[str, dict[str, list[int] | list[float]]] = {}
    for example_index, example in enumerate(examples):
        start, end = offsets[example_index : example_index + 2]
        scores = document_vectors[start:end] @ query_vectors[example_index]
        if not np.isfinite(scores).all():
            raise TwoWikiInvariantError("retrieval scores must be finite")
        records[str(example["_id"])] = {
            "scores": [float(score) for score in scores],
            "document_token_lengths": token_lengths[start:end],
        }
    return records, {
        "elapsed_seconds": elapsed,
        "question_count": len(questions),
        "document_count": len(documents),
        "embedding_dimension": int(query_vectors.shape[1]),
    }


def expected_cache_metadata(
    *,
    source: str | Path,
    ids_path: str | Path,
    model_path: str | Path,
    max_length: int,
) -> dict[str, Any]:
    source_path = Path(source)
    manifest_path = Path(ids_path)
    model = Path(model_path)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "ids_path": str(manifest_path.resolve()),
        "ids_sha256": sha256_file(manifest_path),
        "model_path": str(model.resolve()),
        "max_length": max_length,
        "serialization_version": SERIALIZATION_VERSION,
        "score": "cosine_similarity_of_l2_normalized_bge_m3_dense_vectors",
    }


def cache_matches(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(payload.get(key) == value for key, value in expected.items()) and isinstance(
        payload.get("records"), Mapping
    )


def validate_cache_records(
    payload: Mapping[str, Any], examples: Sequence[Mapping[str, Any]]
) -> None:
    records = payload.get("records")
    if not isinstance(records, Mapping):
        raise TwoWikiInvariantError("retrieval cache records are invalid")
    expected_ids = {str(example["_id"]) for example in examples}
    if set(records) != expected_ids:
        raise TwoWikiInvariantError("retrieval cache IDs do not match frozen examples")
    for example in examples:
        record = records[str(example["_id"])]
        scores = record.get("scores") if isinstance(record, Mapping) else None
        lengths = (
            record.get("document_token_lengths")
            if isinstance(record, Mapping)
            else None
        )
        document_count = len(example["context"])
        if not isinstance(scores, list) or len(scores) != document_count:
            raise TwoWikiInvariantError("retrieval cache score shape is invalid")
        if not all(
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and np.isfinite(score)
            for score in scores
        ):
            raise TwoWikiInvariantError("retrieval cache scores are invalid")
        if not isinstance(lengths, list) or len(lengths) != document_count:
            raise TwoWikiInvariantError("retrieval cache token lengths are invalid")
        if not all(
            isinstance(length, int)
            and not isinstance(length, bool)
            and length >= 1
            for length in lengths
        ):
            raise TwoWikiInvariantError("retrieval cache token lengths are invalid")
