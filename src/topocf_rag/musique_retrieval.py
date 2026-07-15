"""Dense retrieval caches for frozen occurrence-aware MuSiQue splits."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np

from .hotpot import sha256_file
from .musique import iter_jsonl
from .musique_duplicates import diagnose_duplicate_titles
from .musique_splits import (
    HOP_COUNTS,
    MUSIQUE_SPLIT_SCHEMA_VERSION,
    SPLIT_SEED,
    TITLE_COLLISION_STRATA,
    TRAIN_PER_CELL,
    all_cells,
    cell_name,
)


MUSIQUE_RETRIEVAL_CACHE_SCHEMA_VERSION = 1
MUSIQUE_DATASET_NAME = "musique_ans_v1.0"
EXPECTED_STAGE_C_AUDIT_SHA256 = (
    "4904e10eb7cb118cf5ed8bfe26516d24b8f4aca9fc6122364688944aada29a5c"
)
SERIALIZATION_VERSION = "title-newline-paragraph-text-v1"
SCORE_DEFINITION = (
    "cosine_similarity_of_l2_normalized_bge_m3_dense_vectors"
)
TOKENIZER_FILES = (
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class MuSiQueRetrievalError(ValueError):
    """Raised when retrieval provenance or alignment is invalid."""


class DenseEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one L2-normalized row per input text."""

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        """Return truncated token lengths aligned to the inputs."""


class BgeM3DenseEncoder:
    """Lazily load local bge-m3 and return normalized dense embeddings."""

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
            raise MuSiQueRetrievalError(
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
            raise MuSiQueRetrievalError("embedding input must not be empty")
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
            raise MuSiQueRetrievalError(
                "bge-m3 returned an invalid dense embedding shape"
            )
        if not np.isfinite(dense).all():
            raise MuSiQueRetrievalError(
                "bge-m3 returned non-finite embeddings"
            )
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise MuSiQueRetrievalError("bge-m3 returned zero-norm embeddings")
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
            raise MuSiQueRetrievalError(
                "token lengths do not align with embedding inputs"
            )
        return lengths


def fingerprint_files(root: str | Path, names: Sequence[str]) -> str:
    """Hash named local model files with names and boundaries included."""

    model_root = Path(root)
    digest = hashlib.sha256()
    found = False
    for name in names:
        path = model_root / name
        if not path.is_file():
            continue
        found = True
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    if not found:
        raise FileNotFoundError("model directory lacks required local files")
    return digest.hexdigest()


def fingerprint_model(root: str | Path) -> str:
    """Hash bge-m3 configuration and all root-level weight artifacts."""

    model_root = Path(root)
    weight_names = tuple(
        sorted(
            path.name
            for path in model_root.iterdir()
            if path.is_file()
            and (
                (path.name.startswith("pytorch_model") and path.suffix == ".bin")
                or path.suffix == ".safetensors"
                or path.name in {"sparse_linear.pt", "colbert_linear.pt"}
            )
        )
    )
    if not weight_names:
        raise FileNotFoundError(
            "model directory lacks a supported weight artifact"
        )
    return fingerprint_files(
        model_root,
        (
            "config.json",
            "configuration.json",
            "config_sentence_transformers.json",
            "modules.json",
            "sentence_bert_config.json",
            *weight_names,
        ),
    )


def load_frozen_manifest(
    path: str | Path, *, expected_split: str
) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MuSiQueRetrievalError("frozen manifest must be a JSON object")
    if payload.get("schema_version") != MUSIQUE_SPLIT_SCHEMA_VERSION:
        raise MuSiQueRetrievalError("frozen manifest schema version changed")
    if payload.get("dataset") != MUSIQUE_DATASET_NAME:
        raise MuSiQueRetrievalError("frozen manifest dataset changed")
    if payload.get("split") != expected_split:
        raise MuSiQueRetrievalError("frozen manifest split changed")
    selected_count = payload.get("selected_count")
    ids = payload.get("ids")
    if (
        not isinstance(selected_count, int)
        or isinstance(selected_count, bool)
        or selected_count < 1
    ):
        raise MuSiQueRetrievalError("frozen manifest count is invalid")
    if not isinstance(ids, list) or len(ids) != selected_count:
        raise MuSiQueRetrievalError("frozen manifest ID count is invalid")
    if not all(isinstance(qid, str) and qid for qid in ids):
        raise MuSiQueRetrievalError("frozen manifest contains an invalid ID")
    if len(ids) != len(set(ids)):
        raise MuSiQueRetrievalError("frozen manifest IDs are not unique")

    if expected_split == "train":
        selection = payload.get("selection")
        selected_by_cell = payload.get("selected_count_by_cell")
        if not isinstance(selection, Mapping) or not isinstance(
            selected_by_cell, Mapping
        ):
            raise MuSiQueRetrievalError("train allocation metadata is invalid")
        expected_counts = {cell: TRAIN_PER_CELL for cell in all_cells()}
        if dict(selected_by_cell) != expected_counts:
            raise MuSiQueRetrievalError("train cell allocation changed")
        if selection.get("seed") != SPLIT_SEED:
            raise MuSiQueRetrievalError("train sampling seed changed")
        if selection.get("per_cell") != TRAIN_PER_CELL:
            raise MuSiQueRetrievalError("train per-cell sample size changed")
        if selected_count != sum(expected_counts.values()):
            raise MuSiQueRetrievalError("train selected count changed")
    elif expected_split == "dev":
        counts = payload.get("candidate_count_by_cell")
        if not isinstance(counts, Mapping) or set(counts) != set(all_cells()):
            raise MuSiQueRetrievalError("dev cell allocation is invalid")
        if not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            for value in counts.values()
        ):
            raise MuSiQueRetrievalError("dev cell counts are invalid")
        if selected_count != sum(counts.values()):
            raise MuSiQueRetrievalError("dev is not the full candidate pool")
    else:
        raise MuSiQueRetrievalError("unsupported MuSiQue split")
    return payload


def validate_stage_c_audit(
    path: str | Path,
    *,
    train_manifest_path: str | Path,
    dev_manifest_path: str | Path,
    expected_audit_sha256: str = EXPECTED_STAGE_C_AUDIT_SHA256,
) -> dict[str, Any]:
    audit_path = Path(path)
    if sha256_file(audit_path) != expected_audit_sha256:
        raise MuSiQueRetrievalError("Stage C audit SHA256 changed")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MuSiQueRetrievalError("Stage C audit must be a JSON object")
    if payload.get("task") != (
        "musique_hop_collision_stratified_split_materialization"
    ):
        raise MuSiQueRetrievalError("Stage C audit task changed")
    gate = payload.get("gate")
    if not isinstance(gate, Mapping) or not gate.get("passed"):
        raise MuSiQueRetrievalError("Stage C split gate did not pass")
    if gate.get("status") != "authorize_bge_m3_context_scoring":
        raise MuSiQueRetrievalError("Stage C does not authorize bge-m3 scoring")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise MuSiQueRetrievalError("Stage C artifacts are missing")
    expected_hashes = {
        "train_manifest_sha256": sha256_file(train_manifest_path),
        "dev_manifest_sha256": sha256_file(dev_manifest_path),
    }
    if any(artifacts.get(key) != value for key, value in expected_hashes.items()):
        raise MuSiQueRetrievalError("Stage C manifest hash binding failed")
    return payload


def load_selected_examples(
    source: str | Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Stream one official split and retain only frozen exact-20 records."""

    source_path = Path(source)
    if manifest.get("source_sha256") != sha256_file(source_path):
        raise MuSiQueRetrievalError(
            "source SHA256 does not match the frozen manifest"
        )
    ids = manifest.get("ids")
    if not isinstance(ids, list):
        raise MuSiQueRetrievalError("frozen manifest IDs are invalid")
    wanted = set(ids)
    selected: dict[str, dict[str, Any]] = {}
    cell_counts: Counter[str] = Counter()
    for _line_number, record, error in iter_jsonl(source_path):
        if error is not None:
            raise MuSiQueRetrievalError("official source contains invalid JSON")
        if not isinstance(record, Mapping):
            continue
        qid = record.get("id")
        if qid not in wanted:
            continue
        if qid in selected:
            raise MuSiQueRetrievalError("source repeats a frozen question ID")
        diagnostic = diagnose_duplicate_titles(record)
        if (
            not diagnostic.occurrence_structurally_eligible
            or diagnostic.exact_text_mixed_support_label_group_count
            or diagnostic.paragraph_count != 20
            or diagnostic.hop_count not in HOP_COUNTS
            or diagnostic.status not in TITLE_COLLISION_STRATA
        ):
            raise MuSiQueRetrievalError(
                "a frozen question no longer satisfies Stage C eligibility"
            )
        cell_counts[
            cell_name(diagnostic.hop_count, diagnostic.status)
        ] += 1
        selected[str(qid)] = dict(record)
    missing = wanted.difference(selected)
    if missing:
        raise MuSiQueRetrievalError(
            f"source is missing {len(missing)} frozen question IDs"
        )
    split = manifest.get("split")
    expected_counts = (
        manifest.get("selected_count_by_cell")
        if split == "train"
        else manifest.get("candidate_count_by_cell")
    )
    if not isinstance(expected_counts, Mapping) or dict(
        sorted(cell_counts.items())
    ) != dict(sorted(expected_counts.items())):
        raise MuSiQueRetrievalError(
            "frozen hop-collision allocation does not reproduce"
        )
    return tuple(selected[qid] for qid in ids)


def serialize_paragraph(paragraph: Mapping[str, Any]) -> str:
    """Serialize one paragraph occurrence without changing official order."""

    return f"{paragraph['title']}\n{paragraph['paragraph_text']}"


def score_examples(
    examples: Sequence[Mapping[str, Any]],
    encoder: DenseEncoder,
) -> tuple[dict[str, dict[str, list[int] | list[float]]], dict[str, Any]]:
    """Score questions against paragraph occurrences without using labels."""

    if not examples:
        raise MuSiQueRetrievalError("selected examples must not be empty")
    questions = [str(example["question"]) for example in examples]
    documents: list[str] = []
    paragraph_indices: list[int] = []
    offsets = [0]
    for example in examples:
        paragraphs = example.get("paragraphs")
        if not isinstance(paragraphs, list) or len(paragraphs) != 20:
            raise MuSiQueRetrievalError(
                "every selected example must contain 20 paragraphs"
            )
        for paragraph in paragraphs:
            if not isinstance(paragraph, Mapping):
                raise MuSiQueRetrievalError("paragraph must be an object")
            documents.append(serialize_paragraph(paragraph))
            paragraph_indices.append(int(paragraph["idx"]))
        offsets.append(len(documents))

    started = time.perf_counter()
    query_vectors = encoder.encode(questions)
    document_vectors = encoder.encode(documents)
    token_lengths = encoder.token_lengths(documents)
    elapsed = time.perf_counter() - started
    if query_vectors.ndim != 2 or document_vectors.ndim != 2:
        raise MuSiQueRetrievalError("dense embeddings must be matrices")
    if query_vectors.shape[0] != len(questions):
        raise MuSiQueRetrievalError("query embeddings do not align")
    if document_vectors.shape[0] != len(documents):
        raise MuSiQueRetrievalError("document embeddings do not align")
    if query_vectors.shape[1] != document_vectors.shape[1]:
        raise MuSiQueRetrievalError("query and document dimensions differ")
    if len(token_lengths) != len(documents):
        raise MuSiQueRetrievalError("token lengths do not align")

    records: dict[str, dict[str, list[int] | list[float]]] = {}
    for example_index, example in enumerate(examples):
        start, end = offsets[example_index : example_index + 2]
        scores = document_vectors[start:end] @ query_vectors[example_index]
        if not np.isfinite(scores).all():
            raise MuSiQueRetrievalError("retrieval scores must be finite")
        records[str(example["id"])] = {
            "paragraph_indices": paragraph_indices[start:end],
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
    split: str,
    source: str | Path,
    ids_path: str | Path,
    split_audit_path: str | Path,
    model_path: str | Path,
    model_fingerprint_sha256: str,
    tokenizer_fingerprint_sha256: str,
    max_length: int,
) -> dict[str, Any]:
    source_path = Path(source)
    manifest_path = Path(ids_path)
    audit_path = Path(split_audit_path)
    model = Path(model_path)
    return {
        "schema_version": MUSIQUE_RETRIEVAL_CACHE_SCHEMA_VERSION,
        "dataset": MUSIQUE_DATASET_NAME,
        "split": split,
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "ids_path": str(manifest_path.resolve()),
        "ids_sha256": sha256_file(manifest_path),
        "split_audit_path": str(audit_path.resolve()),
        "split_audit_sha256": sha256_file(audit_path),
        "model_path": str(model.resolve()),
        "model_fingerprint_sha256": model_fingerprint_sha256,
        "tokenizer_fingerprint_sha256": tokenizer_fingerprint_sha256,
        "max_length": max_length,
        "inference_precision": "fp16",
        "serialization_version": SERIALIZATION_VERSION,
        "score": SCORE_DEFINITION,
    }


def cache_matches(
    payload: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    return all(payload.get(key) == value for key, value in expected.items()) and (
        isinstance(payload.get("records"), Mapping)
    )


def validate_cache_records(
    payload: Mapping[str, Any], examples: Sequence[Mapping[str, Any]]
) -> None:
    records = payload.get("records")
    if not isinstance(records, Mapping):
        raise MuSiQueRetrievalError("retrieval cache records are invalid")
    expected_ids = {str(example["id"]) for example in examples}
    if set(records) != expected_ids:
        raise MuSiQueRetrievalError(
            "retrieval cache IDs do not match frozen examples"
        )
    for example in examples:
        qid = str(example["id"])
        record = records[qid]
        if not isinstance(record, Mapping):
            raise MuSiQueRetrievalError("retrieval cache row is invalid")
        paragraphs = example["paragraphs"]
        expected_indices = [int(paragraph["idx"]) for paragraph in paragraphs]
        indices = record.get("paragraph_indices")
        scores = record.get("scores")
        lengths = record.get("document_token_lengths")
        if indices != expected_indices:
            raise MuSiQueRetrievalError(
                "retrieval cache paragraph indices are misaligned"
            )
        if not isinstance(scores, list) or len(scores) != len(paragraphs):
            raise MuSiQueRetrievalError("retrieval cache score shape is invalid")
        if not all(
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and np.isfinite(score)
            for score in scores
        ):
            raise MuSiQueRetrievalError("retrieval cache scores are invalid")
        if not isinstance(lengths, list) or len(lengths) != len(paragraphs):
            raise MuSiQueRetrievalError(
                "retrieval cache token-length shape is invalid"
            )
        if not all(
            isinstance(length, int)
            and not isinstance(length, bool)
            and length >= 1
            for length in lengths
        ):
            raise MuSiQueRetrievalError(
                "retrieval cache token lengths are invalid"
            )
