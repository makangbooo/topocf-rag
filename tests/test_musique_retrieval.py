from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from topocf_rag.hotpot import sha256_file
from topocf_rag.musique_retrieval import (
    MuSiQueRetrievalError,
    cache_matches,
    expected_cache_metadata,
    fingerprint_files,
    fingerprint_model,
    load_frozen_manifest,
    load_selected_examples,
    score_examples,
    validate_cache_records,
    validate_stage_c_audit,
)
from topocf_rag.musique_splits import all_cells


def _record(qid: str, hop: int, status: str) -> dict:
    paragraphs = [
        {
            "idx": index,
            "title": f"Title {qid} {index}",
            "paragraph_text": f"Paragraph text {qid} {index}.",
            "is_supporting": index < hop,
        }
        for index in range(20)
    ]
    if status == "distractor_only_title_collision":
        paragraphs[hop + 1]["title"] = paragraphs[hop]["title"]
    elif status == "mixed_text_support_title_collision":
        paragraphs[hop]["title"] = paragraphs[0]["title"]
    return {
        "id": qid,
        "question": f"Synthetic question {qid}?",
        "answerable": True,
        "answer": "Synthetic answer",
        "answer_aliases": [],
        "question_decomposition": [
            {
                "id": index,
                "question": f"Step {index}?",
                "answer": f"Step answer {index}",
                "paragraph_support_idx": index,
            }
            for index in range(hop)
        ],
        "paragraphs": paragraphs,
    }


def _records() -> list[dict]:
    records = []
    for cell in all_cells():
        hop_text, status = cell.split("__", maxsplit=1)
        hop = int(hop_text.removesuffix("hop"))
        records.append(_record(cell, hop, status))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _small_manifest(source: Path, records: list[dict], split: str) -> dict:
    counts = {cell: 1 for cell in all_cells()}
    payload = {
        "schema_version": 1,
        "dataset": "musique_ans_v1.0",
        "split": split,
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "selected_count": len(records),
        "ids": [record["id"] for record in records],
    }
    if split == "train":
        payload["selected_count_by_cell"] = counts
    else:
        payload["candidate_count_by_cell"] = counts
    return payload


def test_load_frozen_train_manifest_enforces_frozen_allocation(
    tmp_path: Path,
) -> None:
    ids = [f"{cell}-{index}" for cell in all_cells() for index in range(200)]
    manifest = {
        "schema_version": 1,
        "dataset": "musique_ans_v1.0",
        "split": "train",
        "source_path": "/private/train.jsonl",
        "source_sha256": "a" * 64,
        "selection": {"seed": 20260715, "per_cell": 200},
        "selected_count": 1800,
        "selected_count_by_cell": {cell: 200 for cell in all_cells()},
        "ids": ids,
    }
    path = tmp_path / "train_ids.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_frozen_manifest(path, expected_split="train") == manifest
    manifest["selection"]["seed"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MuSiQueRetrievalError, match="sampling seed"):
        load_frozen_manifest(path, expected_split="train")


def test_selected_examples_preserve_manifest_order_and_occurrences(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dev.jsonl"
    records = _records()
    _write_jsonl(source, list(reversed(records)))
    manifest = _small_manifest(source, records, "dev")
    selected = load_selected_examples(source, manifest)
    assert [record["id"] for record in selected] == manifest["ids"]
    assert all(len(record["paragraphs"]) == 20 for record in selected)


def test_stage_c_audit_binds_both_manifests(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    dev = tmp_path / "dev.json"
    train.write_text("{}", encoding="utf-8")
    dev.write_text("{}", encoding="utf-8")
    audit = {
        "task": "musique_hop_collision_stratified_split_materialization",
        "gate": {
            "passed": True,
            "status": "authorize_bge_m3_context_scoring",
        },
        "artifacts": {
            "train_manifest_sha256": sha256_file(train),
            "dev_manifest_sha256": sha256_file(dev),
        },
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    assert validate_stage_c_audit(
        audit_path,
        train_manifest_path=train,
        dev_manifest_path=dev,
        expected_audit_sha256=sha256_file(audit_path),
    ) == audit
    dev.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(MuSiQueRetrievalError, match="hash binding"):
        validate_stage_c_audit(
            audit_path,
            train_manifest_path=train,
            dev_manifest_path=dev,
            expected_audit_sha256=sha256_file(audit_path),
        )


class _FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        if self.calls == 1:
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)
        return np.asarray(
            [
                [1.0, 0.0] if index % 20 == 0 else [0.0, 1.0]
                for index, _text in enumerate(texts)
            ],
            dtype=np.float32,
        )

    def token_lengths(self, texts):
        return [min(512, len(text)) for text in texts]


def test_score_cache_is_occurrence_aligned_and_content_free() -> None:
    examples = [
        _record("one", 2, "unique_titles"),
        _record("two", 3, "mixed_text_support_title_collision"),
    ]
    records, runtime = score_examples(examples, _FakeEncoder())
    assert set(records) == {"one", "two"}
    assert records["one"]["paragraph_indices"] == list(range(20))
    assert records["one"]["scores"][0] == pytest.approx(1.0)
    assert records["one"]["scores"][1] == pytest.approx(0.0)
    assert records["two"]["scores"][0] == pytest.approx(1.0)
    assert runtime["question_count"] == 2
    assert runtime["document_count"] == 40
    encoded = json.dumps(records, sort_keys=True)
    for forbidden in (
        "Synthetic question",
        "Synthetic answer",
        "Title",
        "Paragraph text",
    ):
        assert forbidden not in encoded


def test_cache_metadata_and_alignment_validation(tmp_path: Path) -> None:
    source = tmp_path / "dev.jsonl"
    examples = [_record("one", 2, "unique_titles")]
    _write_jsonl(source, examples)
    manifest = tmp_path / "ids.json"
    audit = tmp_path / "audit.json"
    manifest.write_text("{}", encoding="utf-8")
    audit.write_text("{}", encoding="utf-8")
    metadata = expected_cache_metadata(
        split="dev",
        source=source,
        ids_path=manifest,
        split_audit_path=audit,
        model_path=Path("/models/bge-m3"),
        model_fingerprint_sha256="a" * 64,
        tokenizer_fingerprint_sha256="b" * 64,
        max_length=512,
    )
    scored, _runtime = score_examples(examples, _FakeEncoder())
    payload = {**metadata, "records": scored}
    assert cache_matches(payload, metadata)
    validate_cache_records(payload, examples)

    payload["records"]["one"]["paragraph_indices"][0] = 19
    with pytest.raises(MuSiQueRetrievalError, match="indices"):
        validate_cache_records(payload, examples)


def test_model_fingerprints_change_with_local_artifacts(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights-v1")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    first_model = fingerprint_model(model)
    first_tokenizer = fingerprint_files(model, ("tokenizer_config.json",))

    (model / "model.safetensors").write_bytes(b"weights-v2")
    (model / "tokenizer_config.json").write_text(
        '{"changed": true}', encoding="utf-8"
    )
    assert fingerprint_model(model) != first_model
    assert fingerprint_files(model, ("tokenizer_config.json",)) != (
        first_tokenizer
    )
