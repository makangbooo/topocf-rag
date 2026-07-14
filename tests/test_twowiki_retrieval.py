from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import topocf_rag.twowiki_retrieval as retrieval
from topocf_rag.hotpot import sha256_file
from topocf_rag.twowiki import (
    ELIGIBILITY_PREDICATE,
    QUESTION_TYPES,
    SAMPLING_ALGORITHM,
    TwoWikiInvariantError,
)
from topocf_rag.twowiki_retrieval import (
    cache_matches,
    expected_cache_metadata,
    load_frozen_manifest,
    load_selected_examples,
    score_examples,
    validate_cache_records,
)


def _record(qid: str, question_type: str) -> dict[str, object]:
    gold_count = 4 if question_type == "bridge_comparison" else 2
    return {
        "_id": qid,
        "question": f"Synthetic question for {qid}?",
        "answer": "Synthetic answer",
        "type": question_type,
        "context": [
            [f"Title {index}", [f"Sentence {index}."]]
            for index in range(10)
        ],
        "supporting_facts": [
            [f"Title {index}", 0] for index in range(gold_count)
        ],
    }


def _write_source(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def _manifest(source: Path, ids: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": "2wikimultihopqa",
        "official_split": "train",
        "source": {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
            "record_count": len(ids),
        },
        "selection": {
            "seed": 20260714,
            "sample_size": len(ids),
            "sampling_algorithm": SAMPLING_ALGORITHM,
            "eligibility_predicate": ELIGIBILITY_PREDICATE,
            "selected_by_question_type": {kind: 1 for kind in QUESTION_TYPES},
        },
        "ids": ids,
    }


@pytest.fixture
def selected_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "train.json"
    records = [_record(f"id-{kind}", kind) for kind in QUESTION_TYPES]
    _write_source(source, list(reversed(records)))
    ids = [f"id-{kind}" for kind in QUESTION_TYPES]
    manifest = _manifest(source, ids)
    manifest_path = tmp_path / "ids.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def iterator(path: str | Path):
        yield from json.loads(Path(path).read_text(encoding="utf-8"))

    monkeypatch.setattr(retrieval, "iter_twowiki_records", iterator)
    return source, manifest_path, manifest, records


def test_load_manifest_and_selected_examples_preserve_frozen_order(
    selected_bundle,
) -> None:
    source, manifest_path, manifest, _records = selected_bundle
    loaded = load_frozen_manifest(manifest_path)
    selected = load_selected_examples(source, loaded)
    assert [record["_id"] for record in selected] == manifest["ids"]
    assert [record["type"] for record in selected] == list(QUESTION_TYPES)


def test_selected_example_loader_rejects_changed_source(
    selected_bundle,
) -> None:
    source, _manifest_path, manifest, _records = selected_bundle
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(TwoWikiInvariantError, match="SHA256"):
        load_selected_examples(source, manifest)


class _FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        if self.calls == 1:
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)
        rows = []
        for index, _text in enumerate(texts):
            rows.append([1.0, 0.0] if index % 10 == 0 else [0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)

    def token_lengths(self, texts):
        return [min(512, len(text)) for text in texts]


def test_score_cache_is_context_aligned_and_contains_no_text() -> None:
    examples = [_record("one", "comparison"), _record("two", "inference")]
    records, runtime = score_examples(examples, _FakeEncoder())
    assert set(records) == {"one", "two"}
    assert records["one"]["scores"][0] == pytest.approx(1.0)
    assert records["one"]["scores"][1] == pytest.approx(0.0)
    assert records["two"]["scores"][0] == pytest.approx(1.0)
    assert len(records["one"]["document_token_lengths"]) == 10
    assert runtime["question_count"] == 2
    assert runtime["document_count"] == 20
    encoded = json.dumps(records, sort_keys=True)
    for forbidden in ("Synthetic question", "Synthetic answer", "Title", "Sentence"):
        assert forbidden not in encoded


def test_metadata_matching_and_cache_validation(selected_bundle) -> None:
    source, manifest_path, _manifest, records = selected_bundle
    metadata = expected_cache_metadata(
        source=source,
        ids_path=manifest_path,
        model_path=Path("/models/bge-m3"),
        max_length=512,
    )
    scored, _runtime = score_examples(records, _FakeEncoder())
    payload = {**metadata, "records": scored}
    assert cache_matches(payload, metadata)
    validate_cache_records(payload, records)

    changed = {**metadata, "max_length": 256}
    assert not cache_matches(payload, changed)
    payload["records"][records[0]["_id"]]["scores"].pop()
    with pytest.raises(TwoWikiInvariantError, match="score shape"):
        validate_cache_records(payload, records)
