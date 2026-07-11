from __future__ import annotations

import json
from pathlib import Path

import pytest

from topocf_rag.hotpot import (
    prepare_hotpot_id_split,
    proportional_allocation,
    selected_ids_are_bridge,
    sha256_file,
    stratified_sample_ids,
    validate_id_manifest,
)


def _record(record_id: str, *, level: str, question_type: str = "bridge") -> dict:
    return {
        "_id": record_id,
        "question": f"Synthetic question {record_id}?",
        "answer": "Synthetic answer",
        "type": question_type,
        "level": level,
        "context": [["Title", ["Synthetic sentence."]]],
        "supporting_facts": [["Title", 0]],
    }


def _write_source(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def test_proportional_allocation_preserves_level_mix() -> None:
    assert proportional_allocation({"easy": 30, "medium": 50, "hard": 20}, 10) == {
        "easy": 3,
        "hard": 2,
        "medium": 5,
    }
    allocation = proportional_allocation({"easy": 2, "hard": 1, "medium": 1}, 3)
    assert sum(allocation.values()) == 3
    assert all(allocation[level] <= count for level, count in {
        "easy": 2,
        "hard": 1,
        "medium": 1,
    }.items())


def test_stratified_sampling_is_deterministic_and_order_independent() -> None:
    candidates = {
        "easy": [f"easy-{index}" for index in range(20)],
        "hard": [f"hard-{index}" for index in range(10)],
    }
    selected_a, allocation_a = stratified_sample_ids(
        candidates, sample_size=9, seed=20260711, official_split="train"
    )
    selected_b, allocation_b = stratified_sample_ids(
        {level: list(reversed(ids)) for level, ids in reversed(candidates.items())},
        sample_size=9,
        seed=20260711,
        official_split="train",
    )
    assert selected_a == selected_b
    assert allocation_a == allocation_b == {"easy": 6, "hard": 3}


def test_prepare_manifest_filters_bridge_and_records_provenance(tmp_path: Path) -> None:
    source = tmp_path / "hotpot_train_v1.1.json"
    output = tmp_path / "train_ids.json"
    records = [
        *[_record(f"easy-{index}", level="easy") for index in range(12)],
        *[_record(f"hard-{index}", level="hard") for index in range(8)],
        *[
            _record(f"comparison-{index}", level="easy", question_type="comparison")
            for index in range(4)
        ],
    ]
    _write_source(source, records)

    manifest, output_sha256 = prepare_hotpot_id_split(
        source,
        output,
        official_split="train",
        sample_size=10,
        seed=20260711,
    )

    validate_id_manifest(manifest, expected_split="train", expected_size=10)
    assert manifest["selection"]["selected_by_level"] == {"easy": 6, "hard": 4}
    assert manifest["source"]["record_count"] == 24
    assert manifest["source"]["bridge_count"] == 20
    assert manifest["source"]["sha256"] == sha256_file(source)
    assert output_sha256 == sha256_file(output)
    assert "output_sha256" not in output.read_text(encoding="utf-8")
    assert selected_ids_are_bridge(source, manifest["ids"])


def test_manifest_rejects_duplicate_ids() -> None:
    manifest = {
        "official_split": "train",
        "selection": {
            "predicate": "type == 'bridge'",
            "selected_by_level": {"easy": 2},
        },
        "ids": ["same", "same"],
    }
    with pytest.raises(ValueError, match="duplicates"):
        validate_id_manifest(manifest, expected_split="train", expected_size=2)


def test_official_split_cannot_use_the_other_source_filename(tmp_path: Path) -> None:
    source = tmp_path / "hotpot_dev_distractor_v1.json"
    _write_source(source, [_record("dev-1", level="easy")])
    with pytest.raises(ValueError, match="train must come from"):
        prepare_hotpot_id_split(
            source,
            tmp_path / "ids.json",
            official_split="train",
            sample_size=1,
            seed=20260711,
        )
