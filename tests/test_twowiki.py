from __future__ import annotations

import json
from pathlib import Path

import pytest

import topocf_rag.twowiki as twowiki
from topocf_rag.twowiki import (
    ELIGIBILITY_PREDICATE,
    QUESTION_TYPES,
    TwoWikiInvariantError,
    assess_twowiki_eligibility,
    prepare_twowiki_id_split,
    scan_twowiki_source,
    stratified_sample_ids,
    validate_id_manifest,
)


@pytest.fixture(autouse=True)
def _synthetic_json_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests independent of the server-only ijson installation."""

    def iterator(path: str | Path):
        yield from json.loads(Path(path).read_text(encoding="utf-8"))

    monkeypatch.setattr(twowiki, "iter_twowiki_records", iterator)


def _record(qid: str, question_type: str) -> dict[str, object]:
    context = [
        [f"Title {index}", [f"Synthetic sentence {index}."]]
        for index in range(10)
    ]
    gold_count = 4 if question_type == "bridge_comparison" else 2
    return {
        "_id": qid,
        "question": "Synthetic private question?",
        "answer": "Synthetic private answer",
        "type": question_type,
        "context": context,
        "supporting_facts": [
            [f"Title {index}", 0] for index in range(gold_count)
        ],
    }


def _write(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def test_eligibility_uses_exact_gold_mapping_and_rejects_graph_ambiguity() -> None:
    valid = _record("valid", "compositional")
    assert assess_twowiki_eligibility(valid).eligible

    normalized_only = _record("normalized-only", "compositional")
    normalized_only["supporting_facts"][0][0] = "Title_0"  # type: ignore[index]
    assessment = assess_twowiki_eligibility(normalized_only)
    assert "unresolved_support_fact_exact_title" in assessment.reasons

    duplicate = _record("duplicate", "compositional")
    duplicate["context"][9][0] = "TITLE_0"  # type: ignore[index]
    assessment = assess_twowiki_eligibility(duplicate)
    assert "normalized_duplicate_context_title" in assessment.reasons

    out_of_range = _record("out-of-range", "compositional")
    out_of_range["supporting_facts"][0][1] = 1  # type: ignore[index]
    assessment = assess_twowiki_eligibility(out_of_range)
    assert assessment.unresolved_support_fact_count == 1
    assert not assessment.eligible


def test_scan_reports_union_and_reason_counts_without_text(tmp_path: Path) -> None:
    source = tmp_path / "train.json"
    records = [_record(f"valid-{kind}", kind) for kind in QUESTION_TYPES]
    invalid = _record("invalid", "comparison")
    invalid["context"][9][0] = "TITLE_0"  # type: ignore[index]
    invalid["supporting_facts"][0][1] = 2  # type: ignore[index]
    records.append(invalid)
    _write(source, records)

    stats, eligible = scan_twowiki_source(source)
    assert stats["overall"]["record_count"] == 5
    assert stats["overall"]["eligible_count"] == 4
    assert stats["overall"]["excluded_count"] == 1
    reasons = stats["overall"]["exclusion_reason_question_counts"]
    assert reasons["normalized_duplicate_context_title"] == 1
    assert reasons["unresolved_support_fact_exact_title"] == 1
    assert {qid for ids in eligible.values() for qid in ids} == {
        f"valid-{kind}" for kind in QUESTION_TYPES
    }
    encoded = json.dumps(stats, sort_keys=True)
    assert "Synthetic private" not in encoded
    assert "Title 0" not in encoded


def test_balanced_sampling_is_deterministic_and_order_independent() -> None:
    candidates = {
        kind: [f"{kind}-{index}" for index in range(20)]
        for kind in QUESTION_TYPES
    }
    first, allocation = stratified_sample_ids(
        candidates, sample_size=12, seed=20260714, official_split="train"
    )
    second, second_allocation = stratified_sample_ids(
        {
            kind: list(reversed(candidates[kind]))
            for kind in reversed(QUESTION_TYPES)
        },
        sample_size=12,
        seed=20260714,
        official_split="train",
    )
    assert first == second
    assert allocation == second_allocation == {
        "bridge_comparison": 3,
        "comparison": 3,
        "compositional": 3,
        "inference": 3,
    }


def test_prepare_split_records_provenance_and_valid_manifest(tmp_path: Path) -> None:
    source = tmp_path / "train.json"
    output = tmp_path / "ids.json"
    records = [
        _record(f"{kind}-{index}", kind)
        for kind in QUESTION_TYPES
        for index in range(4)
    ]
    _write(source, list(reversed(records)))

    manifest, stats, output_sha256 = prepare_twowiki_id_split(
        source,
        output,
        official_split="train",
        sample_size=8,
        seed=20260714,
    )
    validate_id_manifest(manifest, expected_split="train", expected_size=8)
    assert stats["overall"]["eligible_count"] == 16
    assert manifest["selection"]["selected_by_question_type"] == {
        kind: 2 for kind in QUESTION_TYPES
    }
    assert manifest["selection"]["eligibility_predicate"] == ELIGIBILITY_PREDICATE
    assert output_sha256
    assert "output_sha256" not in output.read_text(encoding="utf-8")


def test_official_filename_and_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    wrong = tmp_path / "renamed.json"
    _write(wrong, [_record("one", "comparison")])
    with pytest.raises(TwoWikiInvariantError, match="train.json"):
        prepare_twowiki_id_split(
            wrong, tmp_path / "ids.json", official_split="train", sample_size=4
        )

    source = tmp_path / "dev.json"
    _write(source, [_record("same", kind) for kind in QUESTION_TYPES])
    with pytest.raises(TwoWikiInvariantError, match="duplicate"):
        scan_twowiki_source(source)
