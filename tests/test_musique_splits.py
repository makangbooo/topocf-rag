from __future__ import annotations

import pytest

from topocf_rag.musique_splits import (
    MuSiQueSplitError,
    all_cells,
    build_split_bundle,
    collect_candidate_records,
    select_train_ids,
)


def _record(qid: str, hop: int, status: str) -> dict:
    support_indices = list(range(hop))
    paragraphs = [
        {
            "idx": index,
            "title": f"Title {qid} {index}",
            "paragraph_text": f"Paragraph {qid} {index}.",
            "is_supporting": index in support_indices,
        }
        for index in range(20)
    ]
    if status == "distractor_only_title_collision":
        paragraphs[hop + 1]["title"] = paragraphs[hop]["title"]
    elif status == "mixed_text_support_title_collision":
        paragraphs[hop]["title"] = paragraphs[0]["title"]
    return {
        "id": qid,
        "question": f"Question {qid}?",
        "answerable": True,
        "answer": "Answer",
        "answer_aliases": [],
        "question_decomposition": [
            {
                "id": index,
                "question": f"Step {index}?",
                "answer": f"Answer {index}",
                "paragraph_support_idx": index,
            }
            for index in support_indices
        ],
        "paragraphs": paragraphs,
    }


def _records(per_cell: int = 2) -> list[dict]:
    records = []
    for cell in all_cells():
        hop_text, status = cell.split("__", maxsplit=1)
        hop = int(hop_text.removesuffix("hop"))
        for index in range(per_cell):
            records.append(_record(f"{cell}-{index}", hop, status))
    return records


def _duplicate_report(record_count: int) -> dict:
    split = {
        "sha256": "a" * 64,
        "record_count": record_count,
        "policy_eligible_count": {
            "occurrence_fanout_label_unambiguous__exactly_20_paragraphs": (
                record_count
            )
        },
    }
    return {
        "task": "musique_duplicate_title_and_support_label_audit",
        "consistency_gate": {"passed": True},
        "splits": {"train": split, "dev": split},
    }


def test_candidate_collection_covers_nine_cells_without_text() -> None:
    records = _records()
    pool = collect_candidate_records(records)
    assert len(pool.ordered_ids) == 18
    assert set(pool.ids_by_cell) == set(all_cells())
    assert all(len(ids) == 2 for ids in pool.ids_by_cell.values())
    assert "Question" not in str(pool)
    assert "Paragraph" not in str(pool)


def test_train_selection_is_balanced_and_deterministic() -> None:
    pool = collect_candidate_records(_records(per_cell=3))
    first, first_counts = select_train_ids(pool, per_cell=2, seed=20260715)
    second, second_counts = select_train_ids(pool, per_cell=2, seed=20260715)
    assert first == second
    assert first_counts == second_counts
    assert len(first) == 18
    assert all(count == 2 for count in first_counts.values())


def test_exact_text_mixed_support_label_is_excluded() -> None:
    record = _record("ambiguous", 2, "unique_titles")
    record["paragraphs"][2]["title"] = record["paragraphs"][0]["title"]
    record["paragraphs"][2]["paragraph_text"] = record["paragraphs"][0][
        "paragraph_text"
    ]
    pool = collect_candidate_records([record])
    assert pool.ordered_ids == ()
    assert pool.exclusion_histogram == {
        "exact_text_mixed_support_label": 1
    }


def test_split_bundle_keeps_full_dev_and_id_only_manifests() -> None:
    records = _records(per_cell=2)
    train_pool = collect_candidate_records(records)
    dev_pool = collect_candidate_records(records)
    train, dev, report = build_split_bundle(
        train_pool,
        dev_pool,
        train_source_path="/private/train.jsonl",
        train_source_sha256="a" * 64,
        dev_source_path="/private/dev.jsonl",
        dev_source_sha256="a" * 64,
        duplicate_audit=_duplicate_report(len(records)),
        duplicate_audit_sha256="b" * 64,
        train_per_cell=1,
        seed=20260715,
    )
    assert train["selected_count"] == 9
    assert dev["selected_count"] == 18
    assert report["gate"]["passed"]
    assert set(train["ids"]).issubset(set(dev["ids"]))
    assert "Question" not in str(train)
    assert "Paragraph" not in str(dev)


def test_split_bundle_rejects_stage_b_candidate_count_drift() -> None:
    records = _records(per_cell=1)
    pool = collect_candidate_records(records)
    duplicate_report = _duplicate_report(len(records))
    duplicate_report["splits"]["dev"]["policy_eligible_count"][
        "occurrence_fanout_label_unambiguous__exactly_20_paragraphs"
    ] -= 1
    with pytest.raises(
        MuSiQueSplitError,
        match="exact-20 candidate count changed after Stage B",
    ):
        build_split_bundle(
            pool,
            pool,
            train_source_path="/private/train.jsonl",
            train_source_sha256="a" * 64,
            dev_source_path="/private/dev.jsonl",
            dev_source_sha256="a" * 64,
            duplicate_audit=duplicate_report,
            duplicate_audit_sha256="b" * 64,
            train_per_cell=1,
            seed=20260715,
        )
