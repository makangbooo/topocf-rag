"""Frozen MuSiQue train/dev ID manifests for the shortcut audit."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from .musique import iter_jsonl
from .musique_duplicates import diagnose_duplicate_titles


MUSIQUE_SPLIT_SCHEMA_VERSION = 1
SPLIT_SEED = 20260715
HOP_COUNTS = (2, 3, 4)
TITLE_COLLISION_STRATA = (
    "unique_titles",
    "distractor_only_title_collision",
    "mixed_text_support_title_collision",
)
TRAIN_PER_CELL = 200
EXPECTED_TRAIN_COUNT = (
    len(HOP_COUNTS) * len(TITLE_COLLISION_STRATA) * TRAIN_PER_CELL
)


class MuSiQueSplitError(ValueError):
    """Raised when a frozen MuSiQue split cannot be reproduced safely."""


@dataclass(frozen=True, slots=True)
class CandidatePool:
    record_count: int
    ordered_ids: tuple[str, ...]
    ids_by_cell: Mapping[str, tuple[str, ...]]
    exclusion_histogram: Mapping[str, int]


def cell_name(hop_count: int, status: str) -> str:
    if hop_count not in HOP_COUNTS:
        raise MuSiQueSplitError("unsupported hop count")
    if status not in TITLE_COLLISION_STRATA:
        raise MuSiQueSplitError("unsupported collision stratum")
    return f"{hop_count}hop__{status}"


def all_cells() -> tuple[str, ...]:
    return tuple(
        cell_name(hop_count, status)
        for hop_count in HOP_COUNTS
        for status in TITLE_COLLISION_STRATA
    )


def collect_candidate_records(records: Iterable[Any]) -> CandidatePool:
    """Collect IDs only for occurrence-aware, exact-20 paragraph records."""

    ordered_ids: list[str] = []
    by_cell: dict[str, list[str]] = {name: [] for name in all_cells()}
    exclusions: Counter[str] = Counter()
    seen_ids: set[str] = set()
    record_count = 0
    for record in records:
        record_count += 1
        if not isinstance(record, Mapping):
            exclusions["non_object"] += 1
            continue
        qid = record.get("id")
        if not isinstance(qid, str) or not qid:
            exclusions["invalid_id"] += 1
            continue
        if qid in seen_ids:
            raise MuSiQueSplitError("duplicate question ID")
        seen_ids.add(qid)
        diagnostic = diagnose_duplicate_titles(record)
        if not diagnostic.occurrence_structurally_eligible:
            exclusions["not_occurrence_structurally_eligible"] += 1
            continue
        if diagnostic.exact_text_mixed_support_label_group_count:
            exclusions["exact_text_mixed_support_label"] += 1
            continue
        if diagnostic.paragraph_count != 20:
            exclusions["not_exactly_20_paragraphs"] += 1
            continue
        if diagnostic.hop_count not in HOP_COUNTS:
            exclusions["unsupported_hop_count"] += 1
            continue
        if diagnostic.status not in TITLE_COLLISION_STRATA:
            exclusions["unsupported_collision_stratum"] += 1
            continue
        cell = cell_name(diagnostic.hop_count, diagnostic.status)
        ordered_ids.append(qid)
        by_cell[cell].append(qid)
    return CandidatePool(
        record_count=record_count,
        ordered_ids=tuple(ordered_ids),
        ids_by_cell={key: tuple(value) for key, value in by_cell.items()},
        exclusion_histogram=dict(sorted(exclusions.items())),
    )


def collect_candidate_file(path: Path) -> CandidatePool:
    if not path.is_file():
        raise FileNotFoundError(path)

    def records():
        for _line_number, record, error in iter_jsonl(path):
            if error is not None:
                raise MuSiQueSplitError("source contains invalid JSON")
            yield record

    return collect_candidate_records(records())


def _selection_rank(
    qid: str, *, split: str, cell: str, seed: int
) -> bytes:
    return hashlib.sha256(
        f"{seed}\0{split}\0{cell}\0{qid}".encode("utf-8")
    ).digest()


def select_train_ids(
    pool: CandidatePool,
    *,
    per_cell: int = TRAIN_PER_CELL,
    seed: int = SPLIT_SEED,
) -> tuple[tuple[str, ...], dict[str, int]]:
    if not isinstance(per_cell, int) or isinstance(per_cell, bool) or per_cell < 1:
        raise MuSiQueSplitError("per-cell sample size must be positive")
    selected: list[str] = []
    counts = {}
    for cell in all_cells():
        candidates = pool.ids_by_cell.get(cell, ())
        if len(candidates) < per_cell:
            raise MuSiQueSplitError(
                f"cell {cell} has {len(candidates)} candidates, needs {per_cell}"
            )
        ranked = sorted(
            candidates,
            key=lambda qid: (
                _selection_rank(qid, split="train", cell=cell, seed=seed),
                qid,
            ),
        )
        chosen = ranked[:per_cell]
        selected.extend(chosen)
        counts[cell] = len(chosen)
    if len(selected) != len(set(selected)):
        raise MuSiQueSplitError("selected train IDs are not unique")
    return tuple(selected), counts


def _cell_counts(pool: CandidatePool) -> dict[str, int]:
    return {cell: len(pool.ids_by_cell.get(cell, ())) for cell in all_cells()}


def build_split_bundle(
    train_pool: CandidatePool,
    dev_pool: CandidatePool,
    *,
    train_source_path: str,
    train_source_sha256: str,
    dev_source_path: str,
    dev_source_sha256: str,
    duplicate_audit: Mapping[str, Any],
    duplicate_audit_sha256: str,
    train_per_cell: int = TRAIN_PER_CELL,
    seed: int = SPLIT_SEED,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build deterministic ID-only manifests and a content-free audit report."""

    consistency = duplicate_audit.get("consistency_gate")
    duplicate_splits = duplicate_audit.get("splits")
    if not isinstance(consistency, Mapping) or not consistency.get("passed"):
        raise MuSiQueSplitError("duplicate-title consistency gate did not pass")
    if not isinstance(duplicate_splits, Mapping):
        raise MuSiQueSplitError("duplicate-title split reports are missing")
    for split, pool, source_sha256 in (
        ("train", train_pool, train_source_sha256),
        ("dev", dev_pool, dev_source_sha256),
    ):
        expected = duplicate_splits.get(split)
        if not isinstance(expected, Mapping):
            raise MuSiQueSplitError("duplicate-title split is missing")
        if expected.get("sha256") != source_sha256:
            raise MuSiQueSplitError("source hash changed after Stage B")
        if expected.get("record_count") != pool.record_count:
            raise MuSiQueSplitError("record count changed after Stage B")
        policy_counts = expected.get("policy_eligible_count")
        if not isinstance(policy_counts, Mapping) or policy_counts.get(
            "occurrence_fanout_label_unambiguous__exactly_20_paragraphs"
        ) != len(pool.ordered_ids):
            raise MuSiQueSplitError(
                "exact-20 candidate count changed after Stage B"
            )

    train_ids, selected_train_cells = select_train_ids(
        train_pool, per_cell=train_per_cell, seed=seed
    )
    dev_ids = dev_pool.ordered_ids
    if len(dev_ids) != len(set(dev_ids)):
        raise MuSiQueSplitError("dev IDs are not unique")
    train_manifest = {
        "schema_version": MUSIQUE_SPLIT_SCHEMA_VERSION,
        "dataset": "musique_ans_v1.0",
        "split": "train",
        "source_path": train_source_path,
        "source_sha256": train_source_sha256,
        "selection": {
            "seed": seed,
            "strata": "hop_count x title_collision_status",
            "per_cell": train_per_cell,
            "cell_order": list(all_cells()),
            "rank": "sha256(seed, split, cell, id), then id",
        },
        "selected_count": len(train_ids),
        "selected_count_by_cell": selected_train_cells,
        "ids": list(train_ids),
    }
    dev_manifest = {
        "schema_version": MUSIQUE_SPLIT_SCHEMA_VERSION,
        "dataset": "musique_ans_v1.0",
        "split": "dev",
        "source_path": dev_source_path,
        "source_sha256": dev_source_sha256,
        "selection": {
            "sampled": False,
            "rule": (
                "all occurrence-structurally-eligible records with exactly 20 "
                "paragraphs and no exact-text mixed-support-label collision"
            ),
            "order": "official JSONL order",
        },
        "selected_count": len(dev_ids),
        "candidate_count_by_cell": _cell_counts(dev_pool),
        "ids": list(dev_ids),
    }
    expected_train = len(all_cells()) * train_per_cell
    gate_checks = {
        "train_selected_count": len(train_ids) == expected_train,
        "train_every_cell_exact": all(
            value == train_per_cell for value in selected_train_cells.values()
        ),
        "dev_uses_all_exact20_candidates": (
            len(dev_ids) == sum(_cell_counts(dev_pool).values())
        ),
        "dev_every_cell_nonempty": all(
            value > 0 for value in _cell_counts(dev_pool).values()
        ),
        "no_exact_text_label_ambiguity_in_train_pool": (
            train_pool.exclusion_histogram.get(
                "exact_text_mixed_support_label", 0
            )
            == 0
        ),
        "no_exact_text_label_ambiguity_in_dev_pool": (
            dev_pool.exclusion_histogram.get(
                "exact_text_mixed_support_label", 0
            )
            == 0
        ),
    }
    report = {
        "schema_version": MUSIQUE_SPLIT_SCHEMA_VERSION,
        "task": "musique_hop_collision_stratified_split_materialization",
        "stage_b": {
            "sha256": duplicate_audit_sha256,
            "task": duplicate_audit.get("task"),
        },
        "protocol": {
            "declared_before_retrieval_scoring": True,
            "train_selection": (
                f"{train_per_cell} per each of 3 hop x 3 collision cells"
            ),
            "dev_selection": "all eligible exact-20 records",
            "primary_aggregate": "macro average across nine hop-collision cells",
            "secondary_aggregates": (
                "official-distribution micro, hop macro, and collision-stratum macro"
            ),
            "unique_title_sensitivity": (
                "the unique_titles cells use the same frozen manifests"
            ),
            "sampling_uses_gold_support_flags": (
                "only to stratify supporting-title collision status; never for "
                "retrieval scores, graph construction, or tie breaking"
            ),
        },
        "train": {
            "source_path": train_source_path,
            "source_sha256": train_source_sha256,
            "source_record_count": train_pool.record_count,
            "candidate_count": len(train_pool.ordered_ids),
            "candidate_count_by_cell": _cell_counts(train_pool),
            "exclusion_histogram": dict(train_pool.exclusion_histogram),
            "selected_count": len(train_ids),
            "selected_count_by_cell": selected_train_cells,
        },
        "dev": {
            "source_path": dev_source_path,
            "source_sha256": dev_source_sha256,
            "source_record_count": dev_pool.record_count,
            "candidate_count": len(dev_pool.ordered_ids),
            "candidate_count_by_cell": _cell_counts(dev_pool),
            "exclusion_histogram": dict(dev_pool.exclusion_histogram),
            "selected_count": len(dev_ids),
        },
        "gate": {
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
            "status": (
                "authorize_bge_m3_context_scoring"
                if all(gate_checks.values())
                else "stop_and_resolve_musique_split"
            ),
        },
        "content_contract": (
            "report contains aggregate counts, paths, and hashes only; manifests "
            "contain IDs and provenance only; no dataset text"
        ),
    }
    return train_manifest, dev_manifest, report
