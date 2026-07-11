"""Streaming HotpotQA inspection and deterministic diagnostic splits."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


OFFICIAL_SOURCE_FILENAMES = {
    "train": "hotpot_train_v1.1.json",
    "dev_distractor": "hotpot_dev_distractor_v1.json",
}

REQUIRED_FIELDS = frozenset(
    {"_id", "question", "answer", "context", "supporting_facts", "type", "level"}
)

VERIFIED_STRUCTURE = {
    "question": "string",
    "answer": "string",
    "context": "list[[title: string, sentences: list[string]]]",
    "supporting_facts": "list[[title: string, sentence_index: integer]]",
}

SAMPLING_ALGORITHM = "hamilton-proportional-allocation+sha256-rank-v1"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_hotpot_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream records from a top-level HotpotQA JSON array with ijson."""

    try:
        import ijson
    except ImportError as exc:  # pragma: no cover - exercised only in a broken env
        raise RuntimeError("ijson is required to stream HotpotQA files") from exc

    with Path(path).open("rb") as handle:
        for record in ijson.items(handle, "item"):
            if not isinstance(record, dict):
                raise ValueError("HotpotQA top-level array must contain JSON objects")
            yield record


def _validate_pair_list(
    value: Any,
    *,
    field_name: str,
    second_value_validator: Any,
) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{field_name} entries must be two-element lists")
        if not isinstance(item[0], str) or not second_value_validator(item[1]):
            raise ValueError(f"{field_name} entry has an invalid value type")


def validate_hotpot_record(record: Mapping[str, Any]) -> None:
    """Validate the fields used by this project without retaining record text."""

    missing = REQUIRED_FIELDS.difference(record)
    if missing:
        raise ValueError(f"HotpotQA record is missing fields: {sorted(missing)}")
    if not isinstance(record["_id"], str) or not record["_id"]:
        raise ValueError("HotpotQA _id must be a non-empty string")
    for field_name in ("question", "answer", "type", "level"):
        if not isinstance(record[field_name], str):
            raise ValueError(f"HotpotQA {field_name} must be a string")

    _validate_pair_list(
        record["context"],
        field_name="context",
        second_value_validator=lambda sentences: isinstance(sentences, list)
        and all(isinstance(sentence, str) for sentence in sentences),
    )
    _validate_pair_list(
        record["supporting_facts"],
        field_name="supporting_facts",
        second_value_validator=lambda index: isinstance(index, int)
        and not isinstance(index, bool)
        and index >= 0,
    )


def scan_hotpot_source(path: str | Path) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Inspect one source and retain only bridge IDs grouped by level."""

    source_path = Path(path)
    record_count = 0
    field_presence: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    bridge_by_level: Counter[str] = Counter()
    bridge_ids_by_level: defaultdict[str, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()

    for record in iter_hotpot_records(source_path):
        validate_hotpot_record(record)
        record_count += 1
        record_id = record["_id"]
        if record_id in seen_ids:
            raise ValueError("HotpotQA source contains a duplicate _id")
        seen_ids.add(record_id)
        field_presence.update(record.keys())
        question_type = record["type"]
        type_counts[question_type] += 1
        if question_type == "bridge":
            level = record["level"]
            bridge_by_level[level] += 1
            bridge_ids_by_level[level].append(record_id)

    if record_count == 0:
        raise ValueError("HotpotQA source is empty")

    stats = {
        "record_count": record_count,
        "fields": sorted(field_presence),
        "field_presence_counts": dict(sorted(field_presence.items())),
        "type_counts": dict(sorted(type_counts.items())),
        "bridge_count": sum(bridge_by_level.values()),
        "bridge_count_by_level": dict(sorted(bridge_by_level.items())),
        "verified_structure": VERIFIED_STRUCTURE,
    }
    return stats, dict(bridge_ids_by_level)


def proportional_allocation(
    counts_by_level: Mapping[str, int], sample_size: int
) -> dict[str, int]:
    """Allocate a fixed sample with Hamilton's largest-remainder method."""

    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    counts = {str(level): int(count) for level, count in counts_by_level.items()}
    if any(count < 0 for count in counts.values()):
        raise ValueError("stratum counts must be non-negative")
    total = sum(counts.values())
    if sample_size > total:
        raise ValueError(f"requested {sample_size} samples, but only {total} are available")
    if total == 0:
        return {level: 0 for level in sorted(counts)}

    allocation = {
        level: (sample_size * count) // total for level, count in counts.items()
    }
    remaining = sample_size - sum(allocation.values())
    remainder_order = sorted(
        counts,
        key=lambda level: (-(sample_size * counts[level] % total), level),
    )
    for level in remainder_order[:remaining]:
        allocation[level] += 1
    return dict(sorted(allocation.items()))


def _stable_sample_rank(*, record_id: str, level: str, split: str, seed: int) -> bytes:
    payload = f"{seed}\n{split}\n{level}\n{record_id}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def stratified_sample_ids(
    ids_by_level: Mapping[str, Sequence[str]],
    *,
    sample_size: int,
    seed: int,
    official_split: str,
) -> tuple[list[str], dict[str, int]]:
    """Choose deterministic bridge IDs while preserving level proportions."""

    counts = {level: len(ids) for level, ids in ids_by_level.items()}
    allocation = proportional_allocation(counts, sample_size)
    all_ids = [record_id for ids in ids_by_level.values() for record_id in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("candidate bridge IDs must be unique")

    selected: list[str] = []
    for level in sorted(ids_by_level):
        ranked = sorted(
            ids_by_level[level],
            key=lambda record_id: (
                _stable_sample_rank(
                    record_id=record_id,
                    level=level,
                    split=official_split,
                    seed=seed,
                ),
                record_id,
            ),
        )
        selected.extend(ranked[: allocation[level]])
    return sorted(selected), allocation


def validate_id_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_split: str,
    expected_size: int,
) -> None:
    """Check invariants needed to keep official diagnostic splits separate."""

    if manifest.get("official_split") != expected_split:
        raise ValueError("manifest official_split does not match the expected split")
    if manifest.get("selection", {}).get("predicate") != "type == 'bridge'":
        raise ValueError("manifest is not restricted to bridge questions")
    ids = manifest.get("ids")
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError("manifest ids must be a list of strings")
    if len(ids) != expected_size or len(ids) != len(set(ids)):
        raise ValueError("manifest IDs have the wrong size or contain duplicates")
    selected_by_level = manifest.get("selection", {}).get("selected_by_level", {})
    if sum(selected_by_level.values()) != expected_size:
        raise ValueError("selected_by_level does not sum to the requested size")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def prepare_hotpot_id_split(
    source_path: str | Path,
    output_path: str | Path,
    *,
    official_split: str,
    sample_size: int,
    seed: int,
) -> tuple[dict[str, Any], str]:
    """Scan, sample, and write one provenance-rich ID manifest."""

    if official_split not in OFFICIAL_SOURCE_FILENAMES:
        raise ValueError(f"unsupported official split: {official_split}")
    source_path = Path(source_path).resolve()
    expected_filename = OFFICIAL_SOURCE_FILENAMES[official_split]
    if source_path.name != expected_filename:
        raise ValueError(
            f"{official_split} must come from {expected_filename}, got {source_path.name}"
        )

    stat_before = source_path.stat()
    source_sha256 = sha256_file(source_path)
    stats, bridge_ids_by_level = scan_hotpot_source(source_path)
    stat_after = source_path.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (
        stat_after.st_size,
        stat_after.st_mtime_ns,
    ):
        raise RuntimeError("HotpotQA source changed while it was being processed")

    selected_ids, allocation = stratified_sample_ids(
        bridge_ids_by_level,
        sample_size=sample_size,
        seed=seed,
        official_split=official_split,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "HotpotQA",
        "official_split": official_split,
        "source": {
            "path": str(source_path),
            "filename": source_path.name,
            "size_bytes": stat_before.st_size,
            "sha256": source_sha256,
            **stats,
        },
        "selection": {
            "predicate": "type == 'bridge'",
            "sample_size": sample_size,
            "seed": seed,
            "stratified_by": "level",
            "algorithm": SAMPLING_ALGORITHM,
            "available_by_level": stats["bridge_count_by_level"],
            "selected_by_level": allocation,
        },
        "ids": selected_ids,
    }
    validate_id_manifest(
        manifest, expected_split=official_split, expected_size=sample_size
    )

    output_path = Path(output_path)
    _atomic_write_json(output_path, manifest)
    return manifest, sha256_file(output_path)


def selected_ids_are_bridge(
    source_path: str | Path, selected_ids: Iterable[str]
) -> bool:
    """Stream-check that selected IDs exist exactly once and are all bridge items."""

    selected = list(selected_ids)
    remaining = set(selected)
    if len(remaining) != len(selected):
        return False
    matched: set[str] = set()
    for record in iter_hotpot_records(source_path):
        record_id = record.get("_id")
        if record_id in remaining:
            if record.get("type") != "bridge" or record_id in matched:
                return False
            matched.add(record_id)
    return matched == remaining
