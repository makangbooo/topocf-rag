"""Streaming schema and eligibility audit for official MuSiQue v1.0."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterator

from .hotpot import sha256_file
from .title_normalization import normalize_title


MUSIQUE_SCHEMA_VERSION = 1
EXPECTED_SOURCE_SHA256 = {
    "train": "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a",
    "dev": "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b",
}


class MuSiQueInvariantError(ValueError):
    """Raised when a MuSiQue audit input violates a structural invariant."""


@dataclass(frozen=True, slots=True)
class MuSiQueAssessment:
    eligible: bool
    reasons: tuple[str, ...]
    hop_count: int | None
    paragraph_count: int | None
    supporting_paragraph_count: int | None


def iter_jsonl(path: Path) -> Iterator[tuple[int, Any | None, str | None]]:
    """Yield parsed JSONL records without retaining dataset text."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield line_number, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield line_number, None, type(exc).__name__


def assess_musique_record(record: Any) -> MuSiQueAssessment:
    """Apply the label-preserving structural eligibility contract."""

    reasons: set[str] = set()
    if not isinstance(record, Mapping):
        return MuSiQueAssessment(
            eligible=False,
            reasons=("non_object_record",),
            hop_count=None,
            paragraph_count=None,
            supporting_paragraph_count=None,
        )

    qid = record.get("id")
    if not isinstance(qid, str) or not qid:
        reasons.add("invalid_id")
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        reasons.add("invalid_question")
    answerable = record.get("answerable")
    if not isinstance(answerable, bool):
        reasons.add("invalid_answerable_flag")
    elif not answerable:
        reasons.add("not_answerable")

    paragraphs = record.get("paragraphs")
    paragraph_count = len(paragraphs) if isinstance(paragraphs, list) else None
    supporting_indices: set[int] = set()
    paragraph_indices: list[int] = []
    normalized_titles: list[str] = []
    if not isinstance(paragraphs, list) or not paragraphs:
        reasons.add("invalid_paragraphs")
    else:
        for paragraph in paragraphs:
            if not isinstance(paragraph, Mapping):
                reasons.add("malformed_paragraph")
                continue
            index = paragraph.get("idx")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                reasons.add("invalid_paragraph_index")
            else:
                paragraph_indices.append(index)
            title = paragraph.get("title")
            if not isinstance(title, str) or not title.strip():
                reasons.add("invalid_paragraph_title")
            else:
                normalized = normalize_title(title)
                if not normalized:
                    reasons.add("empty_normalized_title")
                normalized_titles.append(normalized)
            text = paragraph.get("paragraph_text")
            if not isinstance(text, str) or not text.strip():
                reasons.add("invalid_paragraph_text")
            is_supporting = paragraph.get("is_supporting")
            if not isinstance(is_supporting, bool):
                reasons.add("invalid_supporting_flag")
            elif is_supporting and isinstance(index, int) and not isinstance(
                index, bool
            ):
                supporting_indices.add(index)
        if len(paragraph_indices) != len(set(paragraph_indices)):
            reasons.add("duplicate_paragraph_index")
        if paragraph_indices != list(range(len(paragraphs))):
            reasons.add("paragraph_index_order_mismatch")
        if len(normalized_titles) != len(set(normalized_titles)):
            reasons.add("duplicate_normalized_title")

    decomposition = record.get("question_decomposition")
    hop_count = len(decomposition) if isinstance(decomposition, list) else None
    decomposition_support_indices: list[int] = []
    if not isinstance(decomposition, list) or not decomposition:
        reasons.add("invalid_question_decomposition")
    else:
        for step in decomposition:
            if not isinstance(step, Mapping):
                reasons.add("malformed_decomposition_step")
                continue
            step_id = step.get("id")
            if not isinstance(step_id, int) or isinstance(step_id, bool):
                reasons.add("invalid_decomposition_id")
            step_question = step.get("question")
            if not isinstance(step_question, str) or not step_question.strip():
                reasons.add("invalid_decomposition_question")
            step_answer = step.get("answer")
            if not isinstance(step_answer, str) or not step_answer.strip():
                reasons.add("invalid_decomposition_answer")
            support_index = step.get("paragraph_support_idx")
            if (
                not isinstance(support_index, int)
                or isinstance(support_index, bool)
                or support_index < 0
            ):
                reasons.add("invalid_decomposition_support_index")
            else:
                decomposition_support_indices.append(support_index)
        if len(decomposition_support_indices) != len(
            set(decomposition_support_indices)
        ):
            reasons.add("duplicate_decomposition_support_index")

    if paragraphs and isinstance(paragraphs, list):
        valid_indices = set(paragraph_indices)
        if any(
            index not in valid_indices for index in decomposition_support_indices
        ):
            reasons.add("decomposition_support_missing_from_paragraphs")
        if supporting_indices != set(decomposition_support_indices):
            reasons.add("support_flag_decomposition_mismatch")

    supporting_count = (
        len(supporting_indices) if isinstance(paragraphs, list) else None
    )
    return MuSiQueAssessment(
        eligible=not reasons,
        reasons=tuple(sorted(reasons)),
        hop_count=hop_count,
        paragraph_count=paragraph_count,
        supporting_paragraph_count=supporting_count,
    )


def _string_counter(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): counter[key]
        for key in sorted(counter, key=lambda value: str(value))
    }


def audit_musique_file(path: Path) -> dict[str, Any]:
    """Stream one official split and return content-free aggregate diagnostics."""

    if not path.is_file():
        raise FileNotFoundError(path)
    record_count = 0
    json_error_count = 0
    non_object_count = 0
    duplicate_id_count = 0
    seen_ids: set[str] = set()
    keysets: Counter[tuple[str, ...]] = Counter()
    answerable_types: Counter[str] = Counter()
    paragraph_counts: Counter[int | None] = Counter()
    hop_counts: Counter[int | None] = Counter()
    supporting_counts: Counter[int | None] = Counter()
    eligible_hop_counts: Counter[int | None] = Counter()
    eligible_pool_counts: Counter[int | None] = Counter()
    reason_counts: Counter[str] = Counter()
    reason_set_counts: Counter[tuple[str, ...]] = Counter()
    eligible_count = 0

    for _line_number, record, error in iter_jsonl(path):
        record_count += 1
        if error is not None:
            json_error_count += 1
            reason_counts["invalid_json"] += 1
            reason_set_counts[("invalid_json",)] += 1
            continue
        if not isinstance(record, Mapping):
            non_object_count += 1
        else:
            keysets[tuple(sorted(str(key) for key in record))] += 1
            answerable_types[type(record.get("answerable")).__name__] += 1
            qid = record.get("id")
            if isinstance(qid, str):
                if qid in seen_ids:
                    duplicate_id_count += 1
                seen_ids.add(qid)
        assessment = assess_musique_record(record)
        paragraph_counts[assessment.paragraph_count] += 1
        hop_counts[assessment.hop_count] += 1
        supporting_counts[assessment.supporting_paragraph_count] += 1
        if assessment.eligible:
            eligible_count += 1
            eligible_hop_counts[assessment.hop_count] += 1
            eligible_pool_counts[assessment.paragraph_count] += 1
        else:
            reason_set_counts[assessment.reasons] += 1
            reason_counts.update(assessment.reasons)

    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "record_count": record_count,
        "json_error_count": json_error_count,
        "non_object_count": non_object_count,
        "unique_id_count": len(seen_ids),
        "duplicate_id_count": duplicate_id_count,
        "eligible_count": eligible_count,
        "eligible_rate": eligible_count / record_count if record_count else None,
        "field_contract": {
            "keysets": [
                {"keys": list(keys), "count": count}
                for keys, count in sorted(keysets.items())
            ],
            "answerable_field_type_histogram": _string_counter(
                answerable_types
            ),
        },
        "histograms": {
            "paragraph_count": _string_counter(paragraph_counts),
            "hop_count": _string_counter(hop_counts),
            "supporting_paragraph_count": _string_counter(
                supporting_counts
            ),
            "eligible_hop_count": _string_counter(eligible_hop_counts),
            "eligible_paragraph_count": _string_counter(
                eligible_pool_counts
            ),
        },
        "exclusions": {
            "reason_occurrence_count": _string_counter(reason_counts),
            "reason_set_count": [
                {"reasons": list(reasons), "count": count}
                for reasons, count in sorted(reason_set_counts.items())
            ],
        },
    }


def build_musique_schema_report(
    train: Mapping[str, Any], dev: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the audit to the known official v1.0 answerable files."""

    splits = {"train": dict(train), "dev": dict(dev)}
    checks = {
        split: {
            "expected_sha256": EXPECTED_SOURCE_SHA256[split],
            "observed_sha256": report.get("sha256"),
            "source_hash_matches": (
                report.get("sha256") == EXPECTED_SOURCE_SHA256[split]
            ),
            "record_count_positive": bool(report.get("record_count", 0)),
            "json_valid": report.get("json_error_count") == 0,
            "ids_unique": report.get("duplicate_id_count") == 0,
            "has_structurally_eligible_records": bool(
                report.get("eligible_count", 0)
            ),
        }
        for split, report in splits.items()
    }
    passed = all(all(row.values()) for row in checks.values())
    return {
        "schema_version": MUSIQUE_SCHEMA_VERSION,
        "task": "musique_answerable_v1_schema_and_eligibility_audit",
        "scope": "official answerable train/dev controlled paragraph pools",
        "protocol": {
            "source_variant": "musique_ans_v1.0",
            "streaming_jsonl": True,
            "unknown_external_wikipedia_corpus_used": False,
            "sampling_or_scoring_performed": False,
            "gold_support_definition": (
                "exact equality between paragraph is_supporting indices and "
                "question_decomposition paragraph_support_idx values"
            ),
            "graph_identity_rule": "unique deterministic normalized paragraph titles",
        },
        "splits": splits,
        "schema_gate": {
            "checks": checks,
            "passed": passed,
            "status": (
                "inspect_histograms_then_freeze_sampling_protocol"
                if passed
                else "stop_and_resolve_musique_source_or_schema"
            ),
            "interpretation": (
                "Passing verifies source identity and structural auditability only. "
                "It does not authorize retrieval scoring until hop and paragraph-pool "
                "histograms are reviewed and a sampling protocol is frozen."
            ),
        },
        "content_contract": (
            "aggregate schema counts, paths, and hashes only; no IDs, questions, "
            "answers, titles, paragraph text, or decomposition text"
        ),
    }


def assessment_as_dict(assessment: MuSiQueAssessment) -> dict[str, Any]:
    """Expose a deterministic representation for unit tests and private tools."""

    return asdict(assessment)
