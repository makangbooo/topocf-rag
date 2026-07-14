"""Aggregate duplicate-title and support-label audit for MuSiQue."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .hotpot import sha256_file
from .musique import assess_musique_record, iter_jsonl
from .title_normalization import normalize_title


DUPLICATE_AUDIT_SCHEMA_VERSION = 1


class MuSiQueDuplicateAuditError(ValueError):
    """Raised when a duplicate-title audit invariant is violated."""


@dataclass(frozen=True, slots=True)
class DuplicateTitleDiagnostic:
    occurrence_structurally_eligible: bool
    strict_unique_title_eligible: bool
    hop_count: int | None
    paragraph_count: int | None
    duplicate_group_count: int
    duplicate_occurrence_count: int
    maximum_title_multiplicity: int
    exact_text_group_count: int
    mixed_text_group_count: int
    supporting_duplicate_group_count: int
    exact_text_mixed_support_label_group_count: int
    mixed_text_supporting_group_count: int
    status: str


def diagnose_duplicate_titles(record: Any) -> DuplicateTitleDiagnostic:
    """Describe title collisions without retaining titles or paragraph text."""

    assessment = assess_musique_record(record)
    remaining_reasons = set(assessment.reasons) - {
        "duplicate_normalized_title"
    }
    occurrence_eligible = not remaining_reasons
    if not isinstance(record, Mapping):
        return DuplicateTitleDiagnostic(
            occurrence_structurally_eligible=False,
            strict_unique_title_eligible=False,
            hop_count=assessment.hop_count,
            paragraph_count=assessment.paragraph_count,
            duplicate_group_count=0,
            duplicate_occurrence_count=0,
            maximum_title_multiplicity=0,
            exact_text_group_count=0,
            mixed_text_group_count=0,
            supporting_duplicate_group_count=0,
            exact_text_mixed_support_label_group_count=0,
            mixed_text_supporting_group_count=0,
            status="structurally_invalid",
        )

    paragraphs = record.get("paragraphs")
    if not isinstance(paragraphs, list):
        paragraphs = []
    groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for paragraph in paragraphs:
        if not isinstance(paragraph, Mapping):
            continue
        title = paragraph.get("title")
        if isinstance(title, str) and title.strip():
            groups[normalize_title(title)].append(paragraph)
    duplicates = [group for group in groups.values() if len(group) > 1]
    exact_text_group_count = 0
    mixed_text_group_count = 0
    supporting_duplicate_group_count = 0
    exact_text_mixed_support_label_group_count = 0
    mixed_text_supporting_group_count = 0
    for group in duplicates:
        texts = {paragraph.get("paragraph_text") for paragraph in group}
        support_flags = {paragraph.get("is_supporting") for paragraph in group}
        contains_support = any(
            paragraph.get("is_supporting") is True for paragraph in group
        )
        exact_text = len(texts) == 1
        exact_text_group_count += exact_text
        mixed_text_group_count += not exact_text
        supporting_duplicate_group_count += contains_support
        exact_text_mixed_support_label_group_count += (
            exact_text and support_flags == {False, True}
        )
        mixed_text_supporting_group_count += (not exact_text) and contains_support

    if not occurrence_eligible:
        status = "structurally_invalid"
    elif not duplicates:
        status = "unique_titles"
    elif exact_text_mixed_support_label_group_count:
        status = "exact_text_mixed_support_label"
    elif mixed_text_supporting_group_count:
        status = "mixed_text_support_title_collision"
    elif supporting_duplicate_group_count:
        status = "support_title_collision_other"
    else:
        status = "distractor_only_title_collision"
    return DuplicateTitleDiagnostic(
        occurrence_structurally_eligible=occurrence_eligible,
        strict_unique_title_eligible=assessment.eligible,
        hop_count=assessment.hop_count,
        paragraph_count=assessment.paragraph_count,
        duplicate_group_count=len(duplicates),
        duplicate_occurrence_count=sum(len(group) for group in duplicates),
        maximum_title_multiplicity=max(
            (len(group) for group in duplicates), default=1
        ),
        exact_text_group_count=exact_text_group_count,
        mixed_text_group_count=mixed_text_group_count,
        supporting_duplicate_group_count=supporting_duplicate_group_count,
        exact_text_mixed_support_label_group_count=(
            exact_text_mixed_support_label_group_count
        ),
        mixed_text_supporting_group_count=mixed_text_supporting_group_count,
        status=status,
    )


def _string_counter(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): counter[key]
        for key in sorted(counter, key=lambda value: str(value))
    }


def _nested_counter(
    counters: Mapping[str, Counter[Any]],
) -> dict[str, dict[str, int]]:
    return {
        key: _string_counter(value) for key, value in sorted(counters.items())
    }


def audit_duplicate_titles(path: Path) -> dict[str, Any]:
    """Stream one split and compare strict and occurrence-aware policies."""

    if not path.is_file():
        raise FileNotFoundError(path)
    record_count = 0
    json_error_count = 0
    status_counts: Counter[str] = Counter()
    status_by_hop: defaultdict[str, Counter[int | None]] = defaultdict(Counter)
    duplicate_group_count = 0
    duplicate_occurrence_count = 0
    exact_text_group_count = 0
    mixed_text_group_count = 0
    supporting_duplicate_group_count = 0
    exact_text_mixed_support_label_group_count = 0
    mixed_text_supporting_group_count = 0
    duplicate_groups_per_record: Counter[int] = Counter()
    maximum_multiplicity: Counter[int] = Counter()
    policy_counts: Counter[str] = Counter()
    policy_hops: defaultdict[str, Counter[int | None]] = defaultdict(Counter)

    for _line_number, record, error in iter_jsonl(path):
        record_count += 1
        if error is not None:
            json_error_count += 1
            status_counts["invalid_json"] += 1
            continue
        diagnostic = diagnose_duplicate_titles(record)
        status_counts[diagnostic.status] += 1
        status_by_hop[diagnostic.status][diagnostic.hop_count] += 1
        duplicate_group_count += diagnostic.duplicate_group_count
        duplicate_occurrence_count += diagnostic.duplicate_occurrence_count
        exact_text_group_count += diagnostic.exact_text_group_count
        mixed_text_group_count += diagnostic.mixed_text_group_count
        supporting_duplicate_group_count += (
            diagnostic.supporting_duplicate_group_count
        )
        exact_text_mixed_support_label_group_count += (
            diagnostic.exact_text_mixed_support_label_group_count
        )
        mixed_text_supporting_group_count += (
            diagnostic.mixed_text_supporting_group_count
        )
        duplicate_groups_per_record[diagnostic.duplicate_group_count] += 1
        maximum_multiplicity[diagnostic.maximum_title_multiplicity] += 1

        policy_flags = {
            "strict_unique_title": diagnostic.strict_unique_title_eligible,
            "occurrence_fanout_all_structural": (
                diagnostic.occurrence_structurally_eligible
            ),
            "occurrence_fanout_label_unambiguous": (
                diagnostic.occurrence_structurally_eligible
                and not diagnostic.exact_text_mixed_support_label_group_count
            ),
        }
        for name, eligible in policy_flags.items():
            if eligible:
                policy_counts[name] += 1
                policy_hops[name][diagnostic.hop_count] += 1
                if diagnostic.paragraph_count == 20:
                    exact_name = f"{name}__exactly_20_paragraphs"
                    policy_counts[exact_name] += 1
                    policy_hops[exact_name][diagnostic.hop_count] += 1

    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "record_count": record_count,
        "json_error_count": json_error_count,
        "record_status_histogram": _string_counter(status_counts),
        "record_status_by_hop": _nested_counter(status_by_hop),
        "duplicate_group_aggregates": {
            "duplicate_group_count": duplicate_group_count,
            "duplicate_occurrence_count": duplicate_occurrence_count,
            "exact_text_group_count": exact_text_group_count,
            "mixed_text_group_count": mixed_text_group_count,
            "supporting_duplicate_group_count": (
                supporting_duplicate_group_count
            ),
            "exact_text_mixed_support_label_group_count": (
                exact_text_mixed_support_label_group_count
            ),
            "mixed_text_supporting_group_count": (
                mixed_text_supporting_group_count
            ),
            "duplicate_groups_per_record_histogram": _string_counter(
                duplicate_groups_per_record
            ),
            "maximum_title_multiplicity_histogram": _string_counter(
                maximum_multiplicity
            ),
        },
        "policy_eligible_count": _string_counter(policy_counts),
        "policy_eligible_hop_histogram": _nested_counter(policy_hops),
    }


def build_duplicate_audit_report(
    train: Mapping[str, Any],
    dev: Mapping[str, Any],
    *,
    schema_report: Mapping[str, Any],
    schema_report_sha256: str,
) -> dict[str, Any]:
    """Bind Stage B to the completed Stage A source audit."""

    schema_gate = schema_report.get("schema_gate")
    schema_splits = schema_report.get("splits")
    if not isinstance(schema_gate, Mapping) or not schema_gate.get("passed"):
        raise MuSiQueDuplicateAuditError("Stage A schema gate did not pass")
    if not isinstance(schema_splits, Mapping):
        raise MuSiQueDuplicateAuditError("Stage A split reports are missing")
    splits = {"train": dict(train), "dev": dict(dev)}
    checks = {}
    for split, report in splits.items():
        expected = schema_splits.get(split)
        if not isinstance(expected, Mapping):
            raise MuSiQueDuplicateAuditError("Stage A split is missing")
        checks[split] = {
            "source_hash_matches_stage_a": (
                report.get("sha256") == expected.get("sha256")
            ),
            "record_count_matches_stage_a": (
                report.get("record_count") == expected.get("record_count")
            ),
            "json_valid": report.get("json_error_count") == 0,
        }
    passed = all(all(values.values()) for values in checks.values())
    return {
        "schema_version": DUPLICATE_AUDIT_SCHEMA_VERSION,
        "task": "musique_duplicate_title_and_support_label_audit",
        "scope": "official answerable train/dev paragraph pools",
        "stage_a": {
            "sha256": schema_report_sha256,
            "task": schema_report.get("task"),
        },
        "protocol": {
            "declared_after_unique_title_exclusion_rate_was_observed": True,
            "sampling_or_scoring_performed": False,
            "exact_text_definition": "exact official paragraph_text string equality",
            "occurrence_graph_candidate_rule": (
                "paragraph idx is the node identity; a literal normalized title "
                "mention fans out deterministically to every matching occurrence"
            ),
            "planned_primary_pool": (
                "occurrence_fanout_label_unambiguous__exactly_20_paragraphs"
            ),
            "planned_sensitivity_pool": (
                "strict_unique_title__exactly_20_paragraphs"
            ),
            "primary_exclusion": (
                "an exact-text duplicate-title group mixes supporting and "
                "non-supporting labels"
            ),
        },
        "splits": splits,
        "consistency_gate": {
            "checks": checks,
            "passed": passed,
            "status": (
                "inspect_duplicate_strata_then_freeze_sample_sizes"
                if passed
                else "stop_and_resolve_stage_a_stage_b_mismatch"
            ),
        },
        "content_contract": (
            "aggregate counts, paths, and hashes only; no IDs, questions, answers, "
            "titles, paragraph text, or decomposition text"
        ),
    }


def diagnostic_as_dict(
    diagnostic: DuplicateTitleDiagnostic,
) -> dict[str, Any]:
    return asdict(diagnostic)
