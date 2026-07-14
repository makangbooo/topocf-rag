"""Leakage-safe 2WikiMultiHopQA structure audit and frozen split sampling.

Only aggregate counts and selected question IDs are serialized. Question,
answer, context, evidence, and supporting-fact text stay in memory.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .hotpot import sha256_file
from .title_normalization import normalize_title


DATASET_NAME = "2wikimultihopqa"
SCHEMA_VERSION = 1
SAMPLE_SEED = 20260714
QUESTION_TYPES = (
    "bridge_comparison",
    "comparison",
    "compositional",
    "inference",
)
EXPECTED_GOLD_DOCUMENT_COUNTS = {
    "bridge_comparison": 4,
    "comparison": 2,
    "compositional": 2,
    "inference": 2,
}
OFFICIAL_SOURCE_FILENAMES = {"train": "train.json", "dev": "dev.json"}
EXPECTED_CONTEXT_DOCUMENT_COUNT = 10
SAMPLING_ALGORITHM = "balanced-question-type+sha256-rank-v1"
ELIGIBILITY_PREDICATE = (
    "context has exactly 10 documents; normalized context titles are unique; "
    "every supporting fact resolves by exact title to exactly one context "
    "document with an in-range sentence index; and the unique supporting "
    "document count matches the frozen question-type contract"
)


class TwoWikiInvariantError(ValueError):
    """Raised when 2Wiki input violates a frozen adapter invariant."""


@dataclass(frozen=True, slots=True)
class EligibilityAssessment:
    """Text-free structural assessment for one official record."""

    question_type: str
    reasons: tuple[str, ...]
    gold_document_count: int
    unresolved_support_fact_count: int
    ambiguous_exact_support_fact_count: int

    @property
    def eligible(self) -> bool:
        return not self.reasons


def atomic_json_dump(payload: Mapping[str, Any], path: str | Path) -> None:
    """Write one JSON object atomically with private filesystem permissions."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def iter_twowiki_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream a top-level official 2Wiki JSON array without retaining it."""

    try:
        import ijson
    except ImportError as exc:  # pragma: no cover - broken environment only
        raise RuntimeError("ijson is required to stream 2Wiki files") from exc

    with Path(path).open("rb") as stream:
        for record in ijson.items(stream, "item"):
            if not isinstance(record, dict):
                raise TwoWikiInvariantError(
                    "2Wiki top-level array must contain objects"
                )
            yield record


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TwoWikiInvariantError(f"{field} must be a list")
    return value


def validate_twowiki_record(record: Mapping[str, Any]) -> None:
    """Validate fields without including private text in error messages."""

    if not isinstance(record, Mapping):
        raise TwoWikiInvariantError("record must be a mapping")
    if not isinstance(record.get("_id"), str) or not record["_id"]:
        raise TwoWikiInvariantError("_id must be a non-empty string")
    for field in ("question", "answer", "type"):
        if not isinstance(record.get(field), str):
            raise TwoWikiInvariantError(f"{field} must be a string")

    context = _require_list(record.get("context"), "context")
    for document_index, document in enumerate(context):
        if not isinstance(document, list) or len(document) != 2:
            raise TwoWikiInvariantError(
                f"context[{document_index}] must contain title and sentences"
            )
        title, sentences = document
        if not isinstance(title, str):
            raise TwoWikiInvariantError(
                f"context[{document_index}].title must be a string"
            )
        try:
            normalize_title(title)
        except (TypeError, ValueError) as exc:
            raise TwoWikiInvariantError(
                f"context[{document_index}].title is invalid"
            ) from exc
        if not isinstance(sentences, list) or any(
            not isinstance(sentence, str) for sentence in sentences
        ):
            raise TwoWikiInvariantError(
                f"context[{document_index}].sentences must contain strings"
            )

    supporting_facts = _require_list(
        record.get("supporting_facts"), "supporting_facts"
    )
    if not supporting_facts:
        raise TwoWikiInvariantError("supporting_facts must not be empty")
    for fact_index, fact in enumerate(supporting_facts):
        if not isinstance(fact, list) or len(fact) != 2:
            raise TwoWikiInvariantError(
                f"supporting_facts[{fact_index}] must contain title and index"
            )
        title, sentence_index = fact
        if not isinstance(title, str):
            raise TwoWikiInvariantError(
                f"supporting_facts[{fact_index}].title must be a string"
            )
        if (
            not isinstance(sentence_index, int)
            or isinstance(sentence_index, bool)
            or sentence_index < 0
        ):
            raise TwoWikiInvariantError(
                f"supporting_facts[{fact_index}].index must be non-negative int"
            )


def assess_twowiki_eligibility(
    record: Mapping[str, Any],
) -> EligibilityAssessment:
    """Apply the frozen label-integrity predicate to one record."""

    validate_twowiki_record(record)
    context = record["context"]
    question_type = str(record["type"])
    reasons: list[str] = []

    if len(context) != EXPECTED_CONTEXT_DOCUMENT_COUNT:
        reasons.append("unexpected_context_document_count")

    normalized_titles = [normalize_title(document[0]) for document in context]
    if len(normalized_titles) != len(set(normalized_titles)):
        reasons.append("normalized_duplicate_context_title")

    exact_occurrences: defaultdict[str, list[int]] = defaultdict(list)
    for document_index, (title, _sentences) in enumerate(context):
        exact_occurrences[title].append(document_index)

    unresolved = 0
    ambiguous = 0
    resolved_gold_indices: list[int] = []
    for title, sentence_index in record["supporting_facts"]:
        valid_occurrences = [
            document_index
            for document_index in exact_occurrences.get(title, ())
            if sentence_index < len(context[document_index][1])
        ]
        if not valid_occurrences:
            unresolved += 1
        elif len(valid_occurrences) > 1:
            ambiguous += 1
        else:
            resolved_gold_indices.append(valid_occurrences[0])

    if unresolved:
        reasons.append("unresolved_support_fact_exact_title")
    if ambiguous:
        reasons.append("ambiguous_support_fact_exact_title")

    gold_document_count = len(set(resolved_gold_indices))
    expected_gold_count = EXPECTED_GOLD_DOCUMENT_COUNTS.get(question_type)
    if expected_gold_count is None:
        reasons.append("unsupported_question_type")
    elif gold_document_count != expected_gold_count:
        reasons.append("unexpected_gold_document_count")

    return EligibilityAssessment(
        question_type=question_type,
        reasons=tuple(sorted(set(reasons))),
        gold_document_count=gold_document_count,
        unresolved_support_fact_count=unresolved,
        ambiguous_exact_support_fact_count=ambiguous,
    )


def _new_counter() -> dict[str, Any]:
    return {
        "record_count": 0,
        "eligible_count": 0,
        "excluded_count": 0,
        "exclusion_reason_question_counts": Counter(),
        "exclusion_reason_combination_histogram": Counter(),
        "gold_document_count_histogram": Counter(),
        "eligible_gold_document_count_histogram": Counter(),
        "unresolved_support_fact_count": 0,
        "ambiguous_exact_support_fact_count": 0,
    }


def _update_counter(
    counter: dict[str, Any], assessment: EligibilityAssessment
) -> None:
    counter["record_count"] += 1
    counter["gold_document_count_histogram"][str(assessment.gold_document_count)] += 1
    counter["unresolved_support_fact_count"] += (
        assessment.unresolved_support_fact_count
    )
    counter["ambiguous_exact_support_fact_count"] += (
        assessment.ambiguous_exact_support_fact_count
    )
    if assessment.eligible:
        counter["eligible_count"] += 1
        counter["eligible_gold_document_count_histogram"][
            str(assessment.gold_document_count)
        ] += 1
        return
    counter["excluded_count"] += 1
    for reason in assessment.reasons:
        counter["exclusion_reason_question_counts"][reason] += 1
    counter["exclusion_reason_combination_histogram"][
        "+".join(assessment.reasons)
    ] += 1


def _finalize_counter(counter: Mapping[str, Any]) -> dict[str, Any]:
    record_count = int(counter["record_count"])
    eligible_count = int(counter["eligible_count"])
    return {
        "record_count": record_count,
        "eligible_count": eligible_count,
        "eligible_rate": eligible_count / record_count if record_count else None,
        "excluded_count": int(counter["excluded_count"]),
        "exclusion_reason_question_counts": dict(
            sorted(counter["exclusion_reason_question_counts"].items())
        ),
        "exclusion_reason_combination_histogram": dict(
            sorted(counter["exclusion_reason_combination_histogram"].items())
        ),
        "gold_document_count_histogram": dict(
            sorted(counter["gold_document_count_histogram"].items())
        ),
        "eligible_gold_document_count_histogram": dict(
            sorted(counter["eligible_gold_document_count_histogram"].items())
        ),
        "unresolved_support_fact_count": int(
            counter["unresolved_support_fact_count"]
        ),
        "ambiguous_exact_support_fact_count": int(
            counter["ambiguous_exact_support_fact_count"]
        ),
    }


def scan_twowiki_source(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Return aggregate eligibility statistics and eligible IDs by type."""

    overall = _new_counter()
    by_type: defaultdict[str, dict[str, Any]] = defaultdict(_new_counter)
    eligible_ids: defaultdict[str, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()

    for record in iter_twowiki_records(path):
        assessment = assess_twowiki_eligibility(record)
        qid = record["_id"]
        if qid in seen_ids:
            raise TwoWikiInvariantError("source contains duplicate question IDs")
        seen_ids.add(qid)
        _update_counter(overall, assessment)
        _update_counter(by_type[assessment.question_type], assessment)
        if assessment.eligible:
            eligible_ids[assessment.question_type].append(qid)

    if not seen_ids:
        raise TwoWikiInvariantError("source is empty")
    stats = {
        "overall": _finalize_counter(overall),
        "by_question_type": {
            question_type: _finalize_counter(by_type[question_type])
            for question_type in sorted(by_type)
        },
    }
    return stats, dict(eligible_ids)


def balanced_allocation(
    question_types: Sequence[str], sample_size: int
) -> dict[str, int]:
    """Allocate samples evenly; lexicographic order resolves any remainder."""

    types = tuple(sorted(set(question_types)))
    if not types:
        raise TwoWikiInvariantError("question types must not be empty")
    if sample_size < len(types):
        raise TwoWikiInvariantError(
            "sample size must include at least one item per question type"
        )
    quotient, remainder = divmod(sample_size, len(types))
    return {
        question_type: quotient + (index < remainder)
        for index, question_type in enumerate(types)
    }


def _sample_rank(
    *, qid: str, question_type: str, official_split: str, seed: int
) -> bytes:
    return hashlib.sha256(
        f"{seed}\0{official_split}\0{question_type}\0{qid}".encode("utf-8")
    ).digest()


def stratified_sample_ids(
    eligible_ids_by_type: Mapping[str, Sequence[str]],
    *,
    sample_size: int,
    seed: int,
    official_split: str,
) -> tuple[list[str], dict[str, int]]:
    """Select a balanced deterministic sample across the four frozen types."""

    if set(eligible_ids_by_type) != set(QUESTION_TYPES):
        raise TwoWikiInvariantError(
            "eligible pool must contain exactly the four frozen question types"
        )
    allocation = balanced_allocation(QUESTION_TYPES, sample_size)
    all_ids = [qid for ids in eligible_ids_by_type.values() for qid in ids]
    if len(all_ids) != len(set(all_ids)):
        raise TwoWikiInvariantError("eligible question IDs must be unique")

    selected: list[str] = []
    for question_type in QUESTION_TYPES:
        candidates = eligible_ids_by_type[question_type]
        requested = allocation[question_type]
        if len(candidates) < requested:
            raise TwoWikiInvariantError(
                f"question type {question_type} has too few eligible records"
            )
        ranked = sorted(
            candidates,
            key=lambda qid: (
                _sample_rank(
                    qid=qid,
                    question_type=question_type,
                    official_split=official_split,
                    seed=seed,
                ),
                qid,
            ),
        )
        selected.extend(ranked[:requested])
    return sorted(selected), allocation


def validate_id_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_split: str,
    expected_size: int,
) -> None:
    if manifest.get("dataset") != DATASET_NAME:
        raise TwoWikiInvariantError("manifest dataset is invalid")
    if manifest.get("official_split") != expected_split:
        raise TwoWikiInvariantError("manifest official split is invalid")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise TwoWikiInvariantError("manifest selection is invalid")
    if selection.get("eligibility_predicate") != ELIGIBILITY_PREDICATE:
        raise TwoWikiInvariantError("manifest eligibility predicate is invalid")
    if selection.get("sampling_algorithm") != SAMPLING_ALGORITHM:
        raise TwoWikiInvariantError("manifest sampling algorithm is invalid")
    ids = manifest.get("ids")
    if not isinstance(ids, list) or len(ids) != expected_size:
        raise TwoWikiInvariantError("manifest sample size is invalid")
    if any(not isinstance(qid, str) or not qid for qid in ids):
        raise TwoWikiInvariantError("manifest contains an invalid ID")
    if len(ids) != len(set(ids)):
        raise TwoWikiInvariantError("manifest contains duplicate IDs")
    selected_by_type = selection.get("selected_by_question_type")
    if not isinstance(selected_by_type, Mapping):
        raise TwoWikiInvariantError("manifest type allocation is invalid")
    if sum(selected_by_type.values()) != expected_size:
        raise TwoWikiInvariantError("manifest type allocation has wrong size")


def prepare_twowiki_id_split(
    source: str | Path,
    output: str | Path,
    *,
    official_split: str,
    sample_size: int,
    seed: int = SAMPLE_SEED,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Audit one official split, freeze selected IDs, and return provenance."""

    source_path = Path(source)
    output_path = Path(output)
    expected_filename = OFFICIAL_SOURCE_FILENAMES.get(official_split)
    if expected_filename is None:
        raise TwoWikiInvariantError("official split must be train or dev")
    if source_path.name != expected_filename:
        raise TwoWikiInvariantError(
            f"{official_split} must come from {expected_filename}"
        )
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    stats, eligible_ids_by_type = scan_twowiki_source(source_path)
    ids, allocation = stratified_sample_ids(
        eligible_ids_by_type,
        sample_size=sample_size,
        seed=seed,
        official_split=official_split,
    )
    source_sha256 = sha256_file(source_path)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "official_split": official_split,
        "source": {
            "path": str(source_path.resolve()),
            "sha256": source_sha256,
            "size_bytes": source_path.stat().st_size,
            "record_count": stats["overall"]["record_count"],
        },
        "selection": {
            "seed": seed,
            "sample_size": sample_size,
            "sampling_algorithm": SAMPLING_ALGORITHM,
            "eligibility_predicate": ELIGIBILITY_PREDICATE,
            "gold_mapping": "exact title plus sentence index",
            "title_normalization_use": (
                "duplicate context-title detection and graph construction only"
            ),
            "selected_by_question_type": allocation,
            "eligible_by_question_type": {
                question_type: len(eligible_ids_by_type[question_type])
                for question_type in QUESTION_TYPES
            },
        },
        "ids": ids,
    }
    validate_id_manifest(
        manifest, expected_split=official_split, expected_size=sample_size
    )
    atomic_json_dump(manifest, output_path)
    return manifest, stats, sha256_file(output_path)
