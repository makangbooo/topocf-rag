"""Private two-stage human adjudication for natural retrieval errors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .local_judge import (
    ALLOWED_LABELS,
    stable_json_sha256,
    validate_judgment,
    validate_private_audit_record,
)


HUMAN_AUDIT_SCHEMA_VERSION = 1
PRIVACY_MARKER = "private_text_bearing_artifact_do_not_commit"
EVENT_TYPES = ("blind_decision", "final_decision")
LABEL_DEFINITIONS = {
    "locally_true_globally_incomplete": (
        "At least one retrieved fact is true and relevant, but the retrieved "
        "documents do not contain a complete proof and no alternative proof exists."
    ),
    "complete_via_alternative_proof": (
        "The retrieved documents contain a complete alternative proof despite a "
        "missing annotated gold document."
    ),
    "missing_all_gold_evidence": (
        "No retrieved evidence provides a true, relevant local step toward the answer."
    ),
    "ambiguous_or_annotation_issue": (
        "The evidence or annotation does not permit a reliable decision."
    ),
    "other": "None of the label definitions applies.",
}


class HumanAuditInvariantError(ValueError):
    """Raised when a private artifact or two-stage decision is inconsistent."""


def _atomic_json_dump(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
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
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True, slots=True)
class HumanAuditBundle:
    records: tuple[Mapping[str, Any], ...]
    prelabels: Mapping[str, Mapping[str, Any]]
    input_sha256: str
    prelabels_sha256: str
    run_fingerprint: str

    @property
    def question_ids(self) -> tuple[str, ...]:
        return tuple(str(record["question_id"]) for record in self.records)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_objects(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise HumanAuditInvariantError(
                    f"invalid JSONL at line {line_number}: {path}"
                ) from error
            if not isinstance(payload, Mapping):
                raise HumanAuditInvariantError(
                    f"JSONL line {line_number} is not an object: {path}"
                )
            records.append(payload)
    return records


def load_human_audit_bundle(
    input_path: str | Path, prelabels_path: str | Path
) -> HumanAuditBundle:
    input_path = Path(input_path)
    prelabels_path = Path(prelabels_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not prelabels_path.is_file():
        raise FileNotFoundError(prelabels_path)

    source_records = _jsonl_objects(input_path)
    if not source_records:
        raise HumanAuditInvariantError("private audit input must not be empty")
    ordered_ids: list[str] = []
    source_by_id: dict[str, Mapping[str, Any]] = {}
    for record in source_records:
        validated = validate_private_audit_record(record)
        qid = str(validated["question_id"])
        if qid in source_by_id:
            raise HumanAuditInvariantError("private input has duplicate question IDs")
        ordered_ids.append(qid)
        source_by_id[qid] = validated

    prelabel_records = _jsonl_objects(prelabels_path)
    if len(prelabel_records) != len(source_records):
        raise HumanAuditInvariantError(
            "private input and prelabel record counts do not match"
        )
    prelabels: dict[str, Mapping[str, Any]] = {}
    prelabel_order: list[str] = []
    run_fingerprints: set[str] = set()
    for record in prelabel_records:
        if record.get("privacy") != PRIVACY_MARKER:
            raise HumanAuditInvariantError("prelabel has the wrong privacy marker")
        qid = record.get("question_id")
        if not isinstance(qid, str) or not qid or qid in prelabels:
            raise HumanAuditInvariantError(
                "prelabel question ID is missing or duplicated"
            )
        if qid not in source_by_id:
            raise HumanAuditInvariantError("prelabel ID is outside the private input")
        if record.get("input_sha256") != stable_json_sha256(source_by_id[qid]):
            raise HumanAuditInvariantError(
                "prelabel input hash does not match the private record"
            )
        run_fingerprint = record.get("run_fingerprint")
        if not isinstance(run_fingerprint, str) or not run_fingerprint:
            raise HumanAuditInvariantError("prelabel run fingerprint is invalid")
        run_fingerprints.add(run_fingerprint)
        status = record.get("status")
        if status not in ("valid", "invalid"):
            raise HumanAuditInvariantError("prelabel status is invalid")
        judgment = record.get("judgment")
        if status == "valid":
            normalized, errors = validate_judgment(judgment)
            if normalized is None or errors:
                raise HumanAuditInvariantError(
                    "valid prelabel has an invalid judgment payload"
                )
        elif judgment is not None:
            raise HumanAuditInvariantError("invalid prelabel must not have a judgment")
        prelabels[qid] = record
        prelabel_order.append(qid)
    if prelabel_order != ordered_ids:
        raise HumanAuditInvariantError(
            "prelabels must use exactly the private input order"
        )
    if len(run_fingerprints) != 1:
        raise HumanAuditInvariantError(
            "prelabels contain more than one run fingerprint"
        )
    return HumanAuditBundle(
        records=tuple(source_by_id[qid] for qid in ordered_ids),
        prelabels=prelabels,
        input_sha256=sha256_file(input_path),
        prelabels_sha256=sha256_file(prelabels_path),
        run_fingerprint=next(iter(run_fingerprints)),
    )


def _validate_label(label: Any) -> str:
    if label not in ALLOWED_LABELS:
        raise HumanAuditInvariantError("human label is not allowed")
    return str(label)


def _validate_notes(notes: Any) -> str:
    if not isinstance(notes, str) or len(notes) > 2000:
        raise HumanAuditInvariantError(
            "human notes must be a string of at most 2000 characters"
        )
    return notes


def load_annotation_events(
    path: str | Path, bundle: HumanAuditBundle
) -> list[Mapping[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    if not path.is_file():
        raise HumanAuditInvariantError("annotation path is not a regular file")
    events = _jsonl_objects(path)
    known_ids = set(bundle.question_ids)
    expected_sequence = 1
    for event in events:
        if event.get("schema_version") != HUMAN_AUDIT_SCHEMA_VERSION:
            raise HumanAuditInvariantError("unsupported annotation event schema")
        if event.get("privacy") != PRIVACY_MARKER:
            raise HumanAuditInvariantError(
                "annotation event has the wrong privacy marker"
            )
        if event.get("sequence") != expected_sequence:
            raise HumanAuditInvariantError(
                "annotation event sequence is missing or out of order"
            )
        expected_sequence += 1
        if event.get("event_type") not in EVENT_TYPES:
            raise HumanAuditInvariantError("annotation event type is invalid")
        if event.get("question_id") not in known_ids:
            raise HumanAuditInvariantError(
                "annotation event question ID is outside the private input"
            )
        if event.get("input_sha256") != bundle.input_sha256:
            raise HumanAuditInvariantError("annotation input hash does not match")
        if event.get("prelabels_sha256") != bundle.prelabels_sha256:
            raise HumanAuditInvariantError("annotation prelabel hash does not match")
        _validate_label(event.get("label"))
        _validate_notes(event.get("notes"))
        if not isinstance(event.get("created_at_utc"), str):
            raise HumanAuditInvariantError("annotation timestamp is invalid")
    build_annotation_state(events, bundle)
    return events


def build_annotation_state(
    events: Sequence[Mapping[str, Any]], bundle: HumanAuditBundle
) -> dict[str, dict[str, Mapping[str, Any] | None]]:
    state = {
        qid: {"blind": None, "final": None} for qid in bundle.question_ids
    }
    for event in events:
        qid = str(event["question_id"])
        event_type = event["event_type"]
        if event_type == "blind_decision":
            if state[qid]["blind"] is not None:
                raise HumanAuditInvariantError(
                    "a blind decision is immutable and may only be recorded once"
                )
            state[qid]["blind"] = event
        elif event_type == "final_decision":
            blind = state[qid]["blind"]
            if blind is None:
                raise HumanAuditInvariantError(
                    "final decision cannot precede the blind decision"
                )
            if event.get("blind_label_at_reveal") != blind["label"]:
                raise HumanAuditInvariantError(
                    "final decision blind-label snapshot does not match"
                )
            judgment = bundle.prelabels[qid].get("judgment")
            prelabel_label = (
                judgment.get("label") if isinstance(judgment, Mapping) else None
            )
            if event.get("prelabel_label_at_reveal") != prelabel_label:
                raise HumanAuditInvariantError(
                    "final decision prelabel snapshot does not match"
                )
            state[qid]["final"] = event
    return state


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _append_private_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    event,
                    sort_keys=True,
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o600)
    except BaseException:
        raise


def record_blind_decision(
    bundle: HumanAuditBundle,
    annotations_path: str | Path,
    *,
    question_id: str,
    label: str,
    notes: str = "",
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    path = Path(annotations_path)
    events = load_annotation_events(path, bundle)
    state = build_annotation_state(events, bundle)
    if question_id not in state:
        raise HumanAuditInvariantError("unknown question ID")
    if state[question_id]["blind"] is not None:
        raise HumanAuditInvariantError("blind decision is already locked")
    event = {
        "schema_version": HUMAN_AUDIT_SCHEMA_VERSION,
        "privacy": PRIVACY_MARKER,
        "sequence": len(events) + 1,
        "event_type": "blind_decision",
        "question_id": question_id,
        "input_sha256": bundle.input_sha256,
        "prelabels_sha256": bundle.prelabels_sha256,
        "label": _validate_label(label),
        "notes": _validate_notes(notes),
        "created_at_utc": created_at_utc or _utc_now(),
    }
    _append_private_event(path, event)
    return event


def record_final_decision(
    bundle: HumanAuditBundle,
    annotations_path: str | Path,
    *,
    question_id: str,
    label: str,
    notes: str = "",
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    path = Path(annotations_path)
    events = load_annotation_events(path, bundle)
    state = build_annotation_state(events, bundle)
    if question_id not in state:
        raise HumanAuditInvariantError("unknown question ID")
    blind = state[question_id]["blind"]
    if blind is None:
        raise HumanAuditInvariantError(
            "blind decision must be locked before the final decision"
        )
    judgment = bundle.prelabels[question_id].get("judgment")
    prelabel_label = judgment.get("label") if isinstance(judgment, Mapping) else None
    event = {
        "schema_version": HUMAN_AUDIT_SCHEMA_VERSION,
        "privacy": PRIVACY_MARKER,
        "sequence": len(events) + 1,
        "event_type": "final_decision",
        "question_id": question_id,
        "input_sha256": bundle.input_sha256,
        "prelabels_sha256": bundle.prelabels_sha256,
        "label": _validate_label(label),
        "notes": _validate_notes(notes),
        "blind_label_at_reveal": blind["label"],
        "prelabel_label_at_reveal": prelabel_label,
        "created_at_utc": created_at_utc or _utc_now(),
    }
    _append_private_event(path, event)
    return event


def audit_progress(
    bundle: HumanAuditBundle,
    state: Mapping[str, Mapping[str, Mapping[str, Any] | None]],
) -> dict[str, int | None]:
    blind_completed = sum(value["blind"] is not None for value in state.values())
    final_completed = sum(value["final"] is not None for value in state.values())
    next_incomplete_index = next(
        (
            index
            for index, qid in enumerate(bundle.question_ids)
            if state[qid]["final"] is None
        ),
        None,
    )
    return {
        "total": len(bundle.records),
        "blind_completed": blind_completed,
        "final_completed": final_completed,
        "next_incomplete_index": next_incomplete_index,
    }


def build_item_payload(
    bundle: HumanAuditBundle,
    state: Mapping[str, Mapping[str, Mapping[str, Any] | None]],
    index: int,
) -> dict[str, Any]:
    if not 0 <= index < len(bundle.records):
        raise HumanAuditInvariantError("item index is out of range")
    source = bundle.records[index]
    qid = str(source["question_id"])
    annotation = state[qid]
    blind = annotation["blind"]
    final = annotation["final"]
    prelabel = bundle.prelabels[qid]
    judgment = prelabel.get("judgment") if blind is not None else None
    return {
        "index": index,
        "total": len(bundle.records),
        "question": source["question"],
        "reference_answer": source["answer"],
        "retrieved_documents": source["retrieved_documents"],
        "gold_supporting_facts_for_adjudication": source[
            "gold_supporting_facts_for_adjudication"
        ],
        "label_definitions": LABEL_DEFINITIONS,
        "blind_decision": (
            {"label": blind["label"], "notes": blind["notes"]}
            if blind is not None
            else None
        ),
        "prelabel": (
            {
                "status": prelabel["status"],
                "judgment": judgment,
                "validation_errors": prelabel.get("validation_errors", []),
            }
            if blind is not None
            else None
        ),
        "final_decision": (
            {"label": final["label"], "notes": final["notes"]}
            if final is not None
            else None
        ),
        "progress": audit_progress(bundle, state),
    }


def build_public_human_audit_report(
    bundle: HumanAuditBundle,
    events: Sequence[Mapping[str, Any]],
    *,
    annotations_sha256: str | None,
) -> dict[str, Any]:
    state = build_annotation_state(events, bundle)
    prelabel_histogram: Counter[str] = Counter()
    blind_histogram: Counter[str] = Counter()
    final_histogram: Counter[str] = Counter()
    blind_agreement = 0
    final_agreement = 0
    changed_after_reveal = 0
    blind_count = 0
    final_count = 0
    for qid in bundle.question_ids:
        judgment = bundle.prelabels[qid].get("judgment")
        prelabel_label = judgment.get("label") if isinstance(judgment, Mapping) else None
        if isinstance(prelabel_label, str):
            prelabel_histogram[prelabel_label] += 1
        blind = state[qid]["blind"]
        final = state[qid]["final"]
        if blind is not None:
            blind_count += 1
            blind_label = str(blind["label"])
            blind_histogram[blind_label] += 1
            blind_agreement += blind_label == prelabel_label
        if final is not None:
            final_count += 1
            final_label = str(final["label"])
            final_histogram[final_label] += 1
            final_agreement += final_label == prelabel_label
            changed_after_reveal += final_label != state[qid]["blind"]["label"]

    def complete_histogram(counter: Counter[str]) -> dict[str, int]:
        return {label: counter[label] for label in ALLOWED_LABELS}

    return {
        "schema_version": HUMAN_AUDIT_SCHEMA_VERSION,
        "content_contract": (
            "aggregate counts, rates, protocol metadata, and hashes only; no question "
            "IDs, questions, answers, evidence text, notes, or model rationales"
        ),
        "protocol": {
            "name": "blind-human-then-model-reveal-v1",
            "blind_decision_immutable": True,
            "prelabel_hidden_until_blind_decision": True,
            "final_decision_may_be_revised": True,
            "allowed_labels": list(ALLOWED_LABELS),
        },
        "artifacts": {
            "private_input_sha256": bundle.input_sha256,
            "private_prelabels_sha256": bundle.prelabels_sha256,
            "private_annotations_sha256": annotations_sha256,
            "prelabel_run_fingerprint": bundle.run_fingerprint,
        },
        "counts": {
            "eligible_question_count": len(bundle.records),
            "annotation_event_count": len(events),
            "blind_completed_count": blind_count,
            "final_completed_count": final_count,
        },
        "label_histograms": {
            "qwen_prelabel": complete_histogram(prelabel_histogram),
            "blind_human": complete_histogram(blind_histogram),
            "final_human": complete_histogram(final_histogram),
        },
        "agreement": {
            "blind_vs_qwen": {
                "eligible_count": blind_count,
                "agreement_count": blind_agreement,
                "agreement_rate": blind_agreement / blind_count if blind_count else None,
            },
            "final_vs_qwen": {
                "eligible_count": final_count,
                "agreement_count": final_agreement,
                "agreement_rate": final_agreement / final_count if final_count else None,
            },
            "changed_after_qwen_reveal": {
                "eligible_count": final_count,
                "changed_count": changed_after_reveal,
                "changed_rate": changed_after_reveal / final_count if final_count else None,
            },
        },
    }


def write_public_human_audit_report(
    bundle: HumanAuditBundle,
    annotations_path: str | Path,
    report_path: str | Path,
) -> Mapping[str, Any]:
    annotations_path = Path(annotations_path)
    events = load_annotation_events(annotations_path, bundle)
    annotations_sha256 = (
        sha256_file(annotations_path) if annotations_path.is_file() else None
    )
    report = build_public_human_audit_report(
        bundle, events, annotations_sha256=annotations_sha256
    )
    _atomic_json_dump(report, report_path)
    return report
