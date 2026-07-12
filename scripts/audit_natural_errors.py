#!/usr/bin/env python3
"""Run the certificate-v1 controlled retrieval-failure screening audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import ijson

from topocf_rag.natural_error_audit import (
    NaturalErrorAuditInvariantError,
    audit_question,
    build_private_sample_record,
    build_public_audit_report,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_jsonl_dump(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                )
                stream.write("\n")
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_config(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise NaturalErrorAuditInvariantError("config must be a JSON object")
    if payload.get("schema_version") != 1:
        raise NaturalErrorAuditInvariantError("unsupported config schema version")
    return payload


def load_frozen_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("ids") if isinstance(payload, Mapping) else None
    if not isinstance(ids, list) or any(
        not isinstance(qid, str) or not qid for qid in ids
    ):
        raise NaturalErrorAuditInvariantError(
            "frozen ID manifest must contain non-empty string IDs"
        )
    if len(ids) != len(set(ids)):
        raise NaturalErrorAuditInvariantError("frozen ID manifest has duplicate IDs")
    return ids


def stream_selected_examples(
    source_path: Path, ordered_ids: Sequence[str]
) -> list[dict[str, Any]]:
    wanted = set(ordered_ids)
    selected: dict[str, dict[str, Any]] = {}
    with source_path.open("rb") as stream:
        for record in ijson.items(stream, "item"):
            qid = record.get("_id") if isinstance(record, Mapping) else None
            if qid not in wanted:
                continue
            if qid in selected:
                raise NaturalErrorAuditInvariantError(
                    "source contains a duplicate frozen question ID"
                )
            selected[qid] = record
    missing = wanted.difference(selected)
    if missing:
        raise NaturalErrorAuditInvariantError(
            f"source is missing {len(missing)} frozen questions"
        )
    return [selected[qid] for qid in ordered_ids]


def load_retrieval_records(
    path: Path,
    *,
    expected_source_sha256: str,
    expected_ids_sha256: str,
    expected_ids: Sequence[str],
) -> Mapping[str, Mapping[str, Sequence[int | float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise NaturalErrorAuditInvariantError("invalid retrieval cache schema")
    if payload.get("source_sha256") != expected_source_sha256:
        raise NaturalErrorAuditInvariantError(
            "retrieval cache source SHA256 does not match"
        )
    if payload.get("ids_sha256") != expected_ids_sha256:
        raise NaturalErrorAuditInvariantError(
            "retrieval cache frozen-ID SHA256 does not match"
        )
    records = payload.get("records")
    if not isinstance(records, Mapping):
        raise NaturalErrorAuditInvariantError(
            "retrieval cache must contain a records object"
        )
    expected_set = set(expected_ids)
    record_set = set(records)
    if record_set != expected_set:
        raise NaturalErrorAuditInvariantError(
            "retrieval cache IDs do not exactly match frozen IDs"
        )
    return records


def _required_string(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise NaturalErrorAuditInvariantError(f"config {key} must be a non-empty string")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/certificate_v1/natural_audit.json"),
    )
    parser.add_argument(
        "--private-sample-output",
        type=Path,
        required=True,
        help="Text-bearing JSONL output outside the Git repository.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    source_path = Path(_required_string(config, "source_path"))
    ids_path = Path(_required_string(config, "ids_path"))
    cache_path = Path(_required_string(config, "retrieval_cache_path"))
    public_output = Path(_required_string(config, "public_output"))
    private_output = args.private_sample_output.resolve()
    repository_root = Path.cwd().resolve()
    if private_output.is_relative_to(repository_root):
        raise NaturalErrorAuditInvariantError(
            "private text-bearing sample output must be outside the Git repository"
        )

    for path in (source_path, ids_path, cache_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_sha256 = sha256_file(source_path)
    ids_sha256 = sha256_file(ids_path)
    cache_sha256 = sha256_file(cache_path)
    config_sha256 = sha256_file(config_path)
    ordered_ids = load_frozen_ids(ids_path)
    examples = stream_selected_examples(source_path, ordered_ids)
    retrieval_records = load_retrieval_records(
        cache_path,
        expected_source_sha256=source_sha256,
        expected_ids_sha256=ids_sha256,
        expected_ids=ordered_ids,
    )

    top_ks = config.get("top_ks")
    if not isinstance(top_ks, list):
        raise NaturalErrorAuditInvariantError("config top_ks must be a list")
    primary_top_k = config.get("primary_top_k")
    sample_size = config.get("private_sample_size")
    sample_seed = config.get("sample_seed")
    screening_threshold = config.get("screening_threshold")
    if not isinstance(primary_top_k, int) or isinstance(primary_top_k, bool):
        raise NaturalErrorAuditInvariantError("config primary_top_k must be an integer")
    if not isinstance(sample_size, int) or isinstance(sample_size, bool):
        raise NaturalErrorAuditInvariantError(
            "config private_sample_size must be an integer"
        )
    if not isinstance(sample_seed, int) or isinstance(sample_seed, bool):
        raise NaturalErrorAuditInvariantError("config sample_seed must be an integer")
    if not isinstance(screening_threshold, (int, float)) or isinstance(
        screening_threshold, bool
    ):
        raise NaturalErrorAuditInvariantError(
            "config screening_threshold must be numeric"
        )

    audits = []
    for example in examples:
        qid = str(example["_id"])
        record = retrieval_records[qid]
        scores = record.get("scores") if isinstance(record, Mapping) else None
        token_lengths = (
            record.get("document_token_lengths")
            if isinstance(record, Mapping)
            else None
        )
        if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
            raise NaturalErrorAuditInvariantError(
                "retrieval cache record has invalid scores"
            )
        if not isinstance(token_lengths, Sequence) or isinstance(
            token_lengths, (str, bytes)
        ):
            raise NaturalErrorAuditInvariantError(
                "retrieval cache record has invalid document token lengths"
            )
        if len(scores) != len(token_lengths):
            raise NaturalErrorAuditInvariantError(
                "retrieval score/token-length counts disagree"
            )
        audits.append(
            audit_question(
                example,
                [float(score) for score in scores],
                top_ks=top_ks,
                primary_top_k=primary_top_k,
            )
        )

    public_report, sampled_audits = build_public_audit_report(
        audits,
        audit_id=_required_string(config, "audit_id"),
        dataset=_required_string(config, "dataset"),
        official_split=_required_string(config, "official_split"),
        top_ks=top_ks,
        primary_top_k=primary_top_k,
        sample_size=sample_size,
        sample_seed=sample_seed,
        screening_threshold=float(screening_threshold),
        source_sha256=source_sha256,
        ids_sha256=ids_sha256,
        retrieval_cache_sha256=cache_sha256,
        config_sha256=config_sha256,
    )
    example_by_qid = {str(example["_id"]): example for example in examples}
    private_records = [
        build_private_sample_record(
            example_by_qid[audit.qid],
            audit,
            primary_top_k=primary_top_k,
        )
        for audit in sampled_audits
    ]
    atomic_jsonl_dump(private_records, private_output)
    atomic_json_dump(public_report, public_output)
    print(
        json.dumps(
            {
                "audit_scope": public_report["audit_scope"],
                "by_top_k": public_report["by_top_k"],
                "private_sample_actual_size": len(private_records),
                "private_sample_output": str(private_output),
                "public_output": str(public_output.resolve()),
                "screening_gate": public_report["screening_gate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
