#!/usr/bin/env python3
"""Independently adjudicate human-reviewed natural errors through one API model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import httpx

from topocf_rag.api_judge import (
    API_JUDGE_PROMPT_VERSION,
    API_JUDGE_PROTOCOL_VERSION,
    API_JUDGE_SCHEMA_VERSION,
    ApiJudgeInvariantError,
    api_judge_prompt_template_sha256,
    api_judge_run_fingerprint,
    build_public_api_judge_report,
    execute_api_judge_call,
    select_final_human_records,
)
from topocf_rag.evaluation import atomic_json_dump
from topocf_rag.human_audit import (
    PRIVACY_MARKER,
    load_annotation_events,
    load_human_audit_bundle,
    sha256_file,
)
from topocf_rag.local_judge import (
    build_judge_messages,
    stable_json_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prelabels", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-split", required=True)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-new-records", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _append_private_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                record,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)


def _load_completed(
    path: Path,
    *,
    run_fingerprint: str,
    selected: tuple[Mapping[str, Any], ...],
) -> list[Mapping[str, Any]]:
    if not path.is_file():
        return []
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ApiJudgeInvariantError(
                    f"invalid API judgment JSONL at line {line_number}"
                ) from error
            if not isinstance(record, Mapping):
                raise ApiJudgeInvariantError("API judgment line is not an object")
            records.append(record)
    if len(records) > len(selected):
        raise ApiJudgeInvariantError("API output has more records than selected input")
    for index, record in enumerate(records):
        source = selected[index]
        if record.get("privacy") != PRIVACY_MARKER:
            raise ApiJudgeInvariantError("API output has the wrong privacy marker")
        if record.get("run_fingerprint") != run_fingerprint:
            raise ApiJudgeInvariantError("API output uses a different run fingerprint")
        if record.get("question_id") != source["question_id"]:
            raise ApiJudgeInvariantError("API output is not an input-order prefix")
        if record.get("input_sha256") != stable_json_sha256(source):
            raise ApiJudgeInvariantError("API output input hash does not match")
    return records


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    input_path = args.input.resolve()
    prelabels_path = args.prelabels.resolve()
    annotations_path = args.annotations.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    for name, path in (
        ("private input", input_path),
        ("private prelabels", prelabels_path),
        ("private annotations", annotations_path),
        ("private API output", output_path),
    ):
        if path.is_relative_to(repository_root):
            raise ApiJudgeInvariantError(f"{name} must be outside the Git repository")
    for path in (input_path, prelabels_path, annotations_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.max_completion_tokens < 1:
        raise ApiJudgeInvariantError("max completion tokens must be positive")
    if args.timeout_seconds <= 0:
        raise ApiJudgeInvariantError("timeout must be positive")
    if args.max_new_records is not None and args.max_new_records < 1:
        raise ApiJudgeInvariantError("max new records must be positive")
    if output_path.exists() and not args.resume:
        raise FileExistsError(
            f"private output exists; pass --resume or choose a new path: {output_path}"
        )

    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = args.model or os.environ.get("OPENAI_MODEL", "")
    missing = []
    if not base_url:
        missing.append("OPENAI_BASE_URL")
    if not api_key:
        missing.append("OPENAI_API_KEY")
    if not model:
        missing.append("--model or OPENAI_MODEL")
    if missing:
        print(json.dumps({"missing": missing}, sort_keys=True))
        return 2

    bundle = load_human_audit_bundle(input_path, prelabels_path)
    events = load_annotation_events(annotations_path, bundle)
    selected = select_final_human_records(bundle, events)
    selected_sequence_sha256 = hashlib.sha256(
        json.dumps(
            [record["question_id"] for record in selected],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    run_fingerprint = api_judge_run_fingerprint(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        model=model,
        private_input_sha256=bundle.input_sha256,
        private_annotations_sha256=sha256_file(annotations_path),
        selected_sequence_sha256=selected_sequence_sha256,
        max_completion_tokens=args.max_completion_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    completed = (
        _load_completed(
            output_path,
            run_fingerprint=run_fingerprint,
            selected=selected,
        )
        if args.resume
        else []
    )
    remaining = selected[len(completed) :]
    if args.max_new_records is not None:
        remaining = remaining[: args.max_new_records]

    started = time.perf_counter()
    generated_this_invocation = 0
    if remaining:
        limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)
        timeout = httpx.Timeout(args.timeout_seconds)
        with httpx.Client(timeout=timeout, limits=limits) as client:
            for source in remaining:
                result = execute_api_judge_call(
                    client,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=build_judge_messages(source),
                    max_completion_tokens=args.max_completion_tokens,
                )
                record = {
                    "schema_version": API_JUDGE_SCHEMA_VERSION,
                    "privacy": PRIVACY_MARKER,
                    "question_id": source["question_id"],
                    "input_sha256": stable_json_sha256(source),
                    "run_fingerprint": run_fingerprint,
                    "model": model,
                    "status": (
                        "valid" if result["judgment"] is not None else "invalid"
                    ),
                    **result,
                }
                _append_private_record(output_path, record)
                completed.append(record)
                generated_this_invocation += 1
                print(
                    json.dumps(
                        {
                            "progress_completed_count": len(completed),
                            "progress_eligible_count": len(selected),
                            "last_status": record["status"],
                            "last_status_code": record["status_code"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    invocation_elapsed = time.perf_counter() - started
    protocol = {
        "version": API_JUDGE_PROTOCOL_VERSION,
        "prompt_version": API_JUDGE_PROMPT_VERSION,
        "prompt_template_sha256": api_judge_prompt_template_sha256(),
        "run_fingerprint": run_fingerprint,
        "temperature": 0,
        "max_completion_tokens": args.max_completion_tokens,
        "timeout_seconds": args.timeout_seconds,
        "concurrency": 1,
        "automatic_retries": 0,
        "human_and_qwen_labels_hidden_from_api": True,
        "resume": args.resume,
        "max_new_records_this_invocation": args.max_new_records,
    }
    artifact_hashes = {
        "private_input_sha256": bundle.input_sha256,
        "private_prelabels_sha256": bundle.prelabels_sha256,
        "private_annotations_sha256": sha256_file(annotations_path),
        "private_api_output_sha256": sha256_file(output_path),
        "selected_sequence_sha256": selected_sequence_sha256,
    }
    report = build_public_api_judge_report(
        completed,
        bundle,
        events,
        artifact_hashes=artifact_hashes,
        model=model,
        dataset={"name": args.dataset_name, "split": args.dataset_split},
        protocol=protocol,
        runtime={
            "invocation_elapsed_seconds": invocation_elapsed,
            "generated_this_invocation": generated_this_invocation,
        },
    )
    atomic_json_dump(report, report_path)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["counts"]["api_invalid_count"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
