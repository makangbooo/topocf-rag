#!/usr/bin/env python3
"""Privately prelabel natural-error audit items with local Qwen3-8B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from topocf_rag.evaluation import atomic_json_dump
from topocf_rag.local_judge import (
    PRELABEL_SCHEMA_VERSION,
    PROMPT_VERSION,
    LocalJudgeInvariantError,
    build_judge_messages,
    label_histogram,
    parse_judgment_text,
    prompt_template_sha256,
    stable_json_sha256,
    validate_private_audit_record,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise LocalJudgeInvariantError(
                    f"invalid JSONL at line {line_number}"
                ) from error
            record = validate_private_audit_record(payload)
            qid = str(record["question_id"])
            if qid in seen:
                raise LocalJudgeInvariantError("private input has duplicate question IDs")
            seen.add(qid)
            records.append(record)
    if not records:
        raise LocalJudgeInvariantError("private input must not be empty")
    return records


def load_completed(path: Path, run_fingerprint: str) -> list[Mapping[str, Any]]:
    if not path.is_file():
        return []
    completed: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise LocalJudgeInvariantError(
                    f"prelabel output line {line_number} is not an object"
                )
            if payload.get("run_fingerprint") != run_fingerprint:
                raise LocalJudgeInvariantError(
                    "existing prelabel output uses a different run fingerprint"
                )
            qid = payload.get("question_id")
            if not isinstance(qid, str) or not qid or qid in seen:
                raise LocalJudgeInvariantError(
                    "existing prelabel output has invalid/duplicate question IDs"
                )
            seen.add(qid)
            completed.append(payload)
    return completed


def append_private_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
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
    except BaseException:
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/file_system/models/general-models/Qwen3-8B"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = Path.cwd().resolve()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    model_path = args.model.resolve()
    if output_path.is_relative_to(repository_root):
        raise LocalJudgeInvariantError(
            "text-bearing prelabel output must be outside the Git repository"
        )
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if args.max_input_tokens < 1 or args.max_new_tokens < 1:
        raise LocalJudgeInvariantError("token limits must be positive")
    if args.limit is not None and args.limit < 1:
        raise LocalJudgeInvariantError("limit must be positive")
    if output_path.exists() and not args.resume:
        raise FileExistsError(
            f"private output exists; pass --resume or choose a new path: {output_path}"
        )

    records = load_jsonl(input_path)
    if args.limit is not None:
        records = records[: args.limit]
    config_sha256 = sha256_file(model_path / "config.json")
    tokenizer_config_sha256 = sha256_file(model_path / "tokenizer_config.json")
    run_fingerprint = stable_json_sha256(
        {
            "schema_version": PRELABEL_SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "prompt_template_sha256": prompt_template_sha256(),
            "model_config_sha256": config_sha256,
            "tokenizer_config_sha256": tokenizer_config_sha256,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "enable_thinking": False,
        }
    )
    completed = load_completed(output_path, run_fingerprint) if args.resume else []
    completed_ids = {str(record["question_id"]) for record in completed}
    input_ids = {str(record["question_id"]) for record in records}
    if not completed_ids.issubset(input_ids):
        raise LocalJudgeInvariantError(
            "existing prelabel output contains IDs outside the selected input"
        )
    input_hash_by_id = {
        str(record["question_id"]): stable_json_sha256(record) for record in records
    }
    for completed_record in completed:
        qid = str(completed_record["question_id"])
        if completed_record.get("input_sha256") != input_hash_by_id[qid]:
            raise LocalJudgeInvariantError(
                "existing prelabel output input hash does not match private input"
            )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Qwen prelabeling requires an available CUDA device")
    device_index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
    torch.manual_seed(20260713)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": device_index},
    )
    model.eval()
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()

    for source_record in records:
        qid = str(source_record["question_id"])
        if qid in completed_ids:
            continue
        messages = build_judge_messages(source_record)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_sha256 = stable_json_sha256(source_record)
        record_started = time.perf_counter()
        raw_output = ""
        judgment = None
        errors: tuple[str, ...]
        completion_tokens = 0
        if len(token_ids) > args.max_input_tokens:
            errors = ("input exceeds max_input_tokens",)
        else:
            model_inputs = torch.tensor(
                [token_ids], dtype=torch.long, device=args.device
            )
            attention_mask = torch.ones_like(model_inputs)
            try:
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=model_inputs,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                generated_ids = generated[0, model_inputs.shape[1] :]
                completion_tokens = int(generated_ids.numel())
                raw_output = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                judgment, errors = parse_judgment_text(raw_output)
            except RuntimeError as error:
                errors = (f"generation_runtime_error:{type(error).__name__}",)
        private_record = {
            "schema_version": PRELABEL_SCHEMA_VERSION,
            "privacy": "private_text_bearing_artifact_do_not_commit",
            "question_id": qid,
            "input_sha256": input_sha256,
            "run_fingerprint": run_fingerprint,
            "status": "valid" if judgment is not None else "invalid",
            "judgment": judgment,
            "validation_errors": list(errors),
            "raw_output": raw_output,
            "prompt_tokens": len(token_ids),
            "completion_tokens": completion_tokens,
            "elapsed_seconds": time.perf_counter() - record_started,
        }
        append_private_record(output_path, private_record)
        completed.append(private_record)
        completed_ids.add(qid)

    torch.cuda.synchronize(device_index)
    elapsed = time.perf_counter() - started
    valid_records = [record for record in completed if record.get("status") == "valid"]
    invalid_records = [
        record for record in completed if record.get("status") != "valid"
    ]
    report = {
        "schema_version": PRELABEL_SCHEMA_VERSION,
        "content_contract": (
            "aggregate counts, numeric runtime, generation configuration, and hashes "
            "only; no question IDs, questions, answers, documents, rationales, or raw "
            "model outputs"
        ),
        "model": {
            "name": model_path.name,
            "dtype": "bfloat16",
            "device_name": torch.cuda.get_device_name(device_index),
            "model_config_sha256": config_sha256,
            "tokenizer_config_sha256": tokenizer_config_sha256,
        },
        "protocol": {
            "prompt_version": PROMPT_VERSION,
            "prompt_template_sha256": prompt_template_sha256(),
            "run_fingerprint": run_fingerprint,
            "enable_thinking": False,
            "do_sample": False,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "resume": args.resume,
            "requested_limit": args.limit,
        },
        "input": {
            "private_input_sha256": sha256_file(input_path),
            "selected_record_count": len(records),
        },
        "output": {
            "private_output_sha256": sha256_file(output_path),
            "completed_count": len(completed),
            "valid_count": len(valid_records),
            "invalid_count": len(invalid_records),
            "valid_rate": len(valid_records) / len(completed) if completed else None,
            "label_histogram": label_histogram(valid_records),
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "mean_seconds_per_completed_record": (
                sum(float(record["elapsed_seconds"]) for record in completed)
                / len(completed)
                if completed
                else None
            ),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device_index),
            "prompt_token_count": {
                "min": min(int(record["prompt_tokens"]) for record in completed),
                "max": max(int(record["prompt_tokens"]) for record in completed),
                "mean": sum(int(record["prompt_tokens"]) for record in completed)
                / len(completed),
            },
            "completion_token_count": {
                "min": min(int(record["completion_tokens"]) for record in completed),
                "max": max(int(record["completion_tokens"]) for record in completed),
                "mean": sum(int(record["completion_tokens"]) for record in completed)
                / len(completed),
            },
        },
    }
    atomic_json_dump(report, report_path)
    print(json.dumps(report, sort_keys=True))
    return 0 if not invalid_records else 2


if __name__ == "__main__":
    sys.exit(main())
