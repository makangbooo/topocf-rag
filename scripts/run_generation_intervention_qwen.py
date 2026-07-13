#!/usr/bin/env python3
"""Run private paired evidence interventions with local Qwen3-8B."""

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
from topocf_rag.generation_intervention import (
    CONDITIONS,
    GENERATION_PROMPT_VERSION,
    INTERVENTION_PROTOCOL_VERSION,
    INTERVENTION_SCHEMA_VERSION,
    GenerationInterventionInvariantError,
    build_generation_messages,
    build_public_generation_report,
    construct_evidence_conditions,
    generation_prompt_template_sha256,
    parse_generation_text,
    select_confirmed_records,
)
from topocf_rag.hotpot import iter_hotpot_records
from topocf_rag.human_audit import (
    PRIVACY_MARKER,
    load_annotation_events,
    load_human_audit_bundle,
    sha256_file,
)
from topocf_rag.local_judge import stable_json_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prelabels", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/file_system/models/general-models/Qwen3-8B"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _append_private_record(path: Path, record: Mapping[str, Any]) -> None:
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


def _load_completed(
    path: Path,
    *,
    run_fingerprint: str,
    task_hashes: Mapping[tuple[str, str], str],
) -> list[Mapping[str, Any]]:
    if not path.is_file():
        return []
    records: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise GenerationInterventionInvariantError(
                    f"invalid private generation JSONL at line {line_number}"
                ) from error
            if not isinstance(record, Mapping):
                raise GenerationInterventionInvariantError(
                    "private generation output line is not an object"
                )
            if record.get("privacy") != PRIVACY_MARKER:
                raise GenerationInterventionInvariantError(
                    "private generation output has the wrong privacy marker"
                )
            if record.get("run_fingerprint") != run_fingerprint:
                raise GenerationInterventionInvariantError(
                    "private generation output has a different run fingerprint"
                )
            key = (str(record.get("question_id")), str(record.get("condition")))
            if key in seen or key not in task_hashes:
                raise GenerationInterventionInvariantError(
                    "private generation output has a duplicate or unknown task"
                )
            if record.get("condition_input_sha256") != task_hashes[key]:
                raise GenerationInterventionInvariantError(
                    "private generation task hash does not match"
                )
            seen.add(key)
            records.append(record)
    return records


def _stream_selected_source(
    path: Path, ordered_ids: tuple[str, ...]
) -> dict[str, Mapping[str, Any]]:
    wanted = set(ordered_ids)
    selected: dict[str, Mapping[str, Any]] = {}
    for record in iter_hotpot_records(path):
        qid = record.get("_id")
        if qid not in wanted:
            continue
        if qid in selected:
            raise GenerationInterventionInvariantError(
                "official source contains a duplicate selected question ID"
            )
        selected[str(qid)] = record
    missing = wanted.difference(selected)
    if missing:
        raise GenerationInterventionInvariantError(
            f"official source is missing {len(missing)} selected questions"
        )
    return selected


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    input_path = args.input.resolve()
    prelabels_path = args.prelabels.resolve()
    annotations_path = args.annotations.resolve()
    source_path = args.source.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    model_path = args.model.resolve()
    for name, path in (
        ("private input", input_path),
        ("private prelabels", prelabels_path),
        ("private annotations", annotations_path),
        ("private generation output", output_path),
    ):
        if path.is_relative_to(repository_root):
            raise GenerationInterventionInvariantError(
                f"{name} must be outside the Git repository"
            )
    for path in (
        input_path,
        prelabels_path,
        annotations_path,
        source_path,
        model_path / "config.json",
        model_path / "tokenizer_config.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.max_input_tokens < 1 or args.max_new_tokens < 1:
        raise GenerationInterventionInvariantError(
            "generation token limits must be positive"
        )
    if args.limit_questions is not None and args.limit_questions < 1:
        raise GenerationInterventionInvariantError(
            "limit-questions must be positive"
        )
    if output_path.exists() and not args.resume:
        raise FileExistsError(
            f"private output exists; pass --resume or choose a new path: {output_path}"
        )

    bundle = load_human_audit_bundle(input_path, prelabels_path)
    events = load_annotation_events(annotations_path, bundle)
    selected_records = list(select_confirmed_records(bundle, events))
    if args.limit_questions is not None:
        selected_records = selected_records[: args.limit_questions]
    ordered_ids = tuple(str(record["question_id"]) for record in selected_records)
    source_by_id = _stream_selected_source(source_path, ordered_ids)
    conditions_by_id = {
        qid: construct_evidence_conditions(
            next(
                record
                for record in selected_records
                if str(record["question_id"]) == qid
            ),
            source_by_id[qid],
        )
        for qid in ordered_ids
    }
    selected_by_id = {
        str(record["question_id"]): record for record in selected_records
    }
    tasks = [
        (qid, condition, conditions_by_id[qid][condition])
        for qid in ordered_ids
        for condition in CONDITIONS
    ]
    task_hashes = {
        (qid, condition): stable_json_sha256(
            {
                "protocol_version": INTERVENTION_PROTOCOL_VERSION,
                "question": selected_by_id[qid]["question"],
                "condition": condition,
                "documents": documents,
            }
        )
        for qid, condition, documents in tasks
    }

    config_sha256 = sha256_file(model_path / "config.json")
    tokenizer_config_sha256 = sha256_file(model_path / "tokenizer_config.json")
    generation_config_sha256 = _optional_sha256(
        model_path / "generation_config.json"
    )
    safetensors_index_sha256 = _optional_sha256(
        model_path / "model.safetensors.index.json"
    )
    selected_sequence_sha256 = hashlib.sha256(
        json.dumps(ordered_ids, ensure_ascii=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    run_fingerprint = stable_json_sha256(
        {
            "schema_version": INTERVENTION_SCHEMA_VERSION,
            "protocol_version": INTERVENTION_PROTOCOL_VERSION,
            "prompt_version": GENERATION_PROMPT_VERSION,
            "prompt_template_sha256": generation_prompt_template_sha256(),
            "selected_question_sequence_sha256": selected_sequence_sha256,
            "model_config_sha256": config_sha256,
            "tokenizer_config_sha256": tokenizer_config_sha256,
            "generation_config_sha256": generation_config_sha256,
            "safetensors_index_sha256": safetensors_index_sha256,
            "dtype": "bfloat16",
            "enable_thinking": False,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
        }
    )
    completed = (
        _load_completed(
            output_path,
            run_fingerprint=run_fingerprint,
            task_hashes=task_hashes,
        )
        if args.resume
        else []
    )
    completed_keys = {
        (str(record["question_id"]), str(record["condition"]))
        for record in completed
    }

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("local generation intervention requires CUDA")
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
        dtype=torch.bfloat16,
        device_map={"": device_index},
    )
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.eval()
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    generated_this_invocation = 0

    for qid, condition, documents in tasks:
        key = (qid, condition)
        if key in completed_keys:
            continue
        messages = build_generation_messages(
            str(selected_by_id[qid]["question"]), documents
        )
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response = None
        raw_output = ""
        errors: tuple[str, ...]
        completion_tokens = 0
        record_started = time.perf_counter()
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
                response, errors = parse_generation_text(
                    raw_output, document_count=len(documents)
                )
            except RuntimeError as error:
                errors = (f"generation_runtime_error:{type(error).__name__}",)
        private_record = {
            "schema_version": INTERVENTION_SCHEMA_VERSION,
            "privacy": PRIVACY_MARKER,
            "question_id": qid,
            "condition": condition,
            "condition_input_sha256": task_hashes[key],
            "document_count": len(documents),
            "gold_document_numbers": [
                int(document["slot"])
                for document in documents
                if document.get("is_gold") is True
            ],
            "run_fingerprint": run_fingerprint,
            "status": "valid" if response is not None else "invalid",
            "response": response,
            "validation_errors": list(errors),
            "raw_output": raw_output,
            "prompt_tokens": len(token_ids),
            "completion_tokens": completion_tokens,
            "elapsed_seconds": time.perf_counter() - record_started,
        }
        _append_private_record(output_path, private_record)
        completed.append(private_record)
        completed_keys.add(key)
        generated_this_invocation += 1
        if len(completed_keys) % 10 == 0 or len(completed_keys) == len(tasks):
            print(
                json.dumps(
                    {
                        "progress_completed_condition_count": len(completed_keys),
                        "progress_total_condition_count": len(tasks),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    torch.cuda.synchronize(device_index)
    invocation_elapsed = time.perf_counter() - started
    invalid_count = sum(record.get("status") != "valid" for record in completed)
    runtime = {
        "invocation_elapsed_seconds": invocation_elapsed,
        "generated_this_invocation": generated_this_invocation,
        "mean_seconds_per_condition_record": (
            sum(float(record["elapsed_seconds"]) for record in completed)
            / len(completed)
        ),
        "peak_memory_bytes_this_invocation": torch.cuda.max_memory_allocated(
            device_index
        ),
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
    }
    artifact_hashes = {
        "official_source_sha256": sha256_file(source_path),
        "private_input_sha256": sha256_file(input_path),
        "private_prelabels_sha256": sha256_file(prelabels_path),
        "private_annotations_sha256": sha256_file(annotations_path),
        "private_generation_output_sha256": sha256_file(output_path),
    }
    model_report = {
        "name": model_path.name,
        "dtype": "bfloat16",
        "device_name": torch.cuda.get_device_name(device_index),
        "model_config_sha256": config_sha256,
        "tokenizer_config_sha256": tokenizer_config_sha256,
        "generation_config_sha256": generation_config_sha256,
        "safetensors_index_sha256": safetensors_index_sha256,
    }
    protocol_report = {
        "prompt_version": GENERATION_PROMPT_VERSION,
        "prompt_template_sha256": generation_prompt_template_sha256(),
        "run_fingerprint": run_fingerprint,
        "conditions": list(CONDITIONS),
        "partial_budget": 5,
        "repair_policy": (
            "replace the lowest-retrieval-rank non-gold document with the missing "
            "gold document while preserving the five-document slot budget"
        ),
        "oracle_policy": "the two official gold supporting documents only",
        "enable_thinking": False,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "resume": args.resume,
        "requested_question_limit": args.limit_questions,
    }
    report = build_public_generation_report(
        completed,
        selected_records,
        artifact_hashes=artifact_hashes,
        model=model_report,
        protocol=protocol_report,
        runtime=runtime,
    )
    atomic_json_dump(report, report_path)
    print(json.dumps(report, sort_keys=True))
    return 0 if invalid_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
