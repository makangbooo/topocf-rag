#!/usr/bin/env python3
"""Build text-free bge-m3 caches for frozen MuSiQue paragraph pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from topocf_rag.hotpot import sha256_file
from topocf_rag.musique_retrieval import (
    BgeM3DenseEncoder,
    TOKENIZER_FILES,
    cache_matches,
    expected_cache_metadata,
    fingerprint_files,
    fingerprint_model,
    load_frozen_manifest,
    load_selected_examples,
    score_examples,
    validate_cache_records,
    validate_stage_c_audit,
)
from topocf_rag.twowiki import atomic_json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("/file_system/datasets/musique_official_v1.0")
    parser.add_argument(
        "--train-source",
        type=Path,
        default=root / "musique_ans_v1.0_train.jsonl",
    )
    parser.add_argument(
        "--dev-source",
        type=Path,
        default=root / "musique_ans_v1.0_dev.jsonl",
    )
    parser.add_argument(
        "--train-ids",
        type=Path,
        default=Path(
            "data/splits/musique_train_hop_collision_balanced_1800_ids.json"
        ),
    )
    parser.add_argument(
        "--dev-ids",
        type=Path,
        default=Path("data/splits/musique_dev_exact20_all_ids.json"),
    )
    parser.add_argument(
        "--split-audit",
        type=Path,
        default=Path("reports/musique/split_audit.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/file_system/models/embedding_models/bge-m3"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("reports/musique/cache/a100"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/musique/bge_cache_report.json"),
    )
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def _gpu_runtime(device: str) -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available() or not device.startswith("cuda"):
            return {"cuda_available": False}
        index = int(device.split(":", maxsplit=1)[1]) if ":" in device else 0
        torch.cuda.synchronize(index)
        return {
            "cuda_available": True,
            "device_index": index,
            "device_name": torch.cuda.get_device_name(index),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(index),
        }
    except (ImportError, RuntimeError, ValueError):
        return {"cuda_runtime_available": False}


def main() -> int:
    args = parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("batch size and maximum length must be positive")

    manifests = {
        "train": load_frozen_manifest(args.train_ids, expected_split="train"),
        "dev": load_frozen_manifest(args.dev_ids, expected_split="dev"),
    }
    validate_stage_c_audit(
        args.split_audit,
        train_manifest_path=args.train_ids,
        dev_manifest_path=args.dev_ids,
    )
    inputs = {
        "train": (args.train_source, args.train_ids),
        "dev": (args.dev_source, args.dev_ids),
    }
    encoder = BgeM3DenseEncoder(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    model_fingerprint_sha256 = fingerprint_model(args.model)
    tokenizer_fingerprint_sha256 = fingerprint_files(
        args.model, TOKENIZER_FILES
    )
    split_reports: dict[str, Any] = {}
    for split_name in ("train", "dev"):
        source, ids_path = inputs[split_name]
        examples = load_selected_examples(source, manifests[split_name])
        expected = expected_cache_metadata(
            split=split_name,
            source=source,
            ids_path=ids_path,
            split_audit_path=args.split_audit,
            model_path=args.model,
            model_fingerprint_sha256=model_fingerprint_sha256,
            tokenizer_fingerprint_sha256=tokenizer_fingerprint_sha256,
            max_length=args.max_length,
        )
        cache_path = args.cache_dir / (
            f"musique_{split_name}_bge_retrieval.json"
        )
        payload: Mapping[str, Any]
        runtime: dict[str, Any]
        if cache_path.is_file() and not args.force_recompute:
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(candidate, Mapping) or not cache_matches(
                candidate, expected
            ):
                raise ValueError(
                    f"existing cache metadata mismatch: {cache_path}; "
                    "use --force-recompute only after diagnosing provenance"
                )
            payload = candidate
            runtime = {"computed": False, "cache_hit": True}
        else:
            records, runtime = score_examples(examples, encoder)
            runtime.update(_gpu_runtime(args.device))
            runtime.update({"computed": True, "cache_hit": False})
            payload = {**expected, "records": records}
            atomic_json_dump(payload, cache_path)
        validate_cache_records(payload, examples)
        split_reports[split_name] = {
            "question_count": len(examples),
            "paragraph_occurrence_count": sum(
                len(example["paragraphs"]) for example in examples
            ),
            "cache_path": str(cache_path.resolve()),
            "cache_sha256": sha256_file(cache_path),
            "metadata": expected,
            "runtime": runtime,
        }
        del examples

    report = {
        "schema_version": 1,
        "task": "musique_frozen_paragraph_pool_bge_m3_scoring",
        "stage_c": {
            "path": str(args.split_audit.resolve()),
            "sha256": sha256_file(args.split_audit),
            "gate_verified": True,
        },
        "protocol": {
            "gold_labels_used_for_scoring": False,
            "paragraph_occurrence_identity": "official paragraph idx",
            "paragraph_order_preserved": True,
            "duplicate_titles_merged": False,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "use_fp16": True,
            "force_recompute": args.force_recompute,
            "evaluation_performed": False,
        },
        "model": {
            "path": str(args.model.resolve()),
            "fingerprint_sha256": model_fingerprint_sha256,
            "tokenizer_fingerprint_sha256": tokenizer_fingerprint_sha256,
        },
        "splits": split_reports,
        "content_contract": (
            "aggregate runtime, paths, hashes, and text-free numeric caches only; "
            "no questions, answers, titles, paragraph text, decomposition text, "
            "or supporting labels"
        ),
    }
    atomic_json_dump(report, args.report)
    print(
        json.dumps(
            {
                "output": str(args.report.resolve()),
                "stage_c": report["stage_c"],
                "splits": {
                    split_name: {
                        "question_count": row["question_count"],
                        "paragraph_occurrence_count": row[
                            "paragraph_occurrence_count"
                        ],
                        "cache_sha256": row["cache_sha256"],
                        "runtime": row["runtime"],
                    }
                    for split_name, row in split_reports.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
