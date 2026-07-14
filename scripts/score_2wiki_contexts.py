#!/usr/bin/env python3
"""Build text-free bge-m3 retrieval caches for frozen 2Wiki splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from topocf_rag.hotpot import sha256_file
from topocf_rag.twowiki import atomic_json_dump
from topocf_rag.twowiki_retrieval import (
    BgeM3DenseEncoder,
    cache_matches,
    expected_cache_metadata,
    load_frozen_manifest,
    load_selected_examples,
    score_examples,
    validate_cache_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("/file_system/datasets/2wikimultihopqa_official_20210407")
    parser.add_argument("--train-source", type=Path, default=root / "train.json")
    parser.add_argument("--dev-source", type=Path, default=root / "dev.json")
    parser.add_argument(
        "--train-ids",
        type=Path,
        default=Path("data/splits/2wiki_train_clean_balanced_1000_ids.json"),
    )
    parser.add_argument(
        "--dev-ids",
        type=Path,
        default=Path("data/splits/2wiki_dev_clean_balanced_500_ids.json"),
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
        default=Path("reports/graph_retrieval/cache"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/graph_retrieval/2wiki_bge_cache_report.json"),
    )
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def _gpu_runtime() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        index = torch.cuda.current_device()
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

    split_inputs = {
        "train": (args.train_source, args.train_ids),
        "dev": (args.dev_source, args.dev_ids),
    }
    split_examples: dict[str, tuple[dict[str, Any], ...]] = {}
    split_metadata: dict[str, dict[str, Any]] = {}
    for split_name, (source, ids_path) in split_inputs.items():
        manifest = load_frozen_manifest(ids_path)
        if manifest["official_split"] != split_name:
            raise ValueError("manifest split does not match command input")
        split_examples[split_name] = load_selected_examples(source, manifest)
        split_metadata[split_name] = expected_cache_metadata(
            source=source,
            ids_path=ids_path,
            model_path=args.model,
            max_length=args.max_length,
        )

    encoder = BgeM3DenseEncoder(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    split_reports: dict[str, Any] = {}
    for split_name in ("train", "dev"):
        cache_path = args.cache_dir / f"2wiki_{split_name}_bge_retrieval.json"
        expected = split_metadata[split_name]
        payload: Mapping[str, Any] | None = None
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
            records, runtime = score_examples(split_examples[split_name], encoder)
            runtime.update(_gpu_runtime())
            runtime.update({"computed": True, "cache_hit": False})
            payload = {**expected, "records": records}
            atomic_json_dump(payload, cache_path)
        validate_cache_records(payload, split_examples[split_name])
        split_reports[split_name] = {
            "question_count": len(split_examples[split_name]),
            "document_count": sum(
                len(example["context"]) for example in split_examples[split_name]
            ),
            "cache_path": str(cache_path.resolve()),
            "cache_sha256": sha256_file(cache_path),
            "metadata": expected,
            "runtime": runtime,
        }

    report = {
        "schema_version": 1,
        "task": "2wiki_frozen_context_pool_bge_m3_scoring",
        "protocol": {
            "gold_labels_used_for_scoring": False,
            "context_order_preserved": True,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "use_fp16": True,
            "force_recompute": args.force_recompute,
        },
        "splits": split_reports,
        "content_contract": (
            "aggregate runtime, paths, hashes, and text-free numeric caches only; "
            "no questions, answers, titles, sentences, evidence text, or labels"
        ),
    }
    atomic_json_dump(report, args.report)
    print(
        json.dumps(
            {
                "output": str(args.report.resolve()),
                "splits": {
                    split_name: {
                        "question_count": row["question_count"],
                        "document_count": row["document_count"],
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
