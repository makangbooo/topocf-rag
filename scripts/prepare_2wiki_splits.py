#!/usr/bin/env python3
"""Audit 2WikiMultiHopQA and freeze clean balanced train/dev ID splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from topocf_rag.twowiki import (
    ELIGIBILITY_PREDICATE,
    EXPECTED_GOLD_DOCUMENT_COUNTS,
    QUESTION_TYPES,
    SAMPLE_SEED,
    atomic_json_dump,
    prepare_twowiki_id_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("/file_system/datasets/2wikimultihopqa_official_20210407")
    parser.add_argument("--train-source", type=Path, default=root / "train.json")
    parser.add_argument("--dev-source", type=Path, default=root / "dev.json")
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/splits/2wiki_train_clean_balanced_1000_ids.json"),
    )
    parser.add_argument(
        "--dev-output",
        type=Path,
        default=Path("data/splits/2wiki_dev_clean_balanced_500_ids.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/graph_retrieval/2wiki_adapter_audit.json"),
    )
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    return parser.parse_args()


def _public_manifest_summary(
    manifest: dict[str, Any], output: Path, output_sha256: str
) -> dict[str, Any]:
    return {
        "official_split": manifest["official_split"],
        "source": manifest["source"],
        "selection": manifest["selection"],
        "ids_path": str(output.resolve()),
        "ids_sha256": output_sha256,
        "selected_id_count": len(manifest["ids"]),
    }


def main() -> int:
    args = parse_args()
    train_manifest, train_stats, train_output_sha256 = prepare_twowiki_id_split(
        args.train_source,
        args.train_output,
        official_split="train",
        sample_size=args.train_size,
        seed=args.seed,
    )
    dev_manifest, dev_stats, dev_output_sha256 = prepare_twowiki_id_split(
        args.dev_source,
        args.dev_output,
        official_split="dev",
        sample_size=args.dev_size,
        seed=args.seed,
    )
    overlap = set(train_manifest["ids"]).intersection(dev_manifest["ids"])
    if overlap:
        raise ValueError("official train and dev selected IDs overlap")

    report = {
        "schema_version": 1,
        "task": "2wiki_clean_graph_retrieval_adapter_audit",
        "protocol": {
            "eligibility_predicate": ELIGIBILITY_PREDICATE,
            "gold_mapping": "exact title plus sentence index",
            "normalization_policy": (
                "normalization is not used to repair gold labels; records with "
                "duplicate normalized context titles are excluded"
            ),
            "question_types": list(QUESTION_TYPES),
            "expected_gold_document_counts": EXPECTED_GOLD_DOCUMENT_COUNTS,
            "sampling": (
                "balanced across four question types, deterministic SHA256 rank"
            ),
            "official_splits_kept_separate": True,
        },
        "splits": {
            "train": {
                "audit": train_stats,
                "manifest": _public_manifest_summary(
                    train_manifest, args.train_output, train_output_sha256
                ),
            },
            "dev": {
                "audit": dev_stats,
                "manifest": _public_manifest_summary(
                    dev_manifest, args.dev_output, dev_output_sha256
                ),
            },
        },
        "content_contract": (
            "aggregate counts, protocol metadata, paths, and hashes only; no "
            "question IDs, questions, answers, titles, sentences, evidence text, "
            "or supporting facts"
        ),
    }
    atomic_json_dump(report, args.report)
    print(
        json.dumps(
            {
                "output": str(args.report.resolve()),
                "train": {
                    "eligible_count": train_stats["overall"]["eligible_count"],
                    "excluded_count": train_stats["overall"]["excluded_count"],
                    "selected_count": len(train_manifest["ids"]),
                    "ids_sha256": train_output_sha256,
                },
                "dev": {
                    "eligible_count": dev_stats["overall"]["eligible_count"],
                    "excluded_count": dev_stats["overall"]["excluded_count"],
                    "selected_count": len(dev_manifest["ids"]),
                    "ids_sha256": dev_output_sha256,
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
