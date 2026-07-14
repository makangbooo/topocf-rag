#!/usr/bin/env python3
"""Run post-primary robustness controls for 2Wiki graph retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from evaluate_2wiki_graph_retrieval import prepare_split
from topocf_rag.hotpot import sha256_file
from topocf_rag.twowiki import atomic_json_dump
from topocf_rag.twowiki_graph_validation import (
    PERMUTATION_REPETITIONS,
    PERMUTATION_SEED,
    build_validation_report,
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
        "--train-cache",
        type=Path,
        default=Path(
            "reports/graph_retrieval/cache/a100/"
            "2wiki_train_bge_retrieval.json"
        ),
    )
    parser.add_argument(
        "--dev-cache",
        type=Path,
        default=Path(
            "reports/graph_retrieval/cache/a100/"
            "2wiki_dev_bge_retrieval.json"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/file_system/models/embedding_models/bge-m3"),
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--base-report",
        type=Path,
        default=Path("reports/graph_retrieval/2wiki_graph_retrieval.json"),
    )
    parser.add_argument(
        "--permutation-repetitions",
        type=int,
        default=PERMUTATION_REPETITIONS,
    )
    parser.add_argument(
        "--permutation-seed", type=int, default=PERMUTATION_SEED
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/graph_retrieval/2wiki_graph_retrieval_validation.json"
        ),
    )
    parser.add_argument("--no-fail-on-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.base_report.is_file():
        raise FileNotFoundError(args.base_report)
    base_report_sha256 = sha256_file(args.base_report)
    base_report = json.loads(args.base_report.read_text(encoding="utf-8"))
    train_questions, train_provenance = prepare_split(
        args.train_source,
        args.train_ids,
        args.train_cache,
        model_path=args.model,
        max_length=args.max_length,
    )
    dev_questions, dev_provenance = prepare_split(
        args.dev_source,
        args.dev_ids,
        args.dev_cache,
        model_path=args.model,
        max_length=args.max_length,
    )
    report = build_validation_report(
        train_questions,
        dev_questions,
        base_report=base_report,
        base_report_sha256=base_report_sha256,
        provenance={
            "base_report_path": str(args.base_report.resolve()),
            "train": train_provenance,
            "dev": dev_provenance,
        },
        permutation_repetitions=args.permutation_repetitions,
        permutation_seed=args.permutation_seed,
    )
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "selected_degree_control": {
                    "name": report["train"]["selected_degree_only_name"],
                    "config": report["train"]["selected_degree_only_config"],
                },
                "strongest_baseline_comparison": report["dev"][
                    "actual_vs_strongest_baseline"
                ],
                "degree_comparison": report["dev"]["actual_vs_degree_only"],
                "permutation_macro_mean": report["dev"][
                    "node_permutation_null"
                ]["macro_mean_complete_gold_evidence_rate"],
                "gate": report["gate"],
            },
            sort_keys=True,
        )
    )
    if report["gate"]["passed"] or args.no_fail_on_gate:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
