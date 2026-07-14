#!/usr/bin/env python3
"""Diagnose the 2Wiki receiving-degree shortcut after validation failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from evaluate_2wiki_graph_retrieval import prepare_split
from topocf_rag.hotpot import sha256_file
from topocf_rag.twowiki import atomic_json_dump
from topocf_rag.twowiki_degree_diagnostic import (
    build_degree_diagnostic_report,
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
        "--validation-report",
        type=Path,
        default=Path(
            "reports/graph_retrieval/2wiki_graph_retrieval_validation.json"
        ),
    )
    parser.add_argument(
        "--hotpot-validation-report",
        type=Path,
        default=Path(
            "reports/graph_retrieval/hotpot_kill_test_validation.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/graph_retrieval/2wiki_degree_shortcut_diagnostic.json"
        ),
    )
    return parser.parse_args()


def _load_report(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8")), sha256_file(path)


def main() -> int:
    args = parse_args()
    base_report, base_sha256 = _load_report(args.base_report)
    validation_report, validation_sha256 = _load_report(
        args.validation_report
    )
    hotpot_report, hotpot_sha256 = _load_report(
        args.hotpot_validation_report
    )
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
    report = build_degree_diagnostic_report(
        train_questions,
        dev_questions,
        base_report=base_report,
        base_report_sha256=base_sha256,
        validation_report=validation_report,
        validation_report_sha256=validation_sha256,
        hotpot_validation_report=hotpot_report,
        hotpot_validation_report_sha256=hotpot_sha256,
        provenance={
            "base_report_path": str(args.base_report.resolve()),
            "validation_report_path": str(args.validation_report.resolve()),
            "hotpot_validation_report_path": str(
                args.hotpot_validation_report.resolve()
            ),
            "train": train_provenance,
            "dev": dev_provenance,
        },
    )
    atomic_json_dump(report, args.output)
    complementarity = report["dev"][
        "tuned_graph_degree_complementarity"
    ]
    centrality = report["dev"]["centrality_shortcut"]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "cross_dataset_degree_control": report[
                    "cross_dataset_degree_control"
                ],
                "dev_comparisons": report["dev"]["comparisons"],
                "complementarity_summary": {
                    "macro_by_extra_budget": complementarity[
                        "macro_by_extra_budget"
                    ],
                    "macro_mean_oracle_headroom": complementarity[
                        "macro_mean_oracle_headroom"
                    ],
                    "macro_mean_left_only_rate": complementarity[
                        "macro_mean_left_only_rate"
                    ],
                    "macro_mean_right_only_rate": complementarity[
                        "macro_mean_right_only_rate"
                    ],
                },
                "dev_centrality_auc": {
                    direction: centrality[direction]["overall"][
                        "gold_vs_non_gold_degree_pairwise_auc"
                    ]
                    for direction in ("outgoing", "incoming", "undirected")
                },
                "decision": report["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
