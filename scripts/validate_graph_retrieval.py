#!/usr/bin/env python3
"""Validate HotpotQA graph retrieval against matched and permutation controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from evaluate_graph_retrieval import atomic_json_dump, prepare_split, sha256_file
from topocf_rag.graph_retrieval import DEFAULT_TOP_KS
from topocf_rag.graph_retrieval_validation import (
    PERMUTATION_REPETITIONS,
    PERMUTATION_SEED,
    GraphRetrievalValidationError,
    build_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-report",
        type=Path,
        default=Path("reports/graph_retrieval/hotpot_kill_test.json"),
    )
    parser.add_argument(
        "--train-source",
        type=Path,
        default=Path("/file_system/datasets/hotpotqa/hotpot_train_v1.1.json"),
    )
    parser.add_argument(
        "--train-ids",
        type=Path,
        default=Path("data/splits/hotpot_train_bridge_1000_ids.json"),
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=Path("reports/title_graph/cache/train_bge_retrieval.json"),
    )
    parser.add_argument(
        "--dev-source",
        type=Path,
        default=Path("/file_system/datasets/hotpotqa/hotpot_dev_distractor_v1.json"),
    )
    parser.add_argument(
        "--dev-ids",
        type=Path,
        default=Path("data/splits/hotpot_dev_bridge_500_ids.json"),
    )
    parser.add_argument(
        "--dev-cache",
        type=Path,
        default=Path("reports/title_graph/cache/dev_distractor_bge_retrieval.json"),
    )
    parser.add_argument(
        "--permutation-repetitions",
        type=int,
        default=PERMUTATION_REPETITIONS,
    )
    parser.add_argument("--permutation-seed", type=int, default=PERMUTATION_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/graph_retrieval/hotpot_kill_test_validation.json"),
    )
    parser.add_argument("--no-fail-on-gate", action="store_true")
    return parser.parse_args()


def _load_base_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise GraphRetrievalValidationError("base report has invalid schema")
    if payload.get("task") != "label_free_graph_aware_retrieval_kill_test":
        raise GraphRetrievalValidationError("base report has an unexpected task")
    return payload


def main() -> int:
    args = parse_args()
    base_report = _load_base_report(args.base_report)
    top_ks = tuple(base_report.get("protocol", {}).get("top_ks", ()))
    if top_ks != DEFAULT_TOP_KS:
        raise GraphRetrievalValidationError(
            "base report top-k protocol does not match validation protocol"
        )

    train_questions, train_provenance = prepare_split(
        args.train_source, args.train_ids, args.train_cache
    )
    dev_questions, dev_provenance = prepare_split(
        args.dev_source, args.dev_ids, args.dev_cache
    )
    current_provenance: Mapping[str, Any] = {
        "train": train_provenance,
        "dev_distractor": dev_provenance,
    }
    if base_report.get("provenance") != current_provenance:
        raise GraphRetrievalValidationError(
            "current source, IDs, or retrieval caches do not match the base report"
        )

    report = build_validation_report(
        train_questions,
        dev_questions,
        base_report=base_report,
        base_report_sha256=sha256_file(args.base_report),
        provenance=current_provenance,
        top_ks=top_ks,
        permutation_repetitions=args.permutation_repetitions,
        permutation_seed=args.permutation_seed,
    )
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "dev_matched_non_graph_baseline": report["dev"][
                    "matched_non_graph_baseline"
                ],
                "dev_degree_only_control": report["dev"][
                    "degree_only_control"
                ],
                "dev_paired_transitions": report["dev"][
                    "paired_transitions_vs_matched_baseline"
                ],
                "dev_permutation_null": report["dev"]["permutation_null"],
                "direction_ablation": report["direction_ablation"],
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
