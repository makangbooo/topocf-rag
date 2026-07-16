#!/usr/bin/env python3
"""Run post-primary robustness controls for MuSiQue graph retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from evaluate_musique_graph_retrieval import (
    prepare_split,
    validate_stage_d_report,
)
from topocf_rag.hotpot import sha256_file
from topocf_rag.musique_graph_validation import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    CONFIG_BOOTSTRAP_REPETITIONS,
    CONFIG_BOOTSTRAP_SEED,
    EXPECTED_STAGE_E_REPORT_SHA256,
    build_musique_validation_report,
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
        "--train-cache",
        type=Path,
        default=Path(
            "reports/musique/cache/a100/musique_train_bge_retrieval.json"
        ),
    )
    parser.add_argument(
        "--dev-cache",
        type=Path,
        default=Path(
            "reports/musique/cache/a100/musique_dev_bge_retrieval.json"
        ),
    )
    parser.add_argument(
        "--cache-report",
        type=Path,
        default=Path("reports/musique/bge_cache_report.json"),
    )
    parser.add_argument(
        "--base-report",
        type=Path,
        default=Path("reports/musique/graph_retrieval.json"),
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=BOOTSTRAP_SEED
    )
    parser.add_argument(
        "--configuration-bootstrap-repetitions",
        type=int,
        default=CONFIG_BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument(
        "--configuration-bootstrap-seed",
        type=int,
        default=CONFIG_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/musique/graph_retrieval_validation.json"),
    )
    parser.add_argument("--no-fail-on-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    if args.configuration_bootstrap_repetitions < 1:
        raise ValueError(
            "configuration bootstrap repetitions must be positive"
        )
    if not args.base_report.is_file():
        raise FileNotFoundError(args.base_report)
    base_report_sha256 = sha256_file(args.base_report)
    if base_report_sha256 != EXPECTED_STAGE_E_REPORT_SHA256:
        raise ValueError("Stage E report SHA256 changed")
    base_report = json.loads(args.base_report.read_text(encoding="utf-8"))
    cache_report = validate_stage_d_report(
        args.cache_report,
        train_cache=args.train_cache,
        dev_cache=args.dev_cache,
    )
    train_questions, train_provenance = prepare_split(
        args.train_source,
        args.train_ids,
        args.train_cache,
        split="train",
        cache_report=cache_report,
    )
    dev_questions, dev_provenance = prepare_split(
        args.dev_source,
        args.dev_ids,
        args.dev_cache,
        split="dev",
        cache_report=cache_report,
    )
    report = build_musique_validation_report(
        train_questions,
        dev_questions,
        base_report=base_report,
        base_report_sha256=base_report_sha256,
        provenance={
            "base_report_path": str(args.base_report.resolve()),
            "stage_d_report_path": str(args.cache_report.resolve()),
            "stage_d_report_sha256": sha256_file(args.cache_report),
            "train": train_provenance,
            "dev": dev_provenance,
        },
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
        config_bootstrap_repetitions=(
            args.configuration_bootstrap_repetitions
        ),
        config_bootstrap_seed=args.configuration_bootstrap_seed,
    )
    atomic_json_dump(report, args.output)
    bootstraps = report["dev"]["paired_stratified_bootstrap"]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "base_report_sha256": base_report_sha256,
                "configuration_stability": report["train"][
                    "configuration_bootstrap_stability"
                ],
                "leave_one_cell_out_macro": report["train"][
                    "leave_one_cell_out"
                ]["macro_across_held_out_cells"],
                "bootstrap_overall": {
                    name: value["overall"]["mean_across_extra_budgets"]
                    for name, value in bootstraps.items()
                },
                "heterogeneity": report["dev"]["heterogeneity_summary"],
                "gate": report["gate"],
                "claim_boundary": report["claim_boundary"],
            },
            sort_keys=True,
        )
    )
    if report["gate"]["passed"] or args.no_fail_on_gate:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
