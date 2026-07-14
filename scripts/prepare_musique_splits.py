#!/usr/bin/env python3
"""Materialize frozen MuSiQue train/dev ID manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from topocf_rag.hotpot import sha256_file
from topocf_rag.musique_splits import (
    SPLIT_SEED,
    TRAIN_PER_CELL,
    build_split_bundle,
    collect_candidate_file,
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
        "--duplicate-audit",
        type=Path,
        default=Path("reports/musique/duplicate_title_audit.json"),
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path(
            "data/splits/musique_train_hop_collision_balanced_1800_ids.json"
        ),
    )
    parser.add_argument(
        "--dev-output",
        type=Path,
        default=Path("data/splits/musique_dev_exact20_all_ids.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/musique/split_audit.json"),
    )
    parser.add_argument("--train-per-cell", type=int, default=TRAIN_PER_CELL)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.duplicate_audit.is_file():
        raise FileNotFoundError(args.duplicate_audit)
    duplicate_audit = json.loads(
        args.duplicate_audit.read_text(encoding="utf-8")
    )
    train_pool = collect_candidate_file(args.train_source)
    dev_pool = collect_candidate_file(args.dev_source)
    train_manifest, dev_manifest, report = build_split_bundle(
        train_pool,
        dev_pool,
        train_source_path=str(args.train_source.resolve()),
        train_source_sha256=sha256_file(args.train_source),
        dev_source_path=str(args.dev_source.resolve()),
        dev_source_sha256=sha256_file(args.dev_source),
        duplicate_audit=duplicate_audit,
        duplicate_audit_sha256=sha256_file(args.duplicate_audit),
        train_per_cell=args.train_per_cell,
        seed=args.seed,
    )
    atomic_json_dump(train_manifest, args.train_output)
    atomic_json_dump(dev_manifest, args.dev_output)
    report["artifacts"] = {
        "train_manifest_path": str(args.train_output.resolve()),
        "train_manifest_sha256": sha256_file(args.train_output),
        "dev_manifest_path": str(args.dev_output.resolve()),
        "dev_manifest_sha256": sha256_file(args.dev_output),
    }
    atomic_json_dump(report, args.report)
    print(
        json.dumps(
            {
                "train_output": str(args.train_output.resolve()),
                "dev_output": str(args.dev_output.resolve()),
                "report": str(args.report.resolve()),
                "artifacts": report["artifacts"],
                "train": report["train"],
                "dev": report["dev"],
                "gate": report["gate"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
