#!/usr/bin/env python3
"""Audit official MuSiQue answerable train/dev files without sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from topocf_rag.musique import (
    audit_musique_file,
    build_musique_schema_report,
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
        "--output",
        type=Path,
        default=Path("reports/musique/schema_audit.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_musique_schema_report(
        audit_musique_file(args.train_source),
        audit_musique_file(args.dev_source),
    )
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "schema_gate": report["schema_gate"],
                "splits": {
                    split: {
                        "sha256": value["sha256"],
                        "record_count": value["record_count"],
                        "eligible_count": value["eligible_count"],
                        "eligible_rate": value["eligible_rate"],
                        "histograms": value["histograms"],
                        "exclusions": value["exclusions"],
                    }
                    for split, value in report["splits"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0 if report["schema_gate"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
