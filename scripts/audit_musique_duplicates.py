#!/usr/bin/env python3
"""Audit duplicate MuSiQue titles before freezing a paragraph-node graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from topocf_rag.hotpot import sha256_file
from topocf_rag.musique_duplicates import (
    audit_duplicate_titles,
    build_duplicate_audit_report,
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
        "--schema-report",
        type=Path,
        default=Path("reports/musique/schema_audit.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/musique/duplicate_title_audit.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.schema_report.is_file():
        raise FileNotFoundError(args.schema_report)
    schema_report = json.loads(
        args.schema_report.read_text(encoding="utf-8")
    )
    report = build_duplicate_audit_report(
        audit_duplicate_titles(args.train_source),
        audit_duplicate_titles(args.dev_source),
        schema_report=schema_report,
        schema_report_sha256=sha256_file(args.schema_report),
    )
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "protocol": report["protocol"],
                "consistency_gate": report["consistency_gate"],
                "splits": {
                    split: {
                        "record_count": value["record_count"],
                        "record_status_histogram": value[
                            "record_status_histogram"
                        ],
                        "record_status_by_hop": value[
                            "record_status_by_hop"
                        ],
                        "duplicate_group_aggregates": value[
                            "duplicate_group_aggregates"
                        ],
                        "policy_eligible_count": value[
                            "policy_eligible_count"
                        ],
                        "policy_eligible_hop_histogram": value[
                            "policy_eligible_hop_histogram"
                        ],
                    }
                    for split, value in report["splits"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0 if report["consistency_gate"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
