#!/usr/bin/env python3
"""Audit untouched official 2Wiki test data before any test scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from topocf_rag.hotpot import sha256_file
from topocf_rag.twowiki import atomic_json_dump, iter_twowiki_records
from topocf_rag.twowiki_test_readiness import (
    EXPECTED_MUSIQUE_STAGE_G1_REPORT_SHA256,
    audit_twowiki_test_records,
    build_twowiki_test_readiness_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("/file_system/datasets/2wikimultihopqa_official_20210407")
    parser.add_argument(
        "--test-source",
        type=Path,
        default=root / "test.json",
    )
    parser.add_argument(
        "--selected-train-ids",
        type=Path,
        default=Path(
            "data/splits/2wiki_train_clean_balanced_1000_ids.json"
        ),
    )
    parser.add_argument(
        "--selected-dev-ids",
        type=Path,
        default=Path("data/splits/2wiki_dev_clean_balanced_500_ids.json"),
    )
    parser.add_argument(
        "--musique-g1-report",
        type=Path,
        default=Path("reports/musique/unused_holdout_readiness_audit.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/graph_retrieval/2wiki_test_readiness_audit.json"
        ),
    )
    return parser.parse_args()


def _load_json_object(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def main() -> int:
    args = parse_args()
    for path in (
        args.test_source,
        args.selected_train_ids,
        args.selected_dev_ids,
        args.musique_g1_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    musique_g1_sha256 = sha256_file(args.musique_g1_report)
    if musique_g1_sha256 != EXPECTED_MUSIQUE_STAGE_G1_REPORT_SHA256:
        raise ValueError("MuSiQue Stage G1 report SHA256 changed")

    audit, test_ids, eligible_ids_by_type = audit_twowiki_test_records(
        iter_twowiki_records(args.test_source)
    )
    report = build_twowiki_test_readiness_report(
        source_path=str(args.test_source.resolve()),
        source_sha256=sha256_file(args.test_source),
        source_size_bytes=args.test_source.stat().st_size,
        audit=audit,
        test_ids=test_ids,
        eligible_ids_by_type=eligible_ids_by_type,
        selected_train_manifest=_load_json_object(args.selected_train_ids),
        selected_train_manifest_path=str(args.selected_train_ids.resolve()),
        selected_train_manifest_sha256=sha256_file(
            args.selected_train_ids
        ),
        selected_dev_manifest=_load_json_object(args.selected_dev_ids),
        selected_dev_manifest_path=str(args.selected_dev_ids.resolve()),
        selected_dev_manifest_sha256=sha256_file(args.selected_dev_ids),
        musique_g1_report=_load_json_object(args.musique_g1_report),
        musique_g1_report_path=str(args.musique_g1_report.resolve()),
        musique_g1_report_sha256=musique_g1_sha256,
    )
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "musique_g1": report["musique_g1"],
                "protocol": report["protocol"],
                "source": report["source"],
                "split_separation": report["split_separation"],
                "eligible_test_pool": report["eligible_test_pool"],
                "frozen_test_hypotheses": report[
                    "frozen_test_hypotheses"
                ],
                "readiness_gate": report["readiness_gate"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["readiness_gate"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
