#!/usr/bin/env python3
"""Audit untouched MuSiQue answerable test before any retrieval scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from topocf_rag.hotpot import sha256_file
from topocf_rag.musique import audit_musique_file
from topocf_rag.musique_duplicates import audit_duplicate_titles
from topocf_rag.musique_splits import collect_candidate_file
from topocf_rag.musique_test_readiness import (
    EXPECTED_STAGE_F_REPORT_SHA256,
    build_musique_test_readiness_report,
)
from topocf_rag.twowiki import atomic_json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("/file_system/datasets/musique_official_v1.0")
    parser.add_argument(
        "--test-source",
        type=Path,
        default=root / "musique_ans_v1.0_test.jsonl",
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
        "--stage-f-report",
        type=Path,
        default=Path("reports/musique/graph_retrieval_validation.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/musique/test_readiness_audit.json"),
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
        args.train_ids,
        args.dev_ids,
        args.stage_f_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    stage_f_sha256 = sha256_file(args.stage_f_report)
    if stage_f_sha256 != EXPECTED_STAGE_F_REPORT_SHA256:
        raise ValueError("Stage F report SHA256 changed")

    report = build_musique_test_readiness_report(
        schema_audit=audit_musique_file(args.test_source),
        duplicate_audit=audit_duplicate_titles(args.test_source),
        candidate_pool=collect_candidate_file(args.test_source),
        stage_f_report=_load_json_object(args.stage_f_report),
        stage_f_report_sha256=stage_f_sha256,
        train_manifest=_load_json_object(args.train_ids),
        train_manifest_path=str(args.train_ids.resolve()),
        train_manifest_sha256=sha256_file(args.train_ids),
        dev_manifest=_load_json_object(args.dev_ids),
        dev_manifest_path=str(args.dev_ids.resolve()),
        dev_manifest_sha256=sha256_file(args.dev_ids),
    )
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "stage_f": report["stage_f"],
                "source": {
                    "sha256": report["source"]["sha256"],
                    "record_count": report["source"]["record_count"],
                    "json_error_count": report["source"]["json_error_count"],
                    "duplicate_id_count": report["source"][
                        "duplicate_id_count"
                    ],
                    "histograms": report["source"]["histograms"],
                },
                "duplicate_title_audit": report["duplicate_title_audit"],
                "candidate_pool": report["candidate_pool"],
                "split_separation": report["split_separation"],
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
