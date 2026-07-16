#!/usr/bin/env python3
"""Audit unused labeled MuSiQue train candidates before holdout scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from topocf_rag.hotpot import sha256_file
from topocf_rag.musique_holdout_readiness import (
    EXPECTED_STAGE_C_REPORT_SHA256,
    EXPECTED_STAGE_G0_REPORT_SHA256,
    build_musique_holdout_readiness_report,
)
from topocf_rag.musique_splits import collect_candidate_file
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
        "--selected-train-ids",
        type=Path,
        default=Path(
            "data/splits/musique_train_hop_collision_balanced_1800_ids.json"
        ),
    )
    parser.add_argument(
        "--selected-dev-ids",
        type=Path,
        default=Path("data/splits/musique_dev_exact20_all_ids.json"),
    )
    parser.add_argument(
        "--stage-c-report",
        type=Path,
        default=Path("reports/musique/split_audit.json"),
    )
    parser.add_argument(
        "--stage-g0-report",
        type=Path,
        default=Path("reports/musique/test_readiness_audit.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/musique/unused_holdout_readiness_audit.json"),
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
        args.train_source,
        args.selected_train_ids,
        args.selected_dev_ids,
        args.stage_c_report,
        args.stage_g0_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    stage_c_sha256 = sha256_file(args.stage_c_report)
    stage_g0_sha256 = sha256_file(args.stage_g0_report)
    if stage_c_sha256 != EXPECTED_STAGE_C_REPORT_SHA256:
        raise ValueError("Stage C report SHA256 changed")
    if stage_g0_sha256 != EXPECTED_STAGE_G0_REPORT_SHA256:
        raise ValueError("Stage G0 report SHA256 changed")

    report = build_musique_holdout_readiness_report(
        train_candidate_pool=collect_candidate_file(args.train_source),
        train_source_path=str(args.train_source.resolve()),
        train_source_sha256=sha256_file(args.train_source),
        stage_c_report=_load_json_object(args.stage_c_report),
        stage_c_report_path=str(args.stage_c_report.resolve()),
        stage_c_report_sha256=stage_c_sha256,
        stage_g0_report=_load_json_object(args.stage_g0_report),
        stage_g0_report_path=str(args.stage_g0_report.resolve()),
        stage_g0_report_sha256=stage_g0_sha256,
        selected_train_manifest=_load_json_object(args.selected_train_ids),
        selected_train_manifest_path=str(args.selected_train_ids.resolve()),
        selected_train_manifest_sha256=sha256_file(args.selected_train_ids),
        selected_dev_manifest=_load_json_object(args.selected_dev_ids),
        selected_dev_manifest_path=str(args.selected_dev_ids.resolve()),
        selected_dev_manifest_sha256=sha256_file(args.selected_dev_ids),
    )
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "stage_c": report["stage_c"],
                "stage_g0": report["stage_g0"],
                "protocol": report["protocol"],
                "source": report["source"],
                "selected_splits": report["selected_splits"],
                "candidate_pool": report["candidate_pool"],
                "untouched_remainder": report["untouched_remainder"],
                "frozen_holdout_hypotheses": report[
                    "frozen_holdout_hypotheses"
                ],
                "readiness_gate": report["readiness_gate"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["readiness_gate"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
