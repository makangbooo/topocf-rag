#!/usr/bin/env python3
"""Freeze the question-disjoint TopoCF train-only inner split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from topocf_rag.method_data import (
    INNER_SPLIT_SEED,
    INNER_VALIDATION_QUESTION_COUNT,
    build_inner_split_manifest,
    source_pair_manifest_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=Path("data/phase1/hotpot_train_phase1_pairs.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/topocf_t3_train_inner_v1.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/phase1/method_inner_split.json"),
    )
    parser.add_argument(
        "--validation-question-count",
        type=int,
        default=INNER_VALIDATION_QUESTION_COUNT,
    )
    parser.add_argument("--seed", type=int, default=INNER_SPLIT_SEED)
    return parser.parse_args()


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    source_sha256 = source_pair_manifest_sha256(args.pair_manifest)
    source = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
    manifest = build_inner_split_manifest(
        source,
        pair_manifest_sha256=source_sha256,
        validation_question_count=args.validation_question_count,
        seed=args.seed,
    )
    atomic_json_dump(manifest, args.output)
    output_sha256 = source_pair_manifest_sha256(args.output)
    report = {
        "schema_version": 1,
        "task": "topocf_bind_repair_inner_split_materialization",
        "content_contract": (
            "aggregate counts, protocol, and hashes only; "
            "no IDs or dataset text"
        ),
        "inputs": {
            "pair_manifest_path": str(args.pair_manifest),
            "pair_manifest_sha256": source_sha256,
        },
        "output": {
            "path": str(args.output),
            "sha256": output_sha256,
        },
        "protocol": manifest["protocol"],
        "population": manifest["population"],
        "fit": {
            key: value for key, value in manifest["fit"].items() if key != "ids"
        },
        "validation": {
            key: value
            for key, value in manifest["validation"].items()
            if key != "ids"
        },
        "integrity": manifest["integrity"],
    }
    atomic_json_dump(report, args.report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": output_sha256,
                "report": str(args.report.resolve()),
                "fit_question_count": manifest["fit"]["question_count"],
                "fit_pair_count": manifest["fit"]["pair_count"],
                "validation_question_count": manifest["validation"][
                    "question_count"
                ],
                "validation_pair_count": manifest["validation"]["pair_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
