#!/usr/bin/env python3
"""Run the aggregate-only TopoCF method readiness gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from topocf_rag.method_readiness import build_method_readiness_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/certificate_v1/topocf_bind_repair_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/phase1/method_readiness.json"),
    )
    parser.add_argument(
        "--no-fail-on-development-gate",
        action="store_true",
        help="write a failed report but return exit code zero",
    )
    return parser.parse_args()


def atomic_json_dump(payload: dict, path: Path) -> None:
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
    report = build_method_readiness_report(args.config)
    atomic_json_dump(report, args.output)
    summary = {
        "output": str(args.output.resolve()),
        "development_training": report["gates"]["development_training"],
        "confirmatory_evaluation": report["gates"]["confirmatory_evaluation"],
        "paper_claim": report["gates"]["paper_claim"],
    }
    print(json.dumps(summary, sort_keys=True))
    if (
        not report["gates"]["development_training"]["passed"]
        and not args.no_fail_on_development_gate
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
