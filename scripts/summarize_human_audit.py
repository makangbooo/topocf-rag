#!/usr/bin/env python3
"""Write a content-free aggregate report for private human annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topocf_rag.human_audit import (
    load_human_audit_bundle,
    write_public_human_audit_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prelabels", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = load_human_audit_bundle(args.input, args.prelabels)
    report = write_public_human_audit_report(
        bundle, args.annotations, args.report
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
