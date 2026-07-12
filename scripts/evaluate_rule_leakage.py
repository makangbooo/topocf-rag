#!/usr/bin/env python3
"""Evaluate content-free structural leakage rules on Phase 1 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from topocf_rag.evaluation import atomic_json_dump
from topocf_rag.rule_baselines import (
    build_rule_leakage_audit,
    parse_rule_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic-train",
        type=Path,
        default=Path("data/phase1/hotpot_train_phase1_pairs.json"),
    )
    parser.add_argument(
        "--synthetic-dev",
        type=Path,
        default=Path("data/phase1/hotpot_dev_phase1_pairs.json"),
    )
    parser.add_argument(
        "--natural-train",
        type=Path,
        default=Path("data/phase1/hotpot_train_natural_retrieval_pairs.json"),
    )
    parser.add_argument(
        "--natural-dev",
        type=Path,
        default=Path("data/phase1/hotpot_dev_natural_retrieval_pairs.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/certificate_v1/rule_leakage.json"),
    )
    parser.add_argument("--gate-threshold", type=float, default=0.65)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    return parse_rule_manifest(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    args = parse_args()
    paths = {
        "synthetic_train": args.synthetic_train,
        "synthetic_dev": args.synthetic_dev,
        "natural_train": args.natural_train,
        "natural_dev": args.natural_dev,
    }
    loaded = {name: load(path) for name, path in paths.items()}
    expected_splits = {
        "synthetic_train": "train",
        "synthetic_dev": "dev_distractor",
        "natural_train": "train",
        "natural_dev": "dev_distractor",
    }
    for name, (official_split, _pairs) in loaded.items():
        if official_split != expected_splits[name]:
            raise ValueError(
                f"{name} official split is {official_split!r}, "
                f"expected {expected_splits[name]!r}"
            )

    report = build_rule_leakage_audit(
        synthetic_train=loaded["synthetic_train"][1],
        synthetic_dev=loaded["synthetic_dev"][1],
        natural_train=loaded["natural_train"][1],
        natural_dev=loaded["natural_dev"][1],
        gate_threshold=args.gate_threshold,
    )
    report["input_hashes"] = {
        name: sha256_file(path) for name, path in paths.items()
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "synthetic_dev_gate": report["synthetic_dev_gate"],
                "synthetic_counts": {
                    split: report["synthetic"][split]["counts"]
                    for split in ("train", "dev_distractor")
                },
                "natural_counts": {
                    split: report["natural"][split]["counts"]
                    for split in ("train", "dev_distractor")
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
