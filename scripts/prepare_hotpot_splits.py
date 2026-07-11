#!/usr/bin/env python3
"""Freeze the Phase-0 HotpotQA bridge diagnostic ID sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from topocf_rag.hotpot import prepare_hotpot_id_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream HotpotQA and freeze deterministic bridge-only ID splits."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/file_system/datasets/hotpotqa")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "data" / "splits"
    )
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--dev-size", type=int, default=500)
    return parser.parse_args()


def _public_summary(
    manifest: dict[str, object], output_path: Path, output_sha256: str
) -> dict[str, object]:
    source = manifest["source"]
    selection = manifest["selection"]
    assert isinstance(source, dict)
    assert isinstance(selection, dict)
    return {
        "official_split": manifest["official_split"],
        "source_path": source["path"],
        "source_sha256": source["sha256"],
        "record_count": source["record_count"],
        "fields": source["fields"],
        "type_counts": source["type_counts"],
        "bridge_count_by_level": source["bridge_count_by_level"],
        "selected_by_level": selection["selected_by_level"],
        "output_path": str(output_path.resolve()),
        "output_sha256": output_sha256,
        "selected_count": selection["sample_size"],
    }


def main() -> None:
    args = parse_args()
    train_output = args.output_dir / "hotpot_train_bridge_1000_ids.json"
    dev_output = args.output_dir / "hotpot_dev_bridge_500_ids.json"

    train_manifest, train_output_sha256 = prepare_hotpot_id_split(
        args.data_dir / "hotpot_train_v1.1.json",
        train_output,
        official_split="train",
        sample_size=args.train_size,
        seed=args.seed,
    )
    dev_manifest, dev_output_sha256 = prepare_hotpot_id_split(
        args.data_dir / "hotpot_dev_distractor_v1.json",
        dev_output,
        official_split="dev_distractor",
        sample_size=args.dev_size,
        seed=args.seed,
    )

    overlap = set(train_manifest["ids"]).intersection(dev_manifest["ids"])
    if overlap:
        raise ValueError("official train and dev diagnostic IDs overlap")

    summary = {
        "seed": args.seed,
        "official_splits_mixed": False,
        "train_dev_id_overlap_count": 0,
        "outputs": [
            _public_summary(train_manifest, train_output, train_output_sha256),
            _public_summary(dev_manifest, dev_output, dev_output_sha256),
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

