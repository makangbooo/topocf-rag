#!/usr/bin/env python3
"""Census locally grounded T3 rewires over full official HotpotQA splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping

import ijson

from topocf_rag.all_observed_population import (
    audit_all_observed_population,
    population_gate,
)
from topocf_rag.evaluation import atomic_json_dump, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/certificate_v1/topocf_all_observed_v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/phase1/all_observed_population_audit.json"),
    )
    parser.add_argument("--no-fail-on-gate", action="store_true")
    return parser.parse_args()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _load_config(path: Path, repository_root: Path) -> Mapping[str, Any]:
    config = _require_mapping(json.loads(path.read_text(encoding="utf-8")), "config")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported all-observed config schema")
    artifacts = _require_mapping(config.get("artifacts"), "artifacts")
    for name, value in artifacts.items():
        artifact = _require_mapping(value, f"artifact {name}")
        relative = artifact.get("path")
        expected = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"artifact {name} is invalid")
        if sha256_file(repository_root / relative) != expected:
            raise ValueError(f"artifact {name} hash changed")
    return config


def _stream(path: Path):
    with path.open("rb") as stream:
        yield from ijson.items(stream, "item")


def main() -> int:
    args = parse_args()
    repository_root = Path.cwd().resolve()
    config_path = args.config.resolve()
    config = _load_config(config_path, repository_root)
    audit = _require_mapping(config.get("population_audit"), "population_audit")
    sources = _require_mapping(audit.get("sources"), "population_audit.sources")
    thresholds = _require_mapping(
        audit.get("thresholds"), "population_audit.thresholds"
    )

    started = time.perf_counter()
    summaries = {}
    source_hashes = {}
    for split in ("train", "dev_distractor"):
        source = _require_mapping(sources.get(split), f"source {split}")
        path = Path(str(source.get("path"))).resolve()
        expected_sha256 = source.get("sha256")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"{split} source hash changed")
        summaries[split] = audit_all_observed_population(_stream(path))
        source_hashes[split] = actual_sha256

    gate = population_gate(
        summaries,
        minimum_train_question_count=int(
            thresholds["minimum_train_question_count"]
        ),
        minimum_dev_question_count=int(
            thresholds["minimum_dev_question_count"]
        ),
    )
    report = {
        "schema_version": 1,
        "task": "topocf_all_observed_full_population_audit",
        "content_contract": audit["content_contract"],
        "protocol": {
            "candidate": "all_observed_rewire/t3_all_observed",
            "question_filter": audit["question_filter"],
            "context_pool": audit["context_pool"],
            "retrieval_model_used": False,
            "manual_annotation_used": False,
            "source_records_streamed": True,
        },
        "splits": summaries,
        "gate": gate,
        "artifacts": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "source_sha256": source_hashes,
        },
        "runtime": {"elapsed_seconds": time.perf_counter() - started},
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "splits": {
                    split: {
                        "record_count": summary["record_count"],
                        "bridge_question_count": summary["bridge_question_count"],
                        "all_observed_question_count": summary[
                            "all_observed_question_count"
                        ],
                        "all_observed_pair_count": summary[
                            "all_observed_pair_count"
                        ],
                    }
                    for split, summary in summaries.items()
                },
                "gate": gate,
                "elapsed_seconds": report["runtime"]["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if gate["passed"] or args.no_fail_on_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
