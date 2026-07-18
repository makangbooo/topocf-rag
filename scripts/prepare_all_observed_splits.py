#!/usr/bin/env python3
"""Materialize frozen question-level splits for TopoCF all-observed v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import ijson

from topocf_rag.all_observed_split import (
    INNER_VALIDATION_QUESTION_COUNT,
    SPLIT_SEED,
    build_dev_manifest,
    build_train_manifest,
    scan_all_observed_questions,
)
from topocf_rag.evaluation import atomic_json_dump, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/certificate_v1/topocf_all_observed_split_v2.json"),
    )
    return parser.parse_args()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _stream(path: Path):
    with path.open("rb") as stream:
        yield from ijson.items(stream, "item")


def _public_role(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "ids"}


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    config_path = args.config.resolve()
    config = _require_mapping(
        json.loads(config_path.read_text(encoding="utf-8")), "config"
    )
    if config.get("schema_version") != 1:
        raise ValueError("unsupported split config schema")
    design = _require_mapping(config.get("train_inner_split"), "train_inner_split")
    if design.get("seed") != SPLIT_SEED:
        raise ValueError("split seed changed")
    if (
        design.get("validation_question_count")
        != INNER_VALIDATION_QUESTION_COUNT
        or design.get("fit_question_count") != 1829
    ):
        raise ValueError("train role sizes changed")

    inputs = _require_mapping(config.get("inputs"), "inputs")
    audit = _require_mapping(inputs.get("population_audit"), "population_audit")
    audit_path = root / str(audit.get("path"))
    audit_sha256 = sha256_file(audit_path)
    if audit_sha256 != audit.get("sha256"):
        raise ValueError("population audit hash changed")
    audit_payload = _require_mapping(
        json.loads(audit_path.read_text(encoding="utf-8")), "audit payload"
    )
    if audit_payload.get("gate", {}).get("status") != "authorize_split_design_only":
        raise ValueError("population audit did not authorize split design")

    artifacts = _require_mapping(config.get("artifacts"), "artifacts")
    for name, value in artifacts.items():
        artifact = _require_mapping(value, f"artifact {name}")
        path = root / str(artifact.get("path"))
        if sha256_file(path) != artifact.get("sha256"):
            raise ValueError(f"artifact {name} hash changed")

    sources = _require_mapping(inputs.get("sources"), "sources")
    scans = {}
    questions = {}
    for split in ("train", "dev_distractor"):
        source = _require_mapping(sources.get(split), f"source {split}")
        source_path = Path(str(source.get("path"))).resolve()
        source_sha256 = sha256_file(source_path)
        if source_sha256 != source.get("sha256"):
            raise ValueError(f"{split} source hash changed")
        questions[split], scans[split] = scan_all_observed_questions(
            _stream(source_path)
        )
        audited = _require_mapping(
            audit_payload.get("splits", {}).get(split), f"audit split {split}"
        )
        for key, value in scans[split].items():
            if audited.get(key) != value:
                raise ValueError(f"{split} scan disagrees with population audit: {key}")

    outputs = _require_mapping(config.get("outputs"), "outputs")
    output_paths = {
        name: root / str(value)
        for name, value in outputs.items()
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")

    train_source = _require_mapping(sources["train"], "train source")
    dev_source = _require_mapping(sources["dev_distractor"], "dev source")
    train_manifest = build_train_manifest(
        questions["train"],
        source_path=str(train_source["path"]),
        source_sha256=str(train_source["sha256"]),
        population_audit_path=str(audit["path"]),
        population_audit_sha256=audit_sha256,
    )
    dev_manifest = build_dev_manifest(
        questions["dev_distractor"],
        source_path=str(dev_source["path"]),
        source_sha256=str(dev_source["sha256"]),
        population_audit_path=str(audit["path"]),
        population_audit_sha256=audit_sha256,
    )
    if set(train_manifest["fit"]["ids"]).intersection(
        dev_manifest["evaluation"]["ids"]
    ) or set(train_manifest["validation"]["ids"]).intersection(
        dev_manifest["evaluation"]["ids"]
    ):
        raise ValueError("official train and dev IDs overlap")

    atomic_json_dump(train_manifest, output_paths["train_manifest"])
    atomic_json_dump(dev_manifest, output_paths["dev_manifest"])
    train_sha256 = sha256_file(output_paths["train_manifest"])
    dev_sha256 = sha256_file(output_paths["dev_manifest"])
    report = {
        "schema_version": 1,
        "task": "topocf_all_observed_split_design",
        "content_contract": (
            "aggregate counts, protocol, and hashes only; no IDs or dataset text"
        ),
        "inputs": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "population_audit_path": str(audit_path),
            "population_audit_sha256": audit_sha256,
            "source_sha256": {
                "train": train_source["sha256"],
                "dev_distractor": dev_source["sha256"],
            },
        },
        "train": {
            "population": train_manifest["population"],
            "fit": _public_role(train_manifest["fit"]),
            "validation": _public_role(train_manifest["validation"]),
            "protocol": train_manifest["protocol"],
        },
        "dev_distractor": {
            "evaluation": _public_role(dev_manifest["evaluation"]),
            "protocol": dev_manifest["protocol"],
        },
        "outputs": {
            "train_manifest": {
                "path": str(output_paths["train_manifest"]),
                "sha256": train_sha256,
            },
            "dev_manifest": {
                "path": str(output_paths["dev_manifest"]),
                "sha256": dev_sha256,
            },
        },
        "integrity": {
            "fit_validation_overlap_count": 0,
            "train_dev_overlap_count": 0,
            "official_dev_used_for_selection": False,
            "all_eligible_train_questions_partitioned": True,
            "all_eligible_dev_questions_reserved": True,
        },
        "authorization": {
            "status": "splits_frozen_training_still_not_authorized",
            "next_required_gate": "materialize pairs and run content-free baselines",
        },
    }
    atomic_json_dump(report, output_paths["report"])
    print(
        json.dumps(
            {
                "train_fit_questions": train_manifest["fit"]["question_count"],
                "train_fit_pairs": train_manifest["fit"]["pair_count"],
                "train_validation_questions": train_manifest["validation"][
                    "question_count"
                ],
                "train_validation_pairs": train_manifest["validation"][
                    "pair_count"
                ],
                "dev_questions": dev_manifest["evaluation"]["question_count"],
                "dev_pairs": dev_manifest["evaluation"]["pair_count"],
                "train_manifest_sha256": train_sha256,
                "dev_manifest_sha256": dev_sha256,
                "report": str(output_paths["report"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
