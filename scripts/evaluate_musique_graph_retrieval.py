#!/usr/bin/env python3
"""Evaluate occurrence-aware MuSiQue graph retrieval and shortcut controls."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from topocf_rag.graph_retrieval import GraphMethodConfig, RetrievalQuestion
from topocf_rag.hotpot import sha256_file
from topocf_rag.musique_graph_retrieval import (
    EXPECTED_STAGE_D_REPORT_SHA256,
    PERMUTATION_REPETITIONS,
    PERMUTATION_SEED,
    build_musique_graph_report,
    prepare_musique_retrieval_question,
)
from topocf_rag.musique_retrieval import (
    cache_matches,
    load_frozen_manifest,
    load_selected_examples,
    validate_cache_records,
)
from topocf_rag.twowiki import atomic_json_dump
from topocf_rag.twowiki_graph_retrieval import (
    FROZEN_HOTPOT_TRANSFER_CONFIG,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("/file_system/datasets/musique_official_v1.0")
    parser.add_argument(
        "--train-source",
        type=Path,
        default=root / "musique_ans_v1.0_train.jsonl",
    )
    parser.add_argument(
        "--dev-source",
        type=Path,
        default=root / "musique_ans_v1.0_dev.jsonl",
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
        "--train-cache",
        type=Path,
        default=Path(
            "reports/musique/cache/a100/musique_train_bge_retrieval.json"
        ),
    )
    parser.add_argument(
        "--dev-cache",
        type=Path,
        default=Path(
            "reports/musique/cache/a100/musique_dev_bge_retrieval.json"
        ),
    )
    parser.add_argument(
        "--cache-report",
        type=Path,
        default=Path("reports/musique/bge_cache_report.json"),
    )
    parser.add_argument(
        "--hotpot-report",
        type=Path,
        default=Path("reports/graph_retrieval/hotpot_kill_test.json"),
    )
    parser.add_argument(
        "--hotpot-validation-report",
        type=Path,
        default=Path(
            "reports/graph_retrieval/hotpot_kill_test_validation.json"
        ),
    )
    parser.add_argument(
        "--permutation-repetitions",
        type=int,
        default=PERMUTATION_REPETITIONS,
    )
    parser.add_argument(
        "--permutation-seed", type=int, default=PERMUTATION_SEED
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/musique/graph_retrieval.json"),
    )
    parser.add_argument("--no-fail-on-gate", action="store_true")
    return parser.parse_args()


def validate_stage_d_report(
    path: Path, *, train_cache: Path, dev_cache: Path
) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_STAGE_D_REPORT_SHA256:
        raise ValueError("Stage D report SHA256 changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("task") != (
        "musique_frozen_paragraph_pool_bge_m3_scoring"
    ):
        raise ValueError("Stage D report task changed")
    stage_c = payload.get("stage_c")
    if not isinstance(stage_c, Mapping) or not stage_c.get("gate_verified"):
        raise ValueError("Stage D did not verify Stage C")
    splits = payload.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("Stage D split reports are missing")
    expected = {
        "train": (train_cache, 1800, 36000),
        "dev": (dev_cache, 2401, 48020),
    }
    for split, (cache_path, question_count, paragraph_count) in expected.items():
        row = splits.get(split)
        if not isinstance(row, Mapping):
            raise ValueError("Stage D split report is missing")
        if row.get("cache_sha256") != sha256_file(cache_path):
            raise ValueError("Stage D cache hash binding failed")
        if row.get("question_count") != question_count:
            raise ValueError("Stage D question count changed")
        if row.get("paragraph_occurrence_count") != paragraph_count:
            raise ValueError("Stage D paragraph count changed")
    return payload


def authorize_hotpot_transfer(
    base_path: Path, validation_path: Path
) -> tuple[GraphMethodConfig, dict[str, Any]]:
    for path in (base_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    base_sha256 = sha256_file(base_path)
    validation_sha256 = sha256_file(validation_path)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not isinstance(base, Mapping) or not isinstance(validation, Mapping):
        raise ValueError("Hotpot reports must be JSON objects")
    if not base.get("gate", {}).get("passed"):
        raise ValueError("Hotpot retrieval feasibility gate did not pass")
    if not validation.get("gate", {}).get("passed"):
        raise ValueError("Hotpot graph-alignment validation gate did not pass")
    validation_base = validation.get("base_report")
    if not isinstance(validation_base, Mapping) or validation_base.get(
        "sha256"
    ) != base_sha256:
        raise ValueError("Hotpot validation does not bind the supplied report")
    selection = base.get("selection")
    config_payload = (
        selection.get("selected_graph_config")
        if isinstance(selection, Mapping)
        else None
    )
    if not isinstance(config_payload, Mapping):
        raise ValueError("Hotpot selected configuration is missing")
    config = GraphMethodConfig(**config_payload)
    if config != FROZEN_HOTPOT_TRANSFER_CONFIG:
        raise ValueError("Hotpot selected configuration changed")
    if validation_base.get("selected_graph_config") != asdict(config):
        raise ValueError("Hotpot validation configuration does not match")
    return config, {
        "base_report_path": str(base_path.resolve()),
        "base_report_sha256": base_sha256,
        "base_gate_passed": True,
        "validation_report_path": str(validation_path.resolve()),
        "validation_report_sha256": validation_sha256,
        "validation_gate_passed": True,
        "selected_config_verified": asdict(config),
    }


def prepare_split(
    source: Path,
    ids_path: Path,
    cache_path: Path,
    *,
    split: str,
    cache_report: Mapping[str, Any],
) -> tuple[tuple[RetrievalQuestion, ...], dict[str, Any]]:
    for path in (source, ids_path, cache_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = load_frozen_manifest(ids_path, expected_split=split)
    examples = load_selected_examples(source, manifest)
    row = cache_report["splits"][split]
    expected_metadata = row.get("metadata")
    if not isinstance(expected_metadata, Mapping):
        raise ValueError("Stage D cache metadata is missing")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(cache, Mapping) or not cache_matches(
        cache, expected_metadata
    ):
        raise ValueError("Stage D cache metadata does not match report")
    validate_cache_records(cache, examples)
    records = cache["records"]
    questions = tuple(
        prepare_musique_retrieval_question(example, records[example["id"]])
        for example in examples
    )
    return questions, {
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "ids_path": str(ids_path.resolve()),
        "ids_sha256": sha256_file(ids_path),
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": sha256_file(cache_path),
        "question_count": len(questions),
    }


def main() -> int:
    args = parse_args()
    if args.permutation_repetitions < 1:
        raise ValueError("permutation repetitions must be positive")
    cache_report = validate_stage_d_report(
        args.cache_report,
        train_cache=args.train_cache,
        dev_cache=args.dev_cache,
    )
    hotpot_config, hotpot_authorization = authorize_hotpot_transfer(
        args.hotpot_report, args.hotpot_validation_report
    )
    train, train_provenance = prepare_split(
        args.train_source,
        args.train_ids,
        args.train_cache,
        split="train",
        cache_report=cache_report,
    )
    dev, dev_provenance = prepare_split(
        args.dev_source,
        args.dev_ids,
        args.dev_cache,
        split="dev",
        cache_report=cache_report,
    )
    report = build_musique_graph_report(
        train,
        dev,
        provenance={
            "stage_d_report_path": str(args.cache_report.resolve()),
            "stage_d_report_sha256": sha256_file(args.cache_report),
            "train": train_provenance,
            "dev": dev_provenance,
        },
        hotpot_transfer_config=hotpot_config,
        hotpot_authorization=hotpot_authorization,
        permutation_repetitions=args.permutation_repetitions,
        permutation_seed=args.permutation_seed,
    )
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "selection": report["selection"],
                "dev_selected_graph_macro": report["dev"]["selected_graph"][
                    "macro"
                ],
                "dev_graph_vs_baseline": report["dev"][
                    "selected_graph_vs_selected_baseline"
                ],
                "dev_graph_vs_degree": report["dev"][
                    "selected_graph_vs_selected_degree_only"
                ],
                "gate": report["gate"],
            },
            sort_keys=True,
        )
    )
    if report["gate"]["passed"] or args.no_fail_on_gate:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
