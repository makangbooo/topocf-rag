#!/usr/bin/env python3
"""Evaluate Hotpot-transfer and train-selected graph retrieval on 2Wiki."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from topocf_rag.graph_retrieval import GraphMethodConfig, RetrievalQuestion
from topocf_rag.hotpot import sha256_file
from topocf_rag.twowiki import atomic_json_dump
from topocf_rag.twowiki_graph_retrieval import (
    FROZEN_HOTPOT_TRANSFER_CONFIG,
    build_twowiki_graph_retrieval_report,
    prepare_twowiki_retrieval_question,
)
from topocf_rag.twowiki_retrieval import (
    cache_matches,
    expected_cache_metadata,
    load_frozen_manifest,
    load_selected_examples,
    validate_cache_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("/file_system/datasets/2wikimultihopqa_official_20210407")
    parser.add_argument("--train-source", type=Path, default=root / "train.json")
    parser.add_argument("--dev-source", type=Path, default=root / "dev.json")
    parser.add_argument(
        "--train-ids",
        type=Path,
        default=Path("data/splits/2wiki_train_clean_balanced_1000_ids.json"),
    )
    parser.add_argument(
        "--dev-ids",
        type=Path,
        default=Path("data/splits/2wiki_dev_clean_balanced_500_ids.json"),
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=Path(
            "reports/graph_retrieval/cache/a100/"
            "2wiki_train_bge_retrieval.json"
        ),
    )
    parser.add_argument(
        "--dev-cache",
        type=Path,
        default=Path(
            "reports/graph_retrieval/cache/a100/"
            "2wiki_dev_bge_retrieval.json"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/file_system/models/embedding_models/bge-m3"),
    )
    parser.add_argument("--max-length", type=int, default=512)
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
        "--output",
        type=Path,
        default=Path(
            "reports/graph_retrieval/2wiki_graph_retrieval.json"
        ),
    )
    parser.add_argument("--no-fail-on-gate", action="store_true")
    return parser.parse_args()


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
        raise ValueError("Hotpot validation does not bind the supplied base report")
    selection = base.get("selection")
    config_payload = (
        selection.get("selected_graph_config")
        if isinstance(selection, Mapping)
        else None
    )
    if not isinstance(config_payload, Mapping):
        raise ValueError("Hotpot selected graph configuration is missing")
    config = GraphMethodConfig(**config_payload)
    if asdict(config) != asdict(FROZEN_HOTPOT_TRANSFER_CONFIG):
        raise ValueError("Hotpot selected configuration changed from frozen transfer")
    if validation_base.get("selected_graph_config") != asdict(config):
        raise ValueError("Hotpot validation selected configuration does not match")
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
    model_path: Path,
    max_length: int,
) -> tuple[tuple[RetrievalQuestion, ...], dict[str, Any]]:
    for path in (source, ids_path, cache_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = load_frozen_manifest(ids_path)
    examples = load_selected_examples(source, manifest)
    expected = expected_cache_metadata(
        source=source,
        ids_path=ids_path,
        model_path=model_path,
        max_length=max_length,
    )
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(cache, Mapping) or not cache_matches(cache, expected):
        raise ValueError("A100 retrieval cache metadata does not match inputs")
    validate_cache_records(cache, examples)
    records = cache["records"]
    questions = tuple(
        prepare_twowiki_retrieval_question(example, records[example["_id"]]["scores"])
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
    transfer_config, authorization = authorize_hotpot_transfer(
        args.hotpot_report, args.hotpot_validation_report
    )
    train_questions, train_provenance = prepare_split(
        args.train_source,
        args.train_ids,
        args.train_cache,
        model_path=args.model,
        max_length=args.max_length,
    )
    dev_questions, dev_provenance = prepare_split(
        args.dev_source,
        args.dev_ids,
        args.dev_cache,
        model_path=args.model,
        max_length=args.max_length,
    )
    report = build_twowiki_graph_retrieval_report(
        train_questions,
        dev_questions,
        provenance={"train": train_provenance, "dev": dev_provenance},
        hotpot_transfer_config=transfer_config,
        hotpot_authorization=authorization,
    )
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "selection": report["selection"],
                "dev_zero_shot_macro": report["dev"]["hotpot_zero_shot_graph"][
                    "macro"
                ],
                "dev_zero_shot_comparison": report["dev"][
                    "zero_shot_vs_matching_baseline"
                ]["absolute_macro_mean_complete_delta"],
                "dev_tuned_macro": report["dev"]["selected_2wiki_graph"][
                    "macro"
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
