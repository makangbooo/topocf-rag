#!/usr/bin/env python3
"""Run the train-selected HotpotQA title-graph retrieval kill test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import ijson

from topocf_rag.graph import validate_hotpot_example
from topocf_rag.graph_retrieval import (
    DEFAULT_TOP_KS,
    GraphRetrievalInvariantError,
    RetrievalQuestion,
    build_kill_test_report,
    prepare_retrieval_question,
)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_frozen_ids(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("ids"), list):
        raise GraphRetrievalInvariantError("frozen ID manifest has invalid schema")
    ids = tuple(payload["ids"])
    if any(not isinstance(qid, str) or not qid for qid in ids):
        raise GraphRetrievalInvariantError("frozen ID manifest contains an invalid ID")
    if len(ids) != len(set(ids)):
        raise GraphRetrievalInvariantError("frozen ID manifest contains duplicate IDs")
    return ids


def load_selected_examples(
    source: Path, ordered_ids: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    wanted = set(ordered_ids)
    selected: dict[str, dict[str, Any]] = {}
    with source.open("rb") as stream:
        for value in ijson.items(stream, "item"):
            qid = value.get("_id") if isinstance(value, Mapping) else None
            if qid not in wanted:
                continue
            if qid in selected:
                raise GraphRetrievalInvariantError(
                    "source contains a duplicate selected question ID"
                )
            validate_hotpot_example(value)
            if value.get("type") != "bridge":
                raise GraphRetrievalInvariantError(
                    "frozen selection contains a non-bridge question"
                )
            selected[qid] = value
    missing = wanted.difference(selected)
    if missing:
        raise GraphRetrievalInvariantError(
            f"source is missing {len(missing)} frozen question IDs"
        )
    return tuple(selected[qid] for qid in ordered_ids)


def load_dense_scores(
    cache_path: Path,
    ordered_ids: Sequence[str],
    examples: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, ...], ...]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if not isinstance(records, Mapping):
        raise GraphRetrievalInvariantError("dense retrieval cache has invalid schema")
    vectors: list[tuple[float, ...]] = []
    for qid, example in zip(ordered_ids, examples, strict=True):
        record = records.get(qid)
        scores = record.get("scores") if isinstance(record, Mapping) else None
        if not isinstance(scores, list):
            raise GraphRetrievalInvariantError(
                "dense retrieval cache is missing a selected score vector"
            )
        if len(scores) != len(example["context"]):
            raise GraphRetrievalInvariantError(
                "dense score count does not match official context size"
            )
        vectors.append(tuple(scores))
    return tuple(vectors)


def prepare_split(
    source: Path, ids_path: Path, cache_path: Path
) -> tuple[tuple[RetrievalQuestion, ...], dict[str, Any]]:
    for path in (source, ids_path, cache_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    ordered_ids = load_frozen_ids(ids_path)
    examples = load_selected_examples(source, ordered_ids)
    dense_scores = load_dense_scores(cache_path, ordered_ids, examples)
    questions = tuple(
        prepare_retrieval_question(example, scores)
        for example, scores in zip(examples, dense_scores, strict=True)
    )
    provenance = {
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "ids_path": str(ids_path.resolve()),
        "ids_sha256": sha256_file(ids_path),
        "dense_cache_path": str(cache_path.resolve()),
        "dense_cache_sha256": sha256_file(cache_path),
        "question_count": len(questions),
    }
    return questions, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-source",
        type=Path,
        default=Path("/file_system/datasets/hotpotqa/hotpot_train_v1.1.json"),
    )
    parser.add_argument(
        "--train-ids",
        type=Path,
        default=Path("data/splits/hotpot_train_bridge_1000_ids.json"),
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=Path("reports/title_graph/cache/train_bge_retrieval.json"),
    )
    parser.add_argument(
        "--dev-source",
        type=Path,
        default=Path("/file_system/datasets/hotpotqa/hotpot_dev_distractor_v1.json"),
    )
    parser.add_argument(
        "--dev-ids",
        type=Path,
        default=Path("data/splits/hotpot_dev_bridge_500_ids.json"),
    )
    parser.add_argument(
        "--dev-cache",
        type=Path,
        default=Path("reports/title_graph/cache/dev_distractor_bge_retrieval.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/graph_retrieval/hotpot_kill_test.json"),
    )
    parser.add_argument(
        "--no-fail-on-gate",
        action="store_true",
        help="Write and print a failed gate report but return exit code zero.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_questions, train_provenance = prepare_split(
        args.train_source, args.train_ids, args.train_cache
    )
    dev_questions, dev_provenance = prepare_split(
        args.dev_source, args.dev_ids, args.dev_cache
    )
    report = build_kill_test_report(
        train_questions,
        dev_questions,
        provenance={"train": train_provenance, "dev_distractor": dev_provenance},
        top_ks=DEFAULT_TOP_KS,
    )
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "selection": report["selection"],
                "dev_selected_baseline": report["dev"][
                    "selected_baseline_metrics"
                ],
                "dev_selected_graph": report["dev"]["selected_graph_metrics"],
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
