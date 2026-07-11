#!/usr/bin/env python3
"""Materialize content-free Phase 1 A+ pair manifests from frozen HotpotQA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import ijson

from topocf_rag.pairs import (
    PAIR_SEED,
    Phase1Pair,
    build_pair_manifest,
    generate_question_pairs,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("ids") if isinstance(payload, Mapping) else None
    if not isinstance(ids, list) or any(
        not isinstance(qid, str) or not qid for qid in ids
    ):
        raise ValueError("frozen ID manifest must contain non-empty string IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("frozen ID manifest contains duplicate IDs")
    return set(ids)


def load_retrieval_cache(
    path: Path,
    *,
    expected_source_sha256: str,
    expected_ids_sha256: str,
) -> Mapping[str, Mapping[str, Sequence[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("retrieval cache must be a JSON object")
    if payload.get("source_sha256") != expected_source_sha256:
        raise ValueError("retrieval cache source SHA256 does not match")
    if payload.get("ids_sha256") != expected_ids_sha256:
        raise ValueError("retrieval cache ID SHA256 does not match")
    records = payload.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("retrieval cache must contain a records object")
    return records


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def materialize_split(
    *,
    official_split: str,
    source_path: Path,
    ids_path: Path,
    retrieval_cache_path: Path,
    output_path: Path,
    retrieval_top_k: int,
    seed: int,
) -> dict[str, Any]:
    source_sha256 = sha256_file(source_path)
    ids_sha256 = sha256_file(ids_path)
    retrieval_cache_sha256 = sha256_file(retrieval_cache_path)
    wanted = load_frozen_ids(ids_path)
    retrieval_records = load_retrieval_cache(
        retrieval_cache_path,
        expected_source_sha256=source_sha256,
        expected_ids_sha256=ids_sha256,
    )

    pairs: list[Phase1Pair] = []
    seen: set[str] = set()
    with source_path.open("rb") as stream:
        for example in ijson.items(stream, "item"):
            qid = example.get("_id") if isinstance(example, Mapping) else None
            if qid not in wanted:
                continue
            if qid in seen:
                raise ValueError("source contains a duplicate frozen ID")
            seen.add(qid)
            if example.get("type") != "bridge":
                raise ValueError("frozen selection contains a non-bridge question")
            retrieval_record = retrieval_records.get(qid)
            if not isinstance(retrieval_record, Mapping):
                raise ValueError("retrieval cache is missing a frozen question")
            scores = retrieval_record.get("scores")
            if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
                raise ValueError("retrieval cache scores must be a sequence")
            pairs.extend(
                generate_question_pairs(
                    example,
                    [float(score) for score in scores],
                    retrieval_top_k=retrieval_top_k,
                    seed=seed,
                )
            )

    missing_count = len(wanted.difference(seen))
    if missing_count:
        raise ValueError(f"source is missing {missing_count} frozen questions")
    if len(seen) != len(wanted):
        raise RuntimeError("processed question count does not match frozen selection")

    manifest = build_pair_manifest(
        pairs,
        official_split=official_split,
        source_sha256=source_sha256,
        ids_sha256=ids_sha256,
        retrieval_cache_sha256=retrieval_cache_sha256,
        retrieval_top_k=retrieval_top_k,
        seed=seed,
    )
    manifest["frozen_question_count"] = len(wanted)
    atomic_json_dump(manifest, output_path)
    return {
        "official_split": official_split,
        "output": str(output_path.resolve()),
        "counts": manifest["counts"],
    }


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
        "--train-output",
        type=Path,
        default=Path("data/phase1/hotpot_train_phase1_pairs.json"),
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
        "--dev-output",
        type=Path,
        default=Path("data/phase1/hotpot_dev_phase1_pairs.json"),
    )
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=PAIR_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.retrieval_top_k < 1:
        raise ValueError("retrieval top-k must be positive")
    summaries = [
        materialize_split(
            official_split="train",
            source_path=args.train_source,
            ids_path=args.train_ids,
            retrieval_cache_path=args.train_cache,
            output_path=args.train_output,
            retrieval_top_k=args.retrieval_top_k,
            seed=args.seed,
        ),
        materialize_split(
            official_split="dev_distractor",
            source_path=args.dev_source,
            ids_path=args.dev_ids,
            retrieval_cache_path=args.dev_cache,
            output_path=args.dev_output,
            retrieval_top_k=args.retrieval_top_k,
            seed=args.seed,
        ),
    ]
    print(json.dumps({"splits": summaries}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
