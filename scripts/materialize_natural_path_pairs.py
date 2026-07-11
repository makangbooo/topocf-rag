#!/usr/bin/env python3
"""Materialize real HotpotQA retrieval-error path-pair manifests."""

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

from topocf_rag.natural_pairs import (
    NaturalSelection,
    build_natural_pair_manifest,
    select_natural_retrieval_pair,
)


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_tokenizer_files(root: Path) -> str:
    """Fingerprint every available tokenizer artifact in stable name order."""

    digest = hashlib.sha256()
    found = False
    for name in TOKENIZER_FILES:
        path = root / name
        if not path.is_file():
            continue
        found = True
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    if not found:
        raise FileNotFoundError("tokenizer directory lacks tokenizer artifacts")
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
) -> Mapping[str, Mapping[str, Sequence[int | float]]]:
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
    tokenizer: Any,
    tokenizer_fingerprint_sha256: str,
    max_length: int,
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

    selections: list[NaturalSelection] = []
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
                raise ValueError("retrieval cache record has invalid score schema")
            selections.append(
                select_natural_retrieval_pair(
                    example,
                    [float(score) for score in scores],
                    tokenizer,
                    tokenizer_fingerprint_sha256=tokenizer_fingerprint_sha256,
                    retrieval_top_k=retrieval_top_k,
                    max_length=max_length,
                )
            )

    missing_count = len(wanted.difference(seen))
    if missing_count:
        raise ValueError(f"source is missing {missing_count} frozen questions")
    if len(selections) != len(wanted):
        raise RuntimeError("selection count does not match the frozen question count")

    manifest = build_natural_pair_manifest(
        selections,
        official_split=official_split,
        source_sha256=source_sha256,
        ids_sha256=ids_sha256,
        retrieval_cache_sha256=retrieval_cache_sha256,
        tokenizer_fingerprint_sha256=tokenizer_fingerprint_sha256,
        retrieval_top_k=retrieval_top_k,
        max_length=max_length,
    )
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
        default=Path("data/phase1/hotpot_train_natural_retrieval_pairs.json"),
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
        default=Path("data/phase1/hotpot_dev_natural_retrieval_pairs.json"),
    )
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("/file_system/models/embedding_models/bge-m3"),
    )
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.retrieval_top_k < 1 or args.max_length < 1:
        raise ValueError("retrieval top-k and max length must be positive")
    if not args.tokenizer.is_dir():
        raise FileNotFoundError(args.tokenizer)
    from transformers import AutoTokenizer

    tokenizer_fingerprint_sha256 = fingerprint_tokenizer_files(args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True
    )
    common = {
        "retrieval_top_k": args.retrieval_top_k,
        "tokenizer": tokenizer,
        "tokenizer_fingerprint_sha256": tokenizer_fingerprint_sha256,
        "max_length": args.max_length,
    }
    summaries = [
        materialize_split(
            official_split="train",
            source_path=args.train_source,
            ids_path=args.train_ids,
            retrieval_cache_path=args.train_cache,
            output_path=args.train_output,
            **common,
        ),
        materialize_split(
            official_split="dev_distractor",
            source_path=args.dev_source,
            ids_path=args.dev_ids,
            retrieval_cache_path=args.dev_cache,
            output_path=args.dev_output,
            **common,
        ),
    ]
    print(json.dumps({"splits": summaries}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
