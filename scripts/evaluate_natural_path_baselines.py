#!/usr/bin/env python3
"""Evaluate BM25 and bge-m3 on real natural retrieval-error paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from topocf_rag.baselines import BM25Config
from topocf_rag.evaluation import (
    BGEM3DenseEmbedder,
    atomic_json_dump,
    build_score_cache,
    cache_config,
    load_score_cache,
    score_prepared_split,
)
from topocf_rag.natural_evaluation import (
    evaluate_natural_split,
    prepare_natural_split_from_files,
)


def fingerprint_files(root: Path, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    found = False
    for name in names:
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
        raise FileNotFoundError("model directory lacks required configuration files")
    return digest.hexdigest()


def fingerprint_model(root: Path) -> str:
    weight_names = tuple(
        sorted(
            path.name
            for path in root.iterdir()
            if path.is_file()
            and (
                (path.name.startswith("pytorch_model") and path.suffix == ".bin")
                or path.suffix == ".safetensors"
                or path.name in {"sparse_linear.pt", "colbert_linear.pt"}
            )
        )
    )
    if not weight_names:
        raise FileNotFoundError("model directory lacks a supported weight artifact")
    return fingerprint_files(
        root,
        (
            "config.json",
            "configuration.json",
            "config_sentence_transformers.json",
            "modules.json",
            "sentence_bert_config.json",
            *weight_names,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("data/phase1/hotpot_train_natural_retrieval_pairs.json"),
    )
    parser.add_argument(
        "--train-source",
        type=Path,
        default=Path("/file_system/datasets/hotpotqa/hotpot_train_v1.1.json"),
    )
    parser.add_argument(
        "--dev-manifest",
        type=Path,
        default=Path("data/phase1/hotpot_dev_natural_retrieval_pairs.json"),
    )
    parser.add_argument(
        "--dev-source",
        type=Path,
        default=Path("/file_system/datasets/hotpotqa/hotpot_dev_distractor_v1.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/file_system/models/embedding_models/bge-m3"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("reports/phase1/cache")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/phase1/natural_baselines.json"),
    )
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def evaluate_one_split(
    *,
    split_key: str,
    manifest_path: Path,
    source_path: Path,
    embedder: BGEM3DenseEmbedder,
    model_fingerprint_sha256: str,
    tokenizer_config_sha256: str,
    cache_dir: Path,
    max_length: int,
    bm25_config: BM25Config,
    force_recompute: bool,
) -> dict[str, Any]:
    natural, input_hashes = prepare_natural_split_from_files(
        manifest_path,
        source_path,
        embedder.tokenizer,
        max_length=max_length,
        fail_on_truncation=True,
        expected_tokenizer_files_sha256=tokenizer_config_sha256,
    )
    config = cache_config(
        **input_hashes,
        model_fingerprint_sha256=model_fingerprint_sha256,
        tokenizer_config_sha256=tokenizer_config_sha256,
        max_length=max_length,
        bm25_config=bm25_config,
    )
    cache_path = cache_dir / f"natural_{split_key}_baseline_scores.json"
    scores = None if force_recompute else load_score_cache(
        cache_path, natural.prepared, config
    )
    if scores is None:
        scores = score_prepared_split(
            natural.prepared, embedder, bm25_config=bm25_config
        )
        atomic_json_dump(
            build_score_cache(natural.prepared, scores, config), cache_path
        )
    report = evaluate_natural_split(natural, scores)
    report["input_hashes"] = input_hashes
    report["evaluation_config_sha256"] = config["evaluation_config_sha256"]
    return report


def main() -> int:
    args = parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("batch size and max length must be positive")
    bm25_config = BM25Config(k1=args.bm25_k1, b=args.bm25_b)
    model_fingerprint_sha256 = fingerprint_model(args.model)
    tokenizer_config_sha256 = fingerprint_files(
        args.model,
        (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
        ),
    )
    embedder = BGEM3DenseEmbedder(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    common = {
        "embedder": embedder,
        "model_fingerprint_sha256": model_fingerprint_sha256,
        "tokenizer_config_sha256": tokenizer_config_sha256,
        "cache_dir": args.cache_dir,
        "max_length": args.max_length,
        "bm25_config": bm25_config,
        "force_recompute": args.force_recompute,
    }
    splits = {
        "train": evaluate_one_split(
            split_key="train",
            manifest_path=args.train_manifest,
            source_path=args.train_source,
            **common,
        ),
        "dev_distractor": evaluate_one_split(
            split_key="dev_distractor",
            manifest_path=args.dev_manifest,
            source_path=args.dev_source,
            **common,
        ),
    }
    payload = {
        "schema_version": 1,
        "model_fingerprint_sha256": model_fingerprint_sha256,
        "tokenizer_config_sha256": tokenizer_config_sha256,
        "max_length": args.max_length,
        "fail_on_truncation": True,
        "splits": splits,
    }
    atomic_json_dump(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "coverage": {
                    name: report["question_coverage"]
                    for name, report in splits.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
