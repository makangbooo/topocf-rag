#!/usr/bin/env python3
"""Analyze title-mention graph coverage on frozen HotpotQA bridge splits.

The generated report and retrieval caches contain IDs and aggregate numeric
metadata only. Questions, answers, titles, sentences, and supporting-fact text
are never written or printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import ijson
import numpy as np

from topocf_rag.graph import (
    CandidatePath,
    QueryDocumentGraph,
    build_query_document_graph,
    gold_coverage,
    validate_hotpot_example,
)


CACHE_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
GATE_THRESHOLD = 0.60


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_frozen_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("ids"), list):
        raise ValueError(f"frozen ID manifest has invalid schema: {path}")
    ids = payload["ids"]
    if any(not isinstance(question_id, str) or not question_id for question_id in ids):
        raise ValueError(f"frozen ID manifest contains an invalid ID: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"frozen ID manifest contains duplicate IDs: {path}")
    return ids


def load_selected_examples(source: Path, ordered_ids: Sequence[str]) -> list[dict[str, Any]]:
    wanted = set(ordered_ids)
    selected: dict[str, dict[str, Any]] = {}
    with source.open("rb") as stream:
        for record in ijson.items(stream, "item"):
            question_id = record.get("_id") if isinstance(record, Mapping) else None
            if question_id not in wanted:
                continue
            if question_id in selected:
                raise ValueError("source dataset contains a duplicate selected ID")
            validate_hotpot_example(record)
            if record.get("type") != "bridge":
                raise ValueError("frozen selection contains a non-bridge record")
            selected[question_id] = record

    missing = wanted.difference(selected)
    if missing:
        raise ValueError(f"source dataset is missing {len(missing)} selected IDs")
    return [selected[question_id] for question_id in ordered_ids]


def describe_counts(values: Sequence[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": statistics.median(ordered),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "zero_count": sum(value == 0 for value in ordered),
        "zero_rate": sum(value == 0 for value in ordered) / len(ordered),
    }


def count_histogram(values: Sequence[int]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for value in sorted(set(values)):
        histogram[str(value)] = values.count(value)
    return histogram


class DenseRetriever:
    """Lazily load bge-m3 and produce context-aligned scores and token counts."""

    def __init__(self, model_path: Path, device: str, batch_size: int, max_length: int):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                str(self.model_path), use_fp16=True, devices=self.device
            )
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = np.asarray(encoded["dense_vecs"], dtype=np.float32)
        if dense.ndim != 2 or dense.shape[0] != len(texts):
            raise RuntimeError("bge-m3 returned an unexpected dense embedding shape")
        if not np.isfinite(dense).all():
            raise RuntimeError("bge-m3 returned a non-finite dense embedding")
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("bge-m3 returned a zero-norm dense embedding")
        return dense / norms

    def _token_lengths(self, texts: Sequence[str]) -> list[int]:
        tokenizer = self._load().tokenizer
        lengths: list[int] = []
        for start in range(0, len(texts), self.batch_size * 4):
            batch = list(texts[start : start + self.batch_size * 4])
            tokenized = tokenizer(
                batch,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
                padding=False,
            )
            lengths.extend(len(mask) for mask in tokenized["attention_mask"])
        return lengths

    def score_examples(
        self, examples: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, dict[str, list[int] | list[float]]], dict[str, Any]]:
        questions = [str(example["question"]) for example in examples]
        document_texts: list[str] = []
        offsets = [0]
        for example in examples:
            for title, sentences in example["context"]:
                document_texts.append(f"{title}\n{' '.join(sentences)}")
            offsets.append(len(document_texts))

        started = time.perf_counter()
        query_vectors = self._encode(questions)
        document_vectors = self._encode(document_texts)
        token_lengths = self._token_lengths(document_texts)
        elapsed = time.perf_counter() - started

        records: dict[str, dict[str, list[int] | list[float]]] = {}
        for example_index, example in enumerate(examples):
            start, end = offsets[example_index : example_index + 2]
            scores = document_vectors[start:end] @ query_vectors[example_index]
            records[str(example["_id"])] = {
                "scores": [float(score) for score in scores],
                "document_token_lengths": token_lengths[start:end],
            }

        runtime: dict[str, Any] = {
            "computed": True,
            "elapsed_seconds": elapsed,
            "question_count": len(questions),
            "document_count": len(document_texts),
            "embedding_dimension": int(query_vectors.shape[1]),
        }
        try:
            import torch

            if torch.cuda.is_available() and self.device.startswith("cuda"):
                device_index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
                torch.cuda.synchronize(device_index)
                runtime["device_index"] = device_index
                runtime["device_name"] = torch.cuda.get_device_name(device_index)
                runtime["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device_index)
        except (ImportError, RuntimeError, ValueError):
            pass
        return records, runtime


def expected_cache_metadata(
    *,
    source: Path,
    ids_path: Path,
    model_path: Path,
    max_length: int,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "ids_path": str(ids_path.resolve()),
        "ids_sha256": sha256_file(ids_path),
        "model_path": str(model_path.resolve()),
        "max_length": max_length,
    }


def cache_matches(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(payload.get(key) == value for key, value in expected.items()) and isinstance(
        payload.get("records"), Mapping
    )


def path_approximate_token_length(path: CandidatePath, token_lengths: Sequence[int]) -> int:
    # Both alternatives use the same fixed serialization labels and question.
    # The sum is therefore sufficient for the <=5% matching diagnostic.
    return token_lengths[path.source_index] + token_lengths[path.target_index]


def analyze_split(
    examples: Sequence[Mapping[str, Any]],
    retrieval_records: Mapping[str, Mapping[str, Sequence[int | float]]],
    retrieval_top_k: int,
) -> dict[str, Any]:
    coverage_counts = {
        "eligible": 0,
        "ordered_directed": 0,
        "reverse_directed": 0,
        "any_direction": 0,
        "undirected": 0,
        "reverse_only": 0,
        "bidirectional": 0,
    }
    context_counts: list[int] = []
    mention_edge_counts: list[int] = []
    candidate_counts: list[int] = []
    gold_path_counts: list[int] = []
    natural_negative_counts: list[int] = []
    matched_negative_counts: list[int] = []
    best_matched_score_gaps: list[float] = []
    best_matched_token_relative_gaps: list[float] = []
    questions_with_positive_and_natural_negative = 0
    questions_with_five_percent_token_match = 0

    for example in examples:
        question_id = str(example["_id"])
        record = retrieval_records.get(question_id)
        if record is None:
            raise ValueError("retrieval cache is missing a selected ID")
        scores = record.get("scores")
        token_lengths = record.get("document_token_lengths")
        if not isinstance(scores, Sequence) or not isinstance(token_lengths, Sequence):
            raise ValueError("retrieval cache record has invalid schema")
        top_k = min(retrieval_top_k, len(example["context"]))
        graph = build_query_document_graph(
            example, [float(score) for score in scores], retrieval_top_k=top_k
        )
        coverage = gold_coverage(graph)
        for key, value in asdict(coverage).items():
            coverage_counts[key] += int(value)

        context_counts.append(len(graph.documents))
        mention_edge_counts.append(len(graph.mention_edges))
        candidate_counts.append(len(graph.candidate_paths))
        gold_path_counts.append(len(graph.gold_paths))
        natural_negative_counts.append(len(graph.natural_negative_paths))

        matched_for_question = 0
        if graph.gold_paths and graph.natural_negative_paths:
            questions_with_positive_and_natural_negative += 1
            for positive in graph.gold_paths:
                positive_tokens = path_approximate_token_length(
                    positive, [int(length) for length in token_lengths]
                )
                eligible_negatives: list[tuple[float, float]] = []
                for negative in graph.natural_negative_paths:
                    negative_tokens = path_approximate_token_length(
                        negative, [int(length) for length in token_lengths]
                    )
                    token_gap = abs(negative_tokens - positive_tokens) / max(
                        positive_tokens, 1
                    )
                    if token_gap <= 0.05:
                        score_gap = abs(
                            negative.retrieval_score - positive.retrieval_score
                        )
                        eligible_negatives.append((score_gap, token_gap))
                matched_for_question += len(eligible_negatives)
                if eligible_negatives:
                    best_score_gap, best_token_gap = min(eligible_negatives)
                    best_matched_score_gaps.append(best_score_gap)
                    best_matched_token_relative_gaps.append(best_token_gap)
            if matched_for_question:
                questions_with_five_percent_token_match += 1
        matched_negative_counts.append(matched_for_question)

    eligible = coverage_counts["eligible"]
    coverage_rates = {
        key: (coverage_counts[key] / eligible if eligible else None)
        for key in coverage_counts
        if key != "eligible"
    }
    return {
        "question_count": len(examples),
        "coverage_definition": {
            "ordered_directed": (
                "supporting_facts unique-title first-appearance order title0->title1; "
                "this order is not documented by HotpotQA as semantic chain direction"
            ),
            "any_direction": "a real directed gold candidate path in either direction",
            "undirected": (
                "a real mention edge between the two gold titles after ignoring its "
                "direction, with at least one retrieved endpoint"
            ),
        },
        "coverage_counts": coverage_counts,
        "coverage_rates": coverage_rates,
        "context_document_count": describe_counts(context_counts),
        "title_mention_edge_count": describe_counts(mention_edge_counts),
        "retrieved_candidate_path_count": describe_counts(candidate_counts),
        "gold_candidate_path_count": describe_counts(gold_path_counts),
        "natural_negative_path_count": describe_counts(natural_negative_counts),
        "natural_negative_path_count_histogram": count_histogram(natural_negative_counts),
        "matching_diagnostic": {
            "token_length_measure": (
                "sum of truncated tokenizer lengths for ordered source and target "
                "documents; fixed path labels and the shared question are omitted"
            ),
            "token_relative_gap_threshold": 0.05,
            "questions_with_positive_and_natural_negative": (
                questions_with_positive_and_natural_negative
            ),
            "questions_with_at_least_one_token_matched_negative": (
                questions_with_five_percent_token_match
            ),
            "token_matched_negative_count": describe_counts(matched_negative_counts),
            "best_token_matched_absolute_retrieval_score_gap": describe_counts(
                best_matched_score_gaps
            ),
            "best_token_matched_relative_token_gap": describe_counts(
                best_matched_token_relative_gaps
            ),
        },
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
        "--dev-source",
        type=Path,
        default=Path("/file_system/datasets/hotpotqa/hotpot_dev_distractor_v1.json"),
    )
    parser.add_argument(
        "--dev-ids",
        type=Path,
        default=Path("data/splits/hotpot_dev_bridge_500_ids.json"),
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=Path("reports/title_graph/cache"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/title_graph/analysis.json")
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--no-fail-on-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if args.batch_size < 1 or args.max_length < 1 or args.retrieval_top_k < 1:
        raise ValueError("batch size, max length, and retrieval top-k must be positive")

    split_inputs = {
        "train": (args.train_source, args.train_ids),
        "dev_distractor": (args.dev_source, args.dev_ids),
    }
    split_examples: dict[str, list[dict[str, Any]]] = {}
    split_metadata: dict[str, dict[str, Any]] = {}
    for split_name, (source, ids_path) in split_inputs.items():
        ids = load_frozen_ids(ids_path)
        split_examples[split_name] = load_selected_examples(source, ids)
        split_metadata[split_name] = expected_cache_metadata(
            source=source,
            ids_path=ids_path,
            model_path=args.model,
            max_length=args.max_length,
        )

    retriever = DenseRetriever(
        args.model, args.device, args.batch_size, args.max_length
    )
    split_records: dict[str, Mapping[str, Mapping[str, Sequence[int | float]]]] = {}
    retrieval_runtime: dict[str, Any] = {}
    for split_name in split_inputs:
        cache_path = args.cache_dir / f"{split_name}_bge_retrieval.json"
        expected = split_metadata[split_name]
        cache_payload: Mapping[str, Any] | None = None
        if cache_path.is_file() and not args.force_recompute:
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(candidate, Mapping) and cache_matches(candidate, expected):
                cache_payload = candidate

        if cache_payload is None:
            records, runtime = retriever.score_examples(split_examples[split_name])
            cache_payload = {**expected, "records": records}
            atomic_json_dump(cache_payload, cache_path)
            retrieval_runtime[split_name] = runtime
        else:
            retrieval_runtime[split_name] = {
                "computed": False,
                "cache_path": str(cache_path.resolve()),
            }
        split_records[split_name] = cache_payload["records"]

    analyses = {
        split_name: analyze_split(
            split_examples[split_name],
            split_records[split_name],
            args.retrieval_top_k,
        )
        for split_name in split_inputs
    }
    per_split_pass = {
        split_name: (
            analysis["coverage_rates"]["ordered_directed"] is not None
            and analysis["coverage_rates"]["ordered_directed"] >= GATE_THRESHOLD
        )
        for split_name, analysis in analyses.items()
    }
    passed = all(per_split_pass.values())
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "retrieval": {
            "model_path": str(args.model.resolve()),
            "device": args.device,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "top_k": args.retrieval_top_k,
            "score": "cosine(question_embedding, full_title_and_document_embedding)",
            "runtime": retrieval_runtime,
        },
        "inputs": split_metadata,
        "splits": analyses,
        "gate": {
            "metric": "ordered_directed coverage",
            "threshold": GATE_THRESHOLD,
            "per_split_pass": per_split_pass,
            "passed": passed,
            "status": "continue" if passed else "stop_and_redesign_graph",
            "interpretation_warning": (
                "supporting_facts order is not documented as semantic chain direction; "
                "do not use ordered-vs-undirected differences as evidence of direction "
                "reasoning"
            ),
        },
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "gate": report["gate"],
                "coverage_rates": {
                    split_name: analysis["coverage_rates"]
                    for split_name, analysis in analyses.items()
                },
            },
            sort_keys=True,
        )
    )
    if not passed and not args.no_fail_on_gate:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
