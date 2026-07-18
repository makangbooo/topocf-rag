#!/usr/bin/env python3
"""One-shot official-dev evaluation of one hash-frozen text baseline."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from topocf_rag.evaluation import prepare_split_from_files
from topocf_rag.method_baselines import (
    BASELINE_NAMES,
    RERANKER_INPUT_VERSION,
    BaselineEvaluationPair,
    MethodBaselineInvariantError,
    aggregate_candidate_logits,
    aggregate_orbit_scores,
    baseline_metric_report,
    build_baseline_evaluation_pairs,
    protocol_sha256,
    reranker_token_ids,
    resolve_frozen_baseline_checkpoint,
)
from topocf_rag.method_data import build_official_dev_evaluation_examples
from topocf_rag.metrics import ScoredPair


REPORT_SCHEMA_VERSION = 1


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_fingerprint(path: Path) -> str:
    files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
    if not files:
        raise MethodBaselineInvariantError("checkpoint directory is empty")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
        digest.update(b"\0")
    return digest.hexdigest()


def model_fingerprint(model_path: Path) -> dict[str, Any]:
    weight_names = tuple(
        sorted(
            path.name
            for path in model_path.iterdir()
            if path.is_file()
            and (
                path.suffix == ".safetensors"
                or path.name.startswith("pytorch_model")
                and path.suffix == ".bin"
            )
        )
    )
    if not weight_names:
        raise FileNotFoundError("model directory has no supported weight files")
    required = (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        *weight_names,
    )
    optional = tuple(
        name
        for name in (
            "generation_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        )
        if (model_path / name).is_file()
    )
    files = []
    combined = hashlib.sha256()
    for name in (*required, *optional):
        path = model_path / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        files.append(
            {
                "name": name,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(bytes.fromhex(digest))
        combined.update(b"\0")
    return {"sha256": combined.hexdigest(), "files": files}


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(path.parent, 0o700)
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
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def package_versions() -> dict[str, str]:
    result = {}
    for name in ("torch", "transformers", "accelerate", "peft"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/certificate_v1/topocf_text_baselines_v1.json"),
    )
    parser.add_argument("--baseline", choices=BASELINE_NAMES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/mkb524/topocf-rag-models/Qwen3-Reranker-0.6B"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/home/mkb524/topocf-rag-runs/text-baselines-v1"),
    )
    parser.add_argument(
        "--dev-manifest",
        type=Path,
        default=Path("data/phase1/hotpot_dev_phase1_pairs.json"),
    )
    parser.add_argument(
        "--dev-source",
        type=Path,
        default=Path(
            "/file_system/datasets/hotpotqa/hotpot_dev_distractor_v1.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/mkb524/topocf-rag-runs/text-baseline-dev-v1"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _require_mapping(payload: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise MethodBaselineInvariantError(f"{field} must be an object")
    return payload


def load_config(path: Path, repository_root: Path) -> Mapping[str, Any]:
    config = _require_mapping(
        json.loads(path.read_text(encoding="utf-8")), "config"
    )
    if config.get("schema_version") != 1:
        raise MethodBaselineInvariantError("unsupported text-baseline config schema")
    artifacts = _require_mapping(config.get("artifacts"), "config.artifacts")
    for name, value in artifacts.items():
        artifact = _require_mapping(value, f"artifact {name}")
        relative = artifact.get("path")
        expected = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise MethodBaselineInvariantError(f"artifact {name} is invalid")
        actual = sha256_file(repository_root / relative)
        if actual != expected:
            raise MethodBaselineInvariantError(
                f"artifact {name} hash changed: expected {expected}, got {actual}"
            )
    return config


class RerankerBatcher:
    def __init__(self, tokenizer: Any, *, max_length: int, device: str) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.yes_id = tokenizer.convert_tokens_to_ids("yes")
        self.no_id = tokenizer.convert_tokens_to_ids("no")
        if (
            not isinstance(self.yes_id, int)
            or not isinstance(self.no_id, int)
            or self.yes_id == self.no_id
            or self.yes_id < 0
            or self.no_id < 0
        ):
            raise MethodBaselineInvariantError("yes/no token IDs are invalid")

    def token_ids(self, text: str) -> tuple[int, ...]:
        return reranker_token_ids(
            self.tokenizer, text, max_length=self.max_length
        )

    def batch(self, texts: Sequence[str]) -> Mapping[str, Any]:
        encoded = {"input_ids": [list(self.token_ids(text)) for text in texts]}
        return self.tokenizer.pad(
            encoded,
            padding=True,
            return_tensors="pt",
        ).to(self.device)


def _score_evaluation(
    model: Any,
    pairs: Sequence[BaselineEvaluationPair],
    batcher: RerankerBatcher,
    *,
    batch_size: int,
) -> tuple[ScoredPair, ...]:
    import torch

    texts: list[str] = []
    candidate_sizes: list[int] = []
    pair_layout: list[tuple[str, str, int, int]] = []
    for pair in pairs:
        positive_start = len(candidate_sizes)
        for candidate in pair.positive_orbit:
            candidate_sizes.append(len(candidate.inputs))
            texts.extend(candidate.inputs)
        negative_start = len(candidate_sizes)
        for candidate in pair.negative_orbit:
            candidate_sizes.append(len(candidate.inputs))
            texts.extend(candidate.inputs)
        pair_layout.append(
            (pair.qid, pair.pair_id, positive_start, negative_start)
        )

    logits: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = batcher.batch(texts[start : start + batch_size])
            outputs = model(**batch, use_cache=False)
            final = outputs.logits[:, -1, :]
            scores = (
                final[:, batcher.yes_id].float()
                - final[:, batcher.no_id].float()
            )
            logits.extend(scores.cpu().tolist())

    candidate_scores: list[float] = []
    cursor = 0
    for size in candidate_sizes:
        candidate_scores.append(
            aggregate_candidate_logits(logits[cursor : cursor + size])
        )
        cursor += size
    if cursor != len(logits):
        raise RuntimeError("evaluation score layout changed")

    scored = []
    for qid, pair_id, positive_start, negative_start in pair_layout:
        scored.append(
            ScoredPair(
                qid,
                pair_id,
                aggregate_orbit_scores(
                    candidate_scores[positive_start : positive_start + 24]
                ),
                aggregate_orbit_scores(
                    candidate_scores[negative_start : negative_start + 24]
                ),
            )
        )
    return tuple(scored)


def main() -> int:
    args = parse_args()
    repository_root = Path.cwd().resolve()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    report_path = output_root / args.baseline / str(args.seed) / "report.json"
    if report_path.is_relative_to(repository_root):
        raise MethodBaselineInvariantError(
            "official-dev reports must stay outside the repository"
        )
    if report_path.exists():
        raise FileExistsError(
            "official-dev output already exists; one-shot evaluation cannot rerun"
        )
    output_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(output_root, 0o700)

    config = load_config(config_path, repository_root)
    if config.get("reranker_input_protocol_sha256") != protocol_sha256():
        raise MethodBaselineInvariantError("reranker input protocol changed")
    dev_protocol = _require_mapping(
        config.get("official_dev_evaluation"), "official_dev_evaluation"
    )
    if dev_protocol.get("official_split") != "dev_distractor":
        raise MethodBaselineInvariantError("official-dev split lock changed")
    mandatory_baseline = dev_protocol.get("mandatory_baseline")
    mandatory_seeds = dev_protocol.get("mandatory_seeds")
    if args.baseline != mandatory_baseline:
        raise MethodBaselineInvariantError(
            f"official-dev gate is locked to baseline={mandatory_baseline}"
        )
    if not isinstance(mandatory_seeds, list) or args.seed not in mandatory_seeds:
        raise MethodBaselineInvariantError(
            "official-dev seed is not one of the three mandatory frozen seeds"
        )
    selector = _require_mapping(dev_protocol.get("selector"), "selector")
    if selector != {"stratum": "synthetic_common", "variant": "t3"}:
        raise MethodBaselineInvariantError("official-dev selector changed")
    dev_artifact = _require_mapping(
        _require_mapping(config.get("artifacts"), "config.artifacts").get(
            "official_dev_pair_manifest"
        ),
        "official_dev_pair_manifest",
    )
    if sha256_file(args.dev_manifest) != dev_artifact.get("sha256"):
        raise MethodBaselineInvariantError("official-dev manifest hash changed")
    if args.dev_source.resolve() != Path(str(dev_protocol["source_path"])).resolve():
        raise MethodBaselineInvariantError("official-dev source path changed")
    if sha256_file(args.dev_source) != dev_protocol.get("source_sha256"):
        raise MethodBaselineInvariantError("official-dev source hash changed")

    checkpoint = resolve_frozen_baseline_checkpoint(
        config,
        baseline=args.baseline,
        seed=args.seed,
        run_root=args.checkpoint_root.resolve(),
    )
    if not checkpoint.path.is_dir():
        raise FileNotFoundError(checkpoint.path)
    actual_checkpoint_sha256 = checkpoint_fingerprint(checkpoint.path)
    if actual_checkpoint_sha256 != checkpoint.sha256:
        raise MethodBaselineInvariantError("frozen checkpoint hash changed")

    model_path = args.model.resolve()
    fingerprint = model_fingerprint(model_path)
    model_config = _require_mapping(config.get("model"), "config.model")
    if fingerprint["sha256"] != model_config.get("expected_fingerprint_sha256"):
        raise MethodBaselineInvariantError("local base-model fingerprint changed")
    training = _require_mapping(config.get("training"), "config.training")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("official-dev baseline evaluation requires CUDA")
    device_index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batcher = RerankerBatcher(
        tokenizer,
        max_length=int(training["max_length"]),
        device=args.device,
    )
    prepared, input_hashes = prepare_split_from_files(
        args.dev_manifest,
        args.dev_source,
        tokenizer,
        max_length=int(training["max_length"]),
        fail_on_truncation=True,
    )
    examples = build_official_dev_evaluation_examples(prepared)
    pairs = build_baseline_evaluation_pairs(
        examples,
        baseline=args.baseline,
    )
    question_ids = tuple(sorted({pair.qid for pair in pairs}))
    if len(question_ids) != dev_protocol.get("expected_question_count"):
        raise MethodBaselineInvariantError("official-dev question count changed")
    if len(pairs) != dev_protocol.get("expected_pair_count"):
        raise MethodBaselineInvariantError("official-dev pair count changed")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
    ).to(args.device)
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint.path,
        is_trainable=False,
    ).eval()
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    scored = _score_evaluation(
        model,
        pairs,
        batcher,
        batch_size=int(training["evaluation_batch_size"]),
    )
    metrics = baseline_metric_report(
        scored,
        expected_question_ids=question_ids,
        bootstrap_repetitions=int(training["bootstrap_repetitions"]),
        bootstrap_seed=int(training["bootstrap_seed"]),
    )
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "task": "topocf_text_baseline_official_dev_one_shot",
        "content_contract": (
            "aggregate metrics, runtime metadata, paths, and hashes only; no "
            "question IDs, questions, document text, or model inputs"
        ),
        "baseline": args.baseline,
        "seed": args.seed,
        "protocol": {
            "input_version": RERANKER_INPUT_VERSION,
            "input_protocol_sha256": protocol_sha256(),
            "official_split": "dev_distractor",
            "selector": dict(selector),
            "exact_s4_orbit_size": 24,
            "candidate_aggregation": (
                "single joint logit"
                if args.baseline == "flat_cross_encoder"
                else "arithmetic mean of four local edge logits"
            ),
            "checkpoint_selection": "hash-frozen before official-dev access",
            "training_or_tuning": False,
            "automatic_retries": 0,
        },
        "data": {
            "question_count": len(question_ids),
            "pair_count": len(pairs),
        },
        "metrics": metrics,
        "model": {
            "base_path": str(model_path),
            "base_fingerprint": fingerprint,
            "checkpoint_path": str(checkpoint.path.resolve()),
            "checkpoint_sha256": actual_checkpoint_sha256,
            "learning_rate": checkpoint.learning_rate,
            "epoch": checkpoint.epoch,
        },
        "artifacts": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "dev_manifest_path": str(args.dev_manifest.resolve()),
            **input_hashes,
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device_index),
        },
        "environment": {
            "python": sys.executable,
            "packages": package_versions(),
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device_index),
        },
    }
    report["run_fingerprint"] = stable_sha256(
        {
            "baseline": args.baseline,
            "seed": args.seed,
            "checkpoint_sha256": checkpoint.sha256,
            "config_sha256": report["artifacts"]["config_sha256"],
            "manifest_sha256": input_hashes["manifest_sha256"],
            "source_sha256": input_hashes["source_sha256"],
            "protocol_sha256": protocol_sha256(),
        }
    )
    atomic_json_dump(report, report_path)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "baseline": args.baseline,
                "seed": args.seed,
                "question_count": len(question_ids),
                "pair_count": len(pairs),
                "pairwise_accuracy": metrics["pairwise_accuracy"]["value"],
                "auroc": metrics["auroc"]["value"],
                "elapsed_seconds": elapsed,
                "peak_memory_bytes": report["runtime"]["peak_memory_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
