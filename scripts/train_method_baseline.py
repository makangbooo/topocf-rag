#!/usr/bin/env python3
"""Train one leakage-controlled Qwen3 TopoCF text baseline."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import random
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
    build_baseline_training_pairs,
    protocol_sha256,
    reranker_token_ids,
    resolve_replication_epoch,
)
from topocf_rag.method_data import (
    METHOD_DATA_SEED,
    build_evaluation_examples,
    build_training_examples,
    load_inner_split_manifest,
    source_pair_manifest_sha256,
)
from topocf_rag.metrics import ScoredPair


CONFIG_SCHEMA_VERSION = 1
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


def model_fingerprint(model_path: Path) -> dict[str, Any]:
    names = tuple(
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
    if not names:
        raise FileNotFoundError("model directory has no supported weight files")
    required = ("config.json", "tokenizer_config.json", "tokenizer.json", *names)
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
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/mkb524/topocf-rag-models/Qwen3-Reranker-0.6B"),
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("data/phase1/hotpot_train_phase1_pairs.json"),
    )
    parser.add_argument(
        "--train-source",
        type=Path,
        default=Path("/file_system/datasets/hotpotqa/hotpot_train_v1.1.json"),
    )
    parser.add_argument(
        "--inner-split",
        type=Path,
        default=Path("data/splits/topocf_t3_train_inner_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/mkb524/topocf-rag-runs/text-baselines-v1"),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-pairs", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--replicate-selected",
        action="store_true",
        help=(
            "For frozen non-selection seeds, train exactly the selected learning "
            "rate and epoch without performing another validation selection."
        ),
    )
    return parser.parse_args()


def _require_mapping(payload: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise MethodBaselineInvariantError(f"{field} must be an object")
    return payload


def load_config(path: Path, repository_root: Path) -> Mapping[str, Any]:
    payload = _require_mapping(json.loads(path.read_text(encoding="utf-8")), "config")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise MethodBaselineInvariantError("unsupported text-baseline config schema")
    artifacts = _require_mapping(payload.get("artifacts"), "config.artifacts")
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
    return payload


def resolve_training_config(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> dict[str, Any]:
    frozen = _require_mapping(config.get("training"), "config.training")
    resolved = dict(frozen)
    overrides = (
        (args.epochs, "epochs"),
        (args.max_length, "max_length"),
        (args.evaluation_batch_size, "evaluation_batch_size"),
        (args.gradient_accumulation_pairs, "gradient_accumulation_pairs"),
    )
    if not (args.smoke or args.audit_only) and any(
        argument is not None for argument, _key in overrides
    ):
        raise MethodBaselineInvariantError(
            "formal runs cannot override frozen epoch, length, batch, or accumulation settings"
        )
    for argument, key in overrides:
        if argument is not None:
            resolved[key] = argument
    resolved["learning_rate"] = args.learning_rate
    if args.smoke:
        resolved["epochs"] = 1
        resolved["bootstrap_repetitions"] = 100
        resolved["early_stopping_patience"] = 1
    positive_integer_keys = (
        "epochs",
        "max_length",
        "evaluation_batch_size",
        "gradient_accumulation_pairs",
        "bootstrap_repetitions",
        "early_stopping_patience",
    )
    if any(
        not isinstance(resolved.get(key), int)
        or isinstance(resolved.get(key), bool)
        or resolved[key] < 1
        for key in positive_integer_keys
    ):
        raise MethodBaselineInvariantError("integer training settings must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise MethodBaselineInvariantError("learning rate must be positive and finite")
    candidates = frozen.get("learning_rate_candidates")
    if not isinstance(candidates, list) or args.learning_rate not in candidates:
        raise MethodBaselineInvariantError(
            "learning rate must be one of the frozen candidates"
        )
    seeds = frozen.get("seeds")
    if not isinstance(seeds, list) or args.seed not in seeds:
        raise MethodBaselineInvariantError("seed must be one of the frozen seeds")
    return resolved


def _rank(seed: int, epoch: int, pair_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{epoch}\0{pair_id}".encode("utf-8")).digest()


def _token_length_summary(lengths: Sequence[int]) -> dict[str, Any]:
    if not lengths:
        raise MethodBaselineInvariantError("token inventory is empty")
    ordered = sorted(lengths)

    def quantile(probability: float) -> int:
        index = math.ceil(probability * len(ordered)) - 1
        return ordered[max(0, min(index, len(ordered) - 1))]

    return {
        "input_count": len(ordered),
        "minimum": ordered[0],
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "maximum": ordered[-1],
    }


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


def _model_scores(model: Any, batch: Mapping[str, Any], batcher: RerankerBatcher) -> Any:
    outputs = model(**batch, use_cache=False)
    final = outputs.logits[:, -1, :]
    return final[:, batcher.yes_id].float() - final[:, batcher.no_id].float()


def _candidate_score_tensor(values: Any) -> Any:
    if values.numel() < 1:
        raise MethodBaselineInvariantError("candidate has no local score")
    return values.mean()


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
            logits.extend(_model_scores(model, batch, batcher).cpu().tolist())

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
        positive = aggregate_orbit_scores(
            candidate_scores[positive_start : positive_start + 24]
        )
        negative = aggregate_orbit_scores(
            candidate_scores[negative_start : negative_start + 24]
        )
        scored.append(ScoredPair(qid, pair_id, positive, negative))
    return tuple(scored)


def _validation_key(report: Mapping[str, Any], epoch: int) -> tuple[float, float, int]:
    pairwise = _require_mapping(report.get("pairwise_accuracy"), "pairwise_accuracy")
    margins = _require_mapping(report.get("score_margins"), "score_margins")
    accuracy = pairwise.get("value")
    mean_margin = margins.get("mean_of_question_mean_margins")
    if not isinstance(accuracy, (int, float)) or not isinstance(
        mean_margin, (int, float)
    ):
        raise MethodBaselineInvariantError("validation metrics are unavailable")
    return (float(accuracy), float(mean_margin), -epoch)


def _checkpoint_fingerprint(path: Path) -> str:
    files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
    if not files:
        raise RuntimeError("checkpoint directory is empty")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
        digest.update(b"\0")
    return digest.hexdigest()


def _save_checkpoint(model: Any, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True, mode=0o700)
    model.save_pretrained(path, safe_serialization=True)
    return {
        "path": str(path.resolve()),
        "sha256": _checkpoint_fingerprint(path),
    }


def main() -> int:
    args = parse_args()
    repository_root = Path.cwd().resolve()
    config_path = args.config.resolve()
    model_path = args.model.resolve()
    private_output_root = args.output_dir.resolve()
    learning_rate_label = format(args.learning_rate, ".0e")
    run_suffix = "-audit" if args.audit_only else "-smoke" if args.smoke else ""
    output_dir = (
        private_output_root
        / args.baseline
        / f"lr-{learning_rate_label}"
        / f"{args.seed}{run_suffix}"
    )
    report_path = (
        args.report.resolve() if args.report is not None else output_dir / "report.json"
    )
    for path in (output_dir, report_path):
        if path.is_relative_to(repository_root):
            raise MethodBaselineInvariantError(
                "checkpoints and text-baseline run reports must stay outside the repository"
            )
    if output_dir.exists() and not args.audit_only:
        raise FileExistsError(
            f"output directory already exists; choose a new seed/path: {output_dir}"
        )
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not args.train_source.is_file():
        raise FileNotFoundError(args.train_source)
    private_output_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(private_output_root, 0o700)

    config = load_config(config_path, repository_root)
    if config.get("reranker_input_protocol_sha256") != protocol_sha256():
        raise MethodBaselineInvariantError(
            "reranker input implementation changed from the frozen config"
        )
    training = resolve_training_config(args, config)
    replication_epoch = resolve_replication_epoch(
        config,
        baseline=args.baseline,
        seed=args.seed,
        learning_rate=args.learning_rate,
        replicate_selected=args.replicate_selected,
        smoke_only=args.smoke,
        audit_only=args.audit_only,
    )
    if replication_epoch is not None:
        training["epochs"] = replication_epoch
    versions = package_versions()
    fingerprint = model_fingerprint(model_path)
    model_config = _require_mapping(config.get("model"), "config.model")
    expected_model_fingerprint = model_config.get("expected_fingerprint_sha256")
    if expected_model_fingerprint is None and not (args.audit_only or args.smoke):
        raise MethodBaselineInvariantError(
            "formal training is locked until the audit freezes the model fingerprint"
        )
    if (
        expected_model_fingerprint is not None
        and expected_model_fingerprint != fingerprint["sha256"]
    ):
        raise MethodBaselineInvariantError("local model fingerprint changed")
    pair_manifest_sha256 = source_pair_manifest_sha256(args.train_manifest)
    inner = load_inner_split_manifest(
        args.inner_split,
        expected_pair_manifest_sha256=pair_manifest_sha256,
    )

    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("text-baseline training requires CUDA")
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
        args.train_manifest,
        args.train_source,
        tokenizer,
        max_length=int(training["max_length"]),
        fail_on_truncation=True,
    )
    validation_examples = build_evaluation_examples(prepared, inner)
    validation_pairs = build_baseline_evaluation_pairs(
        validation_examples, baseline=args.baseline
    )
    if args.smoke:
        validation_pairs = validation_pairs[:2]
    validation_ids = tuple(sorted({pair.qid for pair in validation_pairs}))
    validation_texts = tuple(
        text
        for pair in validation_pairs
        for orbit in (pair.positive_orbit, pair.negative_orbit)
        for candidate in orbit
        for text in candidate.inputs
    )
    validation_lengths = tuple(
        len(batcher.token_ids(text)) for text in validation_texts
    )
    training_epoch_zero = build_baseline_training_pairs(
        build_training_examples(prepared, inner, epoch=0, seed=METHOD_DATA_SEED),
        baseline=args.baseline,
    )
    if args.smoke:
        training_epoch_zero = training_epoch_zero[:8]
    training_lengths = tuple(
        len(batcher.token_ids(text))
        for pair in training_epoch_zero
        for candidate in (pair.positive, pair.negative)
        for text in candidate.inputs
    )
    audit = {
        "fit": _token_length_summary(training_lengths),
        "inner_validation_exact_s4": _token_length_summary(validation_lengths),
    }
    common_report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "task": "topocf_text_baseline_training",
        "content_contract": (
            "aggregate metrics, runtime metadata, paths, and hashes only; "
            "no question IDs, questions, document text, or model inputs"
        ),
        "baseline": args.baseline,
        "smoke_only": args.smoke,
        "audit_only": args.audit_only,
        "protocol": {
            "input_version": RERANKER_INPUT_VERSION,
            "input_protocol_sha256": protocol_sha256(),
            "train_s4": "one synchronized epoch-dependent permutation per pair",
            "validation_s4": "exact mean over all 24 synchronized permutations",
            "score": "yes logit minus no logit",
            "candidate_aggregation": (
                "single joint logit"
                if args.baseline == "flat_cross_encoder"
                else "arithmetic mean of four local edge logits"
            ),
            "checkpoint_selection_mode": (
                "fixed_epoch_replication"
                if replication_epoch is not None
                else "inner_validation_selection"
            ),
            "frozen_replication_epoch": replication_epoch,
            "automatic_retries": 0,
        },
        "model": {
            "path": str(model_path),
            "fingerprint": fingerprint,
            "yes_token_id": batcher.yes_id,
            "no_token_id": batcher.no_id,
        },
        "artifacts": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "inner_split_sha256": sha256_file(args.inner_split),
            "train_manifest_sha256": pair_manifest_sha256,
            **input_hashes,
        },
        "data": {
            "fit_question_count": len(
                {pair.qid for pair in training_epoch_zero}
            ),
            "fit_pair_count": len(training_epoch_zero),
            "validation_question_count": len(validation_ids),
            "validation_pair_count": len(validation_pairs),
            "official_dev_used": False,
        },
        "token_lengths": audit,
        "training": training,
        "seed": args.seed,
        "environment": {
            "python": sys.executable,
            "packages": versions,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device_index),
        },
    }
    if args.audit_only:
        atomic_json_dump(common_report, report_path)
        print(json.dumps(common_report, sort_keys=True))
        return 0
    if versions["peft"] == "NOT_INSTALLED":
        raise RuntimeError("peft is required for LoRA baseline training")

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
    ).to(args.device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = _require_mapping(training.get("lora"), "training.lora")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            bias=str(lora["bias"]),
            target_modules=list(lora["target_modules"]),
        ),
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    zero_scores = _score_evaluation(
        model,
        validation_pairs,
        batcher,
        batch_size=int(training["evaluation_batch_size"]),
    )
    zero_report = baseline_metric_report(
        zero_scores,
        expected_question_ids=validation_ids,
        bootstrap_repetitions=int(training["bootstrap_repetitions"]),
        bootstrap_seed=int(training["bootstrap_seed"]),
    )
    history.append({"epoch": 0, "training_loss": None, "validation": zero_report})
    output_dir.mkdir(parents=True, mode=0o700)
    best_epoch: int | None
    best_key: tuple[float, float, int] | None
    if replication_epoch is None:
        checkpoint = _save_checkpoint(model, output_dir / "checkpoint-epoch-0")
        checkpoints.append({"epoch": 0, **checkpoint})
        best_epoch = 0
        best_key = _validation_key(zero_report, 0)
    else:
        best_epoch = None
        best_key = None
    epochs_without_improvement = 0
    accumulation = int(training["gradient_accumulation_pairs"])

    for epoch in range(1, int(training["epochs"]) + 1):
        epoch_pairs = build_baseline_training_pairs(
            build_training_examples(
                prepared, inner, epoch=epoch, seed=METHOD_DATA_SEED
            ),
            baseline=args.baseline,
        )
        if args.smoke:
            epoch_pairs = epoch_pairs[:8]
        epoch_pairs = tuple(
            sorted(
                epoch_pairs,
                key=lambda pair: (
                    _rank(args.seed, epoch, pair.pair_id),
                    pair.pair_id,
                ),
            )
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index, pair in enumerate(epoch_pairs, start=1):
            texts = (*pair.positive.inputs, *pair.negative.inputs)
            scores = _model_scores(model, batcher.batch(texts), batcher)
            positive_size = len(pair.positive.inputs)
            positive = _candidate_score_tensor(scores[:positive_size])
            negative = _candidate_score_tensor(scores[positive_size:])
            loss = torch.relu(float(training["margin"]) - (positive - negative))
            block_start = ((index - 1) // accumulation) * accumulation
            block_size = min(accumulation, len(epoch_pairs) - block_start)
            (loss / block_size).backward()
            losses.append(float(loss.detach().cpu()))
            if index % accumulation == 0 or index == len(epoch_pairs):
                torch.nn.utils.clip_grad_norm_(
                    (
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    float(training["gradient_clip_norm"]),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        scored = _score_evaluation(
            model,
            validation_pairs,
            batcher,
            batch_size=int(training["evaluation_batch_size"]),
        )
        validation_report = baseline_metric_report(
            scored,
            expected_question_ids=validation_ids,
            bootstrap_repetitions=int(training["bootstrap_repetitions"]),
            bootstrap_seed=int(training["bootstrap_seed"]),
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": sum(losses) / len(losses),
                "validation": validation_report,
            }
        )
        if replication_epoch is not None:
            if epoch == replication_epoch:
                best_epoch = epoch
                checkpoint = _save_checkpoint(
                    model, output_dir / f"checkpoint-epoch-{epoch}"
                )
                checkpoints.append({"epoch": epoch, **checkpoint})
        else:
            key = _validation_key(validation_report, epoch)
            if best_key is None:
                raise RuntimeError("selection key was not initialized")
            if key > best_key:
                best_key = key
                best_epoch = epoch
                epochs_without_improvement = 0
                checkpoint = _save_checkpoint(
                    model, output_dir / f"checkpoint-epoch-{epoch}"
                )
                checkpoints.append({"epoch": epoch, **checkpoint})
            else:
                epochs_without_improvement += 1
        partial = {
            **common_report,
            "model_parameters": {
                "trainable": trainable,
                "total": total,
                "trainable_fraction": trainable / total,
            },
            "history": history,
            "selected_epoch": best_epoch,
            "checkpoints": checkpoints,
            "runtime": {
                "elapsed_seconds": time.perf_counter() - started,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device_index),
            },
        }
        atomic_json_dump(partial, report_path)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "training_loss": history[-1]["training_loss"],
                    "pairwise_accuracy": validation_report["pairwise_accuracy"]["value"],
                    "mean_margin": validation_report["score_margins"][
                        "mean_of_question_mean_margins"
                    ],
                    "selected_epoch": best_epoch,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if (
            replication_epoch is None
            and epochs_without_improvement
            >= int(training["early_stopping_patience"])
        ):
            break

    if best_epoch is None:
        raise RuntimeError("frozen replication epoch was not completed")
    final = {
        **common_report,
        "run_fingerprint": stable_sha256(
            {
                "baseline": args.baseline,
                "seed": args.seed,
                "protocol_sha256": protocol_sha256(),
                "model_sha256": fingerprint["sha256"],
                "config_sha256": sha256_file(config_path),
                "training": training,
                "replication_epoch": replication_epoch,
            }
        ),
        "model_parameters": {
            "trainable": trainable,
            "total": total,
            "trainable_fraction": trainable / total,
        },
        "history": history,
        "selected_epoch": best_epoch,
        "selected_validation": history[best_epoch]["validation"],
        "checkpoints": checkpoints,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device_index),
        },
    }
    atomic_json_dump(final, report_path)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "selected_epoch": best_epoch,
                "selected_pairwise_accuracy": final["selected_validation"][
                    "pairwise_accuracy"
                ]["value"],
                "elapsed_seconds": final["runtime"]["elapsed_seconds"],
                "peak_memory_bytes": final["runtime"]["peak_memory_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
