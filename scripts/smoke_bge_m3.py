#!/usr/bin/env python3
"""Run a deterministic 20-text dense-embedding smoke test on one CUDA device."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel


SMOKE_TEXTS = [
    f"Synthetic embedding smoke-test sentence number {index}."
    for index in range(1, 21)
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("reports/bge_smoke.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Phase 0 bge-m3 smoke test")
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)

    device_index = torch.device(args.device).index or 0
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    started = time.perf_counter()
    model = BGEM3FlagModel(str(args.model), use_fp16=True, devices=args.device)
    encoded = model.encode(
        SMOKE_TEXTS,
        batch_size=4,
        max_length=64,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    torch.cuda.synchronize(device_index)
    elapsed = time.perf_counter() - started

    dense = np.asarray(encoded["dense_vecs"])
    if dense.shape[0] != 20 or dense.ndim != 2:
        raise AssertionError(f"unexpected dense output shape: {dense.shape}")
    if not np.isfinite(dense).all():
        raise AssertionError("embedding output contains a non-finite value")

    report = {
        "device_index": device_index,
        "device_name": torch.cuda.get_device_name(device_index),
        "dtype": str(dense.dtype),
        "elapsed_seconds_including_model_load": elapsed,
        "input_count": len(SMOKE_TEXTS),
        "max_length": 64,
        "model_path": str(args.model.resolve()),
        "output_shape": list(dense.shape),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device_index),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
