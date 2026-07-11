#!/usr/bin/env python3
"""Capture a credential-free, machine-readable Phase 0 environment snapshot."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reports/environment"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (args.output_dir / "pip-freeze.txt").write_text(freeze, encoding="utf-8")

    cuda_available = torch.cuda.is_available()
    snapshot = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count() if cuda_available else 0)
        ],
    }
    (args.output_dir / "runtime.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(snapshot, sort_keys=True))


if __name__ == "__main__":
    main()

