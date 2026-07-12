#!/usr/bin/env python3
"""Make the one authorized OpenAI-compatible API smoke call."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import httpx

from topocf_rag.api_generation import (
    SMOKE_TIMEOUT_SECONDS,
    execute_smoke_call,
)
from topocf_rag.evaluation import atomic_json_dump


REQUIRED_ENVIRONMENT_VARIABLES = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/certificate_v1/openai_smoke.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = [name for name in REQUIRED_ENVIRONMENT_VARIABLES if not os.getenv(name)]
    if missing:
        print(json.dumps({"missing_environment_variables": missing}, sort_keys=True))
        return 2

    limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)
    timeout = httpx.Timeout(SMOKE_TIMEOUT_SECONDS)
    with httpx.Client(timeout=timeout, limits=limits) as client:
        report = execute_smoke_call(
            client,
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.environ["OPENAI_MODEL"],
        )
    atomic_json_dump(report, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status_code"] == 200 and report["exact_match"] else 2


if __name__ == "__main__":
    sys.exit(main())
