from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from scripts.evaluate_2wiki_graph_retrieval import authorize_hotpot_transfer
from topocf_rag.hotpot import sha256_file
from topocf_rag.twowiki_graph_retrieval import FROZEN_HOTPOT_TRANSFER_CONFIG


def _write_reports(tmp_path: Path, *, validation_passed: bool = True):
    base_path = tmp_path / "base.json"
    validation_path = tmp_path / "validation.json"
    config = asdict(FROZEN_HOTPOT_TRANSFER_CONFIG)
    base = {
        "selection": {"selected_graph_config": config},
        "gate": {"passed": True},
    }
    base_path.write_text(json.dumps(base), encoding="utf-8")
    validation = {
        "base_report": {
            "sha256": sha256_file(base_path),
            "selected_graph_config": config,
        },
        "gate": {"passed": validation_passed},
    }
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    return base_path, validation_path


def test_hotpot_transfer_requires_both_bound_passed_reports(tmp_path: Path) -> None:
    base, validation = _write_reports(tmp_path)
    config, authorization = authorize_hotpot_transfer(base, validation)
    assert config == FROZEN_HOTPOT_TRANSFER_CONFIG
    assert authorization["base_report_sha256"] == sha256_file(base)
    assert authorization["validation_gate_passed"] is True


def test_hotpot_transfer_rejects_failed_validation(tmp_path: Path) -> None:
    base, validation = _write_reports(tmp_path, validation_passed=False)
    with pytest.raises(ValueError, match="validation gate"):
        authorize_hotpot_transfer(base, validation)
