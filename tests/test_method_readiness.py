from __future__ import annotations

import json
from pathlib import Path

import pytest

from topocf_rag.method_readiness import (
    MethodReadinessInvariantError,
    build_method_readiness_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/certificate_v1/topocf_bind_repair_v1.json"


def test_frozen_method_gate_authorizes_only_bounded_development() -> None:
    report = build_method_readiness_report(CONFIG)

    assert report["gates"]["development_training"]["passed"] is True
    assert report["gates"]["confirmatory_evaluation"]["passed"] is False
    assert report["gates"]["paper_claim"]["passed"] is False
    assert report["primary_development"]["train"]["counts"] == {
        "pair_count": 368,
        "question_count": 161,
        "base_count": 368,
    }
    assert report["primary_development"]["dev_distractor"]["counts"] == {
        "pair_count": 150,
        "question_count": 65,
        "base_count": 150,
    }
    assert report["confirmatory_all_observed"]["dev_distractor"]["counts"][
        "question_count"
    ] == 12
    assert report["rule_controls"]["development_dev"][
        "best_model_visible_permutation_invariant_rule"
    ]["value"] == 0.5
    assert report["rule_controls"]["confirmatory_dev"][
        "alias_sensitive_lookup_accuracy"
    ] > 0.7

    encoded = json.dumps(report, sort_keys=True)
    assert "5a70f5825542994082a3e441" not in encoded


def test_frozen_artifact_hash_mismatch_is_a_hard_error(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for artifact in config["artifacts"].values():
        artifact["path"] = str((ROOT / artifact["path"]).resolve())
    config["artifacts"]["rule_leakage"]["sha256"] = "0" * 64
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(MethodReadinessInvariantError, match="SHA256 mismatch"):
        build_method_readiness_report(altered)
