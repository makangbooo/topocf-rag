from __future__ import annotations

import hashlib
import json
from pathlib import Path

from topocf_rag.all_observed_population import (
    audit_all_observed_population,
    population_gate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/certificate_v1/topocf_all_observed_v2.json"


def _example(*, all_observed: bool) -> dict[str, object]:
    alpha_sentences = ["Private gold fact.", "This refers to Beta."]
    gamma_sentences = ["This refers to Delta."]
    if all_observed:
        alpha_sentences.append("This cross-refers to Delta.")
        gamma_sentences.append("This cross-refers to Beta.")
    return {
        "_id": "fixture-qid-observed" if all_observed else "fixture-qid-common",
        "question": "Private fixture question text?",
        "answer": "Private fixture answer",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": [["Alpha", 0], ["Beta", 0]],
        "context": [
            ["Alpha", alpha_sentences],
            ["Beta", ["Private second gold fact."]],
            ["Gamma", gamma_sentences],
            ["Delta", ["Private control target text."]],
        ],
    }


def test_population_audit_counts_only_all_observed_rewires() -> None:
    report = audit_all_observed_population(
        (_example(all_observed=True), _example(all_observed=False))
    )
    assert report["record_count"] == 2
    assert report["bridge_question_count"] == 2
    assert report["all_observed_question_count"] == 1
    assert report["all_observed_pair_count"] == 1
    assert report["all_observed_question_rate"] == 0.5
    assert report["all_observed_pair_count_per_bridge_question_histogram"] == {
        "0": 1,
        "1": 1,
    }
    assert report["integrity"] == {
        "all_candidate_edges_observed": True,
        "negative_nonobserved_edge_count": 0,
        "positive_nonobserved_edge_count": 0,
    }


def test_population_gate_is_frozen_to_question_counts_and_edge_integrity() -> None:
    splits = {
        "train": {
            "all_observed_question_count": 500,
            "integrity": {"all_candidate_edges_observed": True},
        },
        "dev_distractor": {
            "all_observed_question_count": 100,
            "integrity": {"all_candidate_edges_observed": True},
        },
    }
    assert population_gate(
        splits,
        minimum_train_question_count=500,
        minimum_dev_question_count=100,
    )["passed"] is True
    splits["dev_distractor"]["all_observed_question_count"] = 99
    assert population_gate(
        splits,
        minimum_train_question_count=500,
        minimum_dev_question_count=100,
    )["passed"] is False


def test_all_observed_config_hashes_predecessor_and_generator() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["motivation"]["predecessor_decision"] == "STOP_CURRENT_METHOD"
    assert config["candidate_task"]["status"] == (
        "population_feasibility_audit_only"
    )
    assert config["post_audit_rules"]["manual_annotation_required"] is False
    for artifact in config["artifacts"].values():
        path = ROOT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
