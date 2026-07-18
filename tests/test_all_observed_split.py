from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from topocf_rag.all_observed_split import (
    AllObservedSplitInvariantError,
    EligibleQuestion,
    build_dev_manifest,
    build_train_manifest,
    partition_train_questions,
    scan_all_observed_questions,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/certificate_v1/topocf_all_observed_split_v2.json"


def _record(*, qid: str, level: str = "hard", observed: bool = True):
    alpha = ["Gold fact.", "This mentions Beta."]
    gamma = ["This mentions Delta."]
    if observed:
        alpha.append("This also mentions Delta.")
        gamma.append("This also mentions Beta.")
    return {
        "_id": qid,
        "question": "Private fixture question?",
        "answer": "Private fixture answer",
        "type": "bridge",
        "level": level,
        "supporting_facts": [["Alpha", 0], ["Beta", 0]],
        "context": [
            ["Alpha", alpha],
            ["Beta", ["Second gold fact."]],
            ["Gamma", gamma],
            ["Delta", ["Control target."]],
        ],
    }


def test_scan_matches_all_observed_selector_and_filters_graph_invalid() -> None:
    common = _record(qid="common", observed=False)
    observed = _record(qid="observed", observed=True)
    comparison = _record(qid="comparison", observed=True)
    comparison["type"] = "comparison"
    invalid = _record(qid="invalid", observed=True)
    invalid["context"][1][0] = "Alpha"

    questions, summary = scan_all_observed_questions(
        (common, observed, comparison, invalid)
    )

    assert questions == (
        EligibleQuestion(qid="observed", level="hard", pair_count=1),
    )
    assert summary == {
        "record_count": 4,
        "bridge_question_count": 3,
        "graph_eligible_bridge_question_count": 2,
        "graph_ineligible_bridge_question_count": 1,
        "all_observed_question_count": 1,
        "all_observed_pair_count": 1,
    }


def test_partition_is_deterministic_stratified_and_question_disjoint() -> None:
    questions = tuple(
        EligibleQuestion(
            qid=f"q-{index:03d}",
            level=("easy", "medium", "hard")[index % 3],
            pair_count=(1, 2, 3, 5)[index % 4],
        )
        for index in range(60)
    )
    first = partition_train_questions(
        questions, validation_question_count=12, seed=20260719
    )
    second = partition_train_questions(
        tuple(reversed(questions)), validation_question_count=12, seed=20260719
    )
    assert first == second
    fit, validation, allocation = first
    assert len(fit) == 48
    assert len(validation) == 12
    assert set(q.qid for q in fit).isdisjoint(q.qid for q in validation)
    assert sum(allocation.values()) == 12


def test_official_manifest_builders_freeze_role_sizes() -> None:
    train = tuple(
        EligibleQuestion(
            qid=f"train-{index:04d}",
            level=("easy", "medium", "hard")[index % 3],
            pair_count=3 if index < 1110 else 2,
        )
        for index in range(2286)
    )
    # 1110*3 + 1176*2 = 5682.
    manifest = build_train_manifest(
        train,
        source_path="/private/train.json",
        source_sha256="a" * 64,
        population_audit_path="reports/private-audit.json",
        population_audit_sha256="b" * 64,
    )
    assert manifest["fit"]["question_count"] == 1829
    assert manifest["validation"]["question_count"] == 457
    assert set(manifest["fit"]["ids"]).isdisjoint(
        manifest["validation"]["ids"]
    )
    assert manifest["fit"]["pair_count"] + manifest["validation"][
        "pair_count"
    ] == 5682

    dev = tuple(
        EligibleQuestion(
            qid=f"dev-{index:03d}",
            level="hard",
            pair_count=3 if index < 50 else 2,
        )
        for index in range(162)
    )
    # 50*3 + 112*2 = 374.
    dev_manifest = build_dev_manifest(
        dev,
        source_path="/private/dev.json",
        source_sha256="c" * 64,
        population_audit_path="reports/private-audit.json",
        population_audit_sha256="b" * 64,
    )
    assert dev_manifest["evaluation"]["question_count"] == 162
    assert dev_manifest["evaluation"]["pair_count"] == 374
    assert dev_manifest["protocol"]["model_selection_allowed"] is False


def test_official_manifest_rejects_changed_population() -> None:
    with pytest.raises(
        AllObservedSplitInvariantError,
        match="eligible question count changed",
    ):
        build_train_manifest(
            (EligibleQuestion("q", "hard", 5682),),
            source_path="/private/train.json",
            source_sha256="a" * 64,
            population_audit_path="reports/audit.json",
            population_audit_sha256="b" * 64,
        )


def test_split_config_hashes_all_frozen_artifacts() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["train_inner_split"] == {
        "allocation": "Hamilton proportional allocation across joint strata",
        "fit_question_count": 1829,
        "rank": (
            "ascending SHA256(seed NUL level NUL pair-count bucket NUL "
            "question_id), then ID"
        ),
        "seed": 20260719,
        "stratification": (
            "HotpotQA level x pair-count bucket: 1, 2, 3-4, 5+"
        ),
        "validation_question_count": 457,
    }
    population = config["inputs"]["population_audit"]
    assert hashlib.sha256((ROOT / population["path"]).read_bytes()).hexdigest() == (
        population["sha256"]
    )
    for artifact in config["artifacts"].values():
        digest = hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest()
        assert digest == artifact["sha256"]
