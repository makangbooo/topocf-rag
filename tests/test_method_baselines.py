from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from topocf_rag.method_baselines import (
    MethodBaselineInvariantError,
    aggregate_candidate_logits,
    aggregate_orbit_scores,
    baseline_metric_report,
    build_baseline_evaluation_pairs,
    build_baseline_training_pairs,
    candidate_inputs,
    protocol_sha256,
    render_local_edge,
    reranker_token_ids,
    resolve_frozen_baseline_checkpoint,
    resolve_replication_epoch,
)
from topocf_rag.method_data import (
    build_evaluation_examples,
    build_training_examples,
)
from topocf_rag.metrics import ScoredPair
from topocf_rag.serialization import serialize_evidence_topology
from topocf_rag.topology import EvidenceDocument, EvidenceTopology, TypedEdge


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = ROOT / "configs/certificate_v1/topocf_text_baselines_v1.json"


def _edge(relation: str, source: str, target: str, observed: bool) -> TypedEdge:
    return TypedEdge(relation, source, target, observed, ("PRIVATE PROVENANCE",))


def _topology(*, switched: bool = False) -> EvidenceTopology:
    documents = tuple(
        EvidenceDocument(
            alias=f"d{index}",
            title=f"PRIVATE TITLE {index}",
            sentences=(f"PRIVATE SENTENCE {index}",),
        )
        for index in range(4)
    )
    mentions = (
        (
            _edge("title_mention", "d0", "d3", False),
            _edge("title_mention", "d2", "d1", False),
        )
        if switched
        else (
            _edge("title_mention", "d0", "d1", True),
            _edge("title_mention", "d2", "d3", True),
        )
    )
    return EvidenceTopology(
        question_id="fixture-qid",
        question="PRIVATE QUESTION",
        documents=documents,
        edges=tuple(
            sorted(
                (
                    _edge("retrieval", "q", "d0", True),
                    _edge("retrieval", "q", "d2", True),
                    *mentions,
                ),
                key=lambda edge: edge.structural_key,
            )
        ),
    )


def _prepared() -> SimpleNamespace:
    positive = _topology()
    negative = _topology(switched=True)
    return SimpleNamespace(
        official_split="train",
        pairs=(
            SimpleNamespace(
                pair_id="pair-fixture",
                qid="fixture-qid",
                stratum="synthetic_common",
                variant="t3",
                positive=SimpleNamespace(
                    topology=positive,
                    serialization=serialize_evidence_topology(positive),
                ),
                negative=SimpleNamespace(
                    topology=negative,
                    serialization=serialize_evidence_topology(negative),
                ),
            ),
        ),
    )


def _split() -> dict[str, object]:
    return {
        "fit": {"ids": ["fixture-qid"]},
        "validation": {"ids": ["fixture-qid"]},
    }


def test_flat_input_sees_joint_topology_without_hidden_metadata() -> None:
    topology = _topology()
    rendered = candidate_inputs(
        topology.question,
        topology,
        serialize_evidence_topology(topology),
        baseline="flat_cross_encoder",
    )
    assert len(rendered.inputs) == 1
    text = rendered.inputs[0]
    assert "complete, correctly connected multi-hop evidence chain" in text
    assert "PRIVATE QUESTION" in text
    assert text.count("title_mention") == 2
    assert "observed" not in text
    assert "PRIVATE PROVENANCE" not in text


def test_independent_edge_never_sees_other_candidate_edges() -> None:
    topology = _topology()
    rendered = candidate_inputs(
        topology.question,
        topology,
        serialize_evidence_topology(topology),
        baseline="independent_edge",
    )
    assert len(rendered.inputs) == 4
    assert all("single directed typed edge" in text for text in rendered.inputs)
    assert all(text.count("[typed_edge]") == 1 for text in rendered.inputs)
    assert all(text.count("title_mention") <= 1 for text in rendered.inputs)
    assert all("observed" not in text for text in rendered.inputs)
    retrieval = next(edge for edge in topology.edges if edge.relation == "retrieval")
    retrieval_text = render_local_edge(topology, retrieval)
    assert "[source_document]" not in retrieval_text
    assert "[target_document]" in retrieval_text


@pytest.mark.parametrize(
    ("baseline", "views"),
    (("flat_cross_encoder", 1), ("independent_edge", 4)),
)
def test_training_and_exact_evaluation_views(baseline: str, views: int) -> None:
    training = build_baseline_training_pairs(
        build_training_examples(_prepared(), _split(), epoch=0),
        baseline=baseline,
    )
    evaluation = build_baseline_evaluation_pairs(
        build_evaluation_examples(_prepared(), _split()),
        baseline=baseline,
    )
    assert len(training) == len(evaluation) == 1
    assert len(training[0].positive.inputs) == views
    assert len(training[0].negative.inputs) == views
    assert len(evaluation[0].positive_orbit) == 24
    assert len(evaluation[0].negative_orbit) == 24
    assert all(len(candidate.inputs) == views for candidate in evaluation[0].positive_orbit)


def test_aggregations_are_exact_and_reject_wrong_orbit_size() -> None:
    assert aggregate_candidate_logits([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert aggregate_orbit_scores([float(value) for value in range(24)]) == 11.5
    with pytest.raises(MethodBaselineInvariantError, match="24"):
        aggregate_orbit_scores([1.0] * 23)


def test_metric_report_is_question_macro_and_bootstrap_is_deterministic() -> None:
    pairs = (
        ScoredPair("q1", "p1", 2.0, 1.0),
        ScoredPair("q1", "p2", 0.0, 1.0),
        ScoredPair("q2", "p3", 1.0, 1.0),
    )
    first = baseline_metric_report(
        pairs,
        expected_question_ids=("q1", "q2"),
        bootstrap_repetitions=100,
        bootstrap_seed=9,
    )
    second = baseline_metric_report(
        pairs,
        expected_question_ids=("q1", "q2"),
        bootstrap_repetitions=100,
        bootstrap_seed=9,
    )
    assert first == second
    assert first["pairwise_accuracy"]["value"] == 0.5
    assert first["counts"] == {"question_count": 2, "pair_count": 3}
    assert len(protocol_sha256()) == 64


def test_rejects_unknown_baseline_and_mismatched_serialization() -> None:
    topology = _topology()
    with pytest.raises(MethodBaselineInvariantError, match="baseline"):
        candidate_inputs(
            topology.question,
            topology,
            serialize_evidence_topology(topology),
            baseline="unknown",
        )
    with pytest.raises(MethodBaselineInvariantError, match="disagree"):
        candidate_inputs(
            topology.question,
            topology,
            serialize_evidence_topology(_topology(switched=True)),
            baseline="flat_cross_encoder",
        )


def test_reranker_token_budget_fails_instead_of_truncating() -> None:
    class Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return list(range(len(text.split())))

        def __call__(self, text: str, **options: object) -> dict[str, list[int]]:
            assert options["truncation"] is False
            return {"input_ids": list(range(len(text.split())))}

    tokenizer = Tokenizer()
    token_ids = reranker_token_ids(tokenizer, "short input", max_length=100)
    assert token_ids
    with pytest.raises(MethodBaselineInvariantError, match="truncation is forbidden"):
        reranker_token_ids(tokenizer, "short input", max_length=1)


def test_baseline_config_freezes_prompt_and_never_uses_official_dev() -> None:
    config = json.loads(BASELINE_CONFIG.read_text(encoding="utf-8"))
    assert config["reranker_input_protocol_sha256"] == protocol_sha256()
    assert config["data"]["official_dev_used"] is False
    assert config["model"]["expected_fingerprint_sha256"] == (
        "01f807839563e5e18293e9498f59e5e025ecd134fcf8a1cd2076e840faf8b4fb"
    )
    assert config["model"]["audited_files"]["model.safetensors"] == {
        "sha256": "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b",
        "size_bytes": 1191588280,
    }
    assert set(config["baselines"]) == {
        "flat_cross_encoder",
        "independent_edge",
    }
    assert config["official_dev_evaluation"]["mandatory_baseline"] == (
        "independent_edge"
    )
    assert config["official_dev_evaluation"]["mandatory_seeds"] == [
        20260718,
        20260719,
        20260720,
    ]
    for artifact in config["artifacts"].values():
        path = ROOT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_selected_configurations_lock_replication_learning_rate_and_epoch() -> None:
    config = json.loads(BASELINE_CONFIG.read_text(encoding="utf-8"))
    assert resolve_replication_epoch(
        config,
        baseline="flat_cross_encoder",
        seed=20260719,
        learning_rate=0.0001,
        replicate_selected=True,
    ) == 1
    assert resolve_replication_epoch(
        config,
        baseline="independent_edge",
        seed=20260720,
        learning_rate=0.0002,
        replicate_selected=True,
    ) == 4
    assert resolve_replication_epoch(
        config,
        baseline="flat_cross_encoder",
        seed=20260718,
        learning_rate=0.0001,
        replicate_selected=False,
    ) is None


def test_replication_rejects_fresh_selection_or_wrong_learning_rate() -> None:
    config = json.loads(BASELINE_CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(MethodBaselineInvariantError, match="require"):
        resolve_replication_epoch(
            config,
            baseline="flat_cross_encoder",
            seed=20260719,
            learning_rate=0.0001,
            replicate_selected=False,
        )
    with pytest.raises(MethodBaselineInvariantError, match="learning_rate=0.0002"):
        resolve_replication_epoch(
            config,
            baseline="independent_edge",
            seed=20260719,
            learning_rate=0.0001,
            replicate_selected=True,
        )
    with pytest.raises(MethodBaselineInvariantError, match="selection seed"):
        resolve_replication_epoch(
            config,
            baseline="independent_edge",
            seed=20260718,
            learning_rate=0.0002,
            replicate_selected=True,
        )


def test_selection_report_is_aggregate_only_and_matches_frozen_choices() -> None:
    path = ROOT / "reports/phase1/text_baseline_selection.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["data"] == {
        "fit_pair_count": 294,
        "fit_question_count": 129,
        "official_dev_used": False,
        "validation_pair_count": 74,
        "validation_question_count": 32,
    }
    assert report["selected_configurations"]["flat_cross_encoder"][
        "learning_rate"
    ] == 0.0001
    assert report["selected_configurations"]["flat_cross_encoder"][
        "selected_epoch"
    ] == 1
    assert report["selected_configurations"]["independent_edge"][
        "learning_rate"
    ] == 0.0002
    assert report["selected_configurations"]["independent_edge"][
        "selected_epoch"
    ] == 4
    assert report["interpretation"]["candidate_method_stop_signal"] is True
    assert report["interpretation"]["formal_method_claim_killed"] is True
    assert report["interpretation"]["official_dev_gate_decision"] == (
        "STOP_CURRENT_METHOD"
    )


def test_official_dev_checkpoint_resolution_is_hash_and_seed_locked() -> None:
    config = json.loads(BASELINE_CONFIG.read_text(encoding="utf-8"))
    root = Path("/private/checkpoints")
    resolved = resolve_frozen_baseline_checkpoint(
        config,
        baseline="independent_edge",
        seed=20260720,
        run_root=root,
    )
    assert resolved.learning_rate == 0.0002
    assert resolved.epoch == 4
    assert resolved.path == (
        root
        / "independent_edge/lr-2e-04/20260720/checkpoint-epoch-4"
    )
    assert resolved.sha256 == (
        "dfb0d37c5fa8d92b12c0521b0cde21c811583d5be58bdec27c833b356f1371aa"
    )
    with pytest.raises(MethodBaselineInvariantError, match="no hash-frozen"):
        resolve_frozen_baseline_checkpoint(
            config,
            baseline="independent_edge",
            seed=7,
            run_root=root,
        )
