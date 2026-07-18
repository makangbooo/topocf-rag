from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from topocf_rag.method_data import (
    MethodDataInvariantError,
    build_evaluation_examples,
    build_inner_split_manifest,
    build_official_dev_evaluation_examples,
    build_training_examples,
    inner_role_ids,
    load_inner_split_manifest,
)
from topocf_rag.method_data import source_pair_manifest_sha256
from topocf_rag.serialization import (
    deterministic_four_document_permutation,
    serialize_evidence_topology,
)
from topocf_rag.topology import EvidenceDocument, EvidenceTopology, TypedEdge


ROOT = Path(__file__).resolve().parents[1]
PAIR_MANIFEST = ROOT / "data/phase1/hotpot_train_phase1_pairs.json"
INNER_MANIFEST = ROOT / "data/splits/topocf_t3_train_inner_v1.json"
INNER_REPORT = ROOT / "reports/phase1/method_inner_split.json"
DATA_CONFIG = ROOT / "configs/certificate_v1/topocf_data_v1.json"


def _documents() -> tuple[EvidenceDocument, ...]:
    return tuple(
        EvidenceDocument(
            alias=f"d{index}",
            title=f"PRIVATE TITLE {index}",
            sentences=(f"PRIVATE SENTENCE {index}",),
        )
        for index in range(4)
    )


def _edge(relation: str, source: str, target: str, observed: bool) -> TypedEdge:
    return TypedEdge(relation, source, target, observed)


def _topology(*, switched: bool = False) -> EvidenceTopology:
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
    edges = tuple(
        sorted(
            (
                _edge("retrieval", "q", "d0", True),
                _edge("retrieval", "q", "d2", True),
                *mentions,
            ),
            key=lambda edge: edge.structural_key,
        )
    )
    return EvidenceTopology(
        question_id="fixture-qid",
        question="PRIVATE QUESTION",
        documents=_documents(),
        edges=edges,
    )


def _prepared() -> SimpleNamespace:
    positive = _topology()
    negative = _topology(switched=True)
    pair = SimpleNamespace(
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
    )
    return SimpleNamespace(official_split="train", pairs=(pair,))


def _inner_split(
    *, fit: list[str] | None = None, validation: list[str] | None = None
) -> dict[str, object]:
    return {
        "fit": {"ids": sorted(fit or ["fixture-qid"])},
        "validation": {"ids": sorted(validation or ["fixture-qid"])},
    }


def test_frozen_inner_split_is_question_disjoint_and_hash_bound() -> None:
    pair_sha = source_pair_manifest_sha256(PAIR_MANIFEST)
    manifest = load_inner_split_manifest(
        INNER_MANIFEST, expected_pair_manifest_sha256=pair_sha
    )
    fit = inner_role_ids(manifest, "fit")
    validation = inner_role_ids(manifest, "validation")

    assert len(fit) == 129
    assert len(validation) == 32
    assert set(fit).isdisjoint(validation)
    assert manifest["population"] == {
        "bucket_question_counts": {"1": 86, "2": 27, "3-4": 26, "5+": 22},
        "pair_count": 368,
        "question_count": 161,
        "validation_allocation": {"1": 17, "2": 5, "3-4": 5, "5+": 5},
    }
    assert manifest["fit"]["pair_count"] == 294
    assert manifest["validation"]["pair_count"] == 74


def test_inner_split_is_deterministic_under_pair_record_reordering() -> None:
    payload = json.loads(PAIR_MANIFEST.read_text(encoding="utf-8"))
    digest = source_pair_manifest_sha256(PAIR_MANIFEST)
    first = build_inner_split_manifest(payload, pair_manifest_sha256=digest)
    changed_order = deepcopy(payload)
    changed_order["pairs"] = list(reversed(changed_order["pairs"]))
    second = build_inner_split_manifest(
        changed_order, pair_manifest_sha256=digest
    )

    assert first == second


def test_inner_split_rejects_official_dev_and_source_hash_change() -> None:
    payload = json.loads(PAIR_MANIFEST.read_text(encoding="utf-8"))
    digest = source_pair_manifest_sha256(PAIR_MANIFEST)
    payload["official_split"] = "dev_distractor"
    with pytest.raises(MethodDataInvariantError, match="official train"):
        build_inner_split_manifest(payload, pair_manifest_sha256=digest)

    with pytest.raises(MethodDataInvariantError, match="source hash changed"):
        load_inner_split_manifest(
            INNER_MANIFEST, expected_pair_manifest_sha256="0" * 64
        )

    payload["official_split"] = "train"
    with pytest.raises(MethodDataInvariantError, match="size is frozen"):
        build_inner_split_manifest(
            payload,
            pair_manifest_sha256=digest,
            validation_question_count=31,
        )


def test_public_report_contains_no_selected_ids_and_config_hashes_are_exact() -> None:
    inner = json.loads(INNER_MANIFEST.read_text(encoding="utf-8"))
    report_text = INNER_REPORT.read_text(encoding="utf-8")
    for qid in (*inner["fit"]["ids"], *inner["validation"]["ids"]):
        assert qid not in report_text

    config = json.loads(DATA_CONFIG.read_text(encoding="utf-8"))
    for artifact in config["artifacts"].values():
        path = ROOT / artifact["path"]
        assert source_pair_manifest_sha256(path) == artifact["sha256"]


def test_training_pair_uses_one_shared_permutation_and_repair_target() -> None:
    examples = build_training_examples(
        _prepared(), _inner_split(), epoch=3, seed=20260718
    )
    assert len(examples) == 1
    example = examples[0]
    assert example.old_aliases_in_new_order == (
        deterministic_four_document_permutation(
            seed=20260718, epoch=3, item_key="pair-fixture"
        )
    )
    assert sum(map(sum, example.positive_binding_target)) == 2
    assert sum(map(sum, example.negative_binding)) == 2
    assert example.positive_binding_target != example.negative_binding
    assert sum(example.retrieval_root_mask) == 2
    positive_documents = example.positive.text.split("[documents]", 1)[1]
    negative_documents = example.negative.text.split("[documents]", 1)[1]
    assert positive_documents == negative_documents
    assert "PRIVATE QUESTION" not in example.positive.text
    assert "observed" not in example.negative.text


def test_exact_evaluation_orbit_is_synchronized_for_both_members() -> None:
    examples = build_evaluation_examples(_prepared(), _inner_split())
    assert len(examples) == 1
    example = examples[0]
    assert len(example.positive_orbit) == 24
    assert len(example.negative_orbit) == 24
    assert len(set(item.sha256 for item in example.positive_orbit)) == 24
    assert len(set(item.sha256 for item in example.negative_orbit)) == 24
    for positive, negative, target, corrupted in zip(
        example.positive_orbit,
        example.negative_orbit,
        example.positive_binding_targets,
        example.negative_bindings,
        strict=True,
    ):
        assert positive.text.split("[documents]", 1)[1] == negative.text.split(
            "[documents]", 1
        )[1]
        assert target != corrupted


def test_official_dev_evaluation_uses_all_primary_pairs_without_ids() -> None:
    prepared = _prepared()
    prepared.official_split = "dev_distractor"
    examples = build_official_dev_evaluation_examples(prepared)
    assert len(examples) == 1
    assert len(examples[0].positive_orbit) == 24
    assert len(examples[0].negative_orbit) == 24

    prepared.official_split = "train"
    with pytest.raises(MethodDataInvariantError, match="dev_distractor"):
        build_official_dev_evaluation_examples(prepared)


def test_training_data_rejects_split_and_missing_ids() -> None:
    prepared = _prepared()
    prepared.official_split = "dev_distractor"
    with pytest.raises(MethodDataInvariantError, match="official train"):
        build_training_examples(prepared, _inner_split(), epoch=0)

    with pytest.raises(MethodDataInvariantError, match="cover"):
        build_evaluation_examples(
            _prepared(), _inner_split(validation=["missing-qid"])
        )
