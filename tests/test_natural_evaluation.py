from __future__ import annotations

from copy import deepcopy
import json
import re
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_natural_path_baselines import fingerprint_model
from topocf_rag.baselines import BM25Config
from topocf_rag.evaluation import (
    EvaluationInvariantError,
    build_score_cache,
    cache_config,
    score_prepared_split,
)
from topocf_rag.natural_evaluation import (
    evaluate_natural_split,
    prepare_natural_manifest,
    prepare_natural_split_from_files,
)
from topocf_rag.natural_pairs import (
    build_natural_pair_manifest,
    select_natural_retrieval_pair,
)


TOKENIZER_FINGERPRINT = "f" * 64


class FakeHFTokenizer:
    pattern = re.compile(r"[a-z0-9_]+|->|[^\s]", re.IGNORECASE)

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int | None = None,
        **_options,
    ) -> dict[str, list[int]]:
        tokens = ["cls", *self.pattern.findall(text), "sep"]
        if truncation:
            assert max_length is not None
            tokens = tokens[:max_length]
        return {"input_ids": list(range(len(tokens)))}


class FakeEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [
                [1.0, float(len(text) % 19 + 1), float(text.count("d0") + 1)]
                for text in texts
            ],
            dtype=np.float32,
        )


def natural_example(qid: str = "natural-qid") -> dict[str, object]:
    return {
        "_id": qid,
        "question": "PRIVATE NATURAL QUESTION BODY",
        "answer": "PRIVATE NATURAL ANSWER BODY",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": [["Alpha Topic", 2], ["Beta Topic", 2]],
        "context": [
            [
                "Alpha Topic",
                [
                    "PRIVATE ALPHA INTRODUCTION.",
                    "This article links Beta Topic.",
                    "PRIVATE ALPHA SUPPORTING FACT.",
                ],
            ],
            [
                "Beta Topic",
                ["PRIVATE BETA INTRODUCTION.", "Middle.", "PRIVATE BETA FACT."],
            ],
            [
                "Gamma Topic",
                ["PRIVATE GAMMA INTRODUCTION.", "This article links Delta Topic."],
            ],
            ["Delta Topic", ["PRIVATE DELTA INTRODUCTION."]],
        ],
    }


def natural_manifest(qid: str = "natural-qid") -> dict[str, object]:
    selection = select_natural_retrieval_pair(
        natural_example(qid),
        [0.8, 0.6, 0.79, 0.5],
        FakeHFTokenizer(),
        tokenizer_fingerprint_sha256=TOKENIZER_FINGERPRINT,
        max_length=1024,
    )
    assert selection.pair is not None
    digest = "0" * 64
    return build_natural_pair_manifest(
        [selection],
        official_split="train",
        source_sha256=digest,
        ids_sha256=digest,
        retrieval_cache_sha256=digest,
        tokenizer_fingerprint_sha256=TOKENIZER_FINGERPRINT,
        retrieval_top_k=10,
        max_length=1024,
    )


def prepared_natural():
    return prepare_natural_manifest(
        natural_manifest(),
        {"natural-qid": natural_example()},
        FakeHFTokenizer(),
        max_length=1024,
        expected_tokenizer_files_sha256=TOKENIZER_FINGERPRINT,
    )


def test_reconstructs_two_real_paths_and_enforces_gold_definition() -> None:
    natural = prepared_natural()
    pair = natural.prepared.pairs[0]

    assert pair.stratum == "natural_retrieval_error"
    assert pair.variant == "natural"
    assert len(pair.positive.topology.documents) == 2
    assert len(pair.negative.topology.documents) == 2
    assert all(edge.observed for edge in pair.positive.topology.edges)
    assert all(edge.observed for edge in pair.negative.topology.edges)
    assert {document.title for document in pair.positive.topology.documents} == {
        "Alpha Topic",
        "Beta Topic",
    }
    assert {document.title for document in pair.negative.topology.documents} == {
        "Gamma Topic",
        "Delta Topic",
    }


def test_negative_covering_both_gold_titles_is_rejected() -> None:
    manifest = deepcopy(natural_manifest())
    manifest["pairs"][0]["negative"] = deepcopy(
        manifest["pairs"][0]["positive"]
    )

    with pytest.raises(EvaluationInvariantError, match="negative path covers both golds"):
        prepare_natural_manifest(
            manifest,
            {"natural-qid": natural_example()},
            FakeHFTokenizer(),
        )


def test_selected_source_sentence_must_prove_observed_mention() -> None:
    manifest = deepcopy(natural_manifest())
    manifest["pairs"][0]["negative"]["sentence_indices"]["d0"] = [0]

    with pytest.raises(EvaluationInvariantError, match="lacks selected source"):
        prepare_natural_manifest(
            manifest,
            {"natural-qid": natural_example()},
            FakeHFTokenizer(),
        )


def test_metrics_and_both_matching_gap_definitions_are_reported() -> None:
    natural = prepared_natural()
    scores = score_prepared_split(natural.prepared, FakeEmbedder())
    report = evaluate_natural_split(natural, scores)

    assert report["question_coverage"] == {
        "frozen_question_count": 1,
        "pair_eligible_question_count": 1,
        "excluded_no_pair_question_count": 0,
        "pair_eligible_rate": 1.0,
    }
    baseline = report["by_stratum_variant"]["natural_retrieval_error"]["natural"][
        "baselines"
    ]["bm25"]
    assert baseline["pairwise_accuracy"]["eligible_question_count"] == 1
    assert baseline["auroc"]["eligible_question_count"] == 1
    matching = report["matching_diagnostics"]
    assert matching["manifest_selection_matching"]["relative_serialized_token_gap"][
        "count"
    ] == 1
    assert matching["exact_serialization_token_matching"]["relative_token_gap"][
        "count"
    ] == 1
    assert report["selection_coverage"]["eligible_pair_count"] == 1


@pytest.mark.parametrize("field", ["serialized_token_count", "serialization_sha256"])
def test_reconstructed_serialization_must_match_manifest(field: str) -> None:
    manifest = deepcopy(natural_manifest())
    if field == "serialized_token_count":
        manifest["pairs"][0]["positive"][field] += 1
        message = "serialized token count"
    else:
        manifest["pairs"][0]["positive"][field] = "0" * 64
        message = "serialization SHA256"

    with pytest.raises(EvaluationInvariantError, match=message):
        prepare_natural_manifest(
            manifest,
            {"natural-qid": natural_example()},
            FakeHFTokenizer(),
        )


def test_cache_and_report_are_content_free() -> None:
    natural = prepared_natural()
    scores = score_prepared_split(natural.prepared, FakeEmbedder())
    digest = "a" * 64
    config = cache_config(
        manifest_sha256=digest,
        source_sha256=digest,
        model_fingerprint_sha256=digest,
        tokenizer_config_sha256=digest,
        max_length=1024,
        bm25_config=BM25Config(),
    )
    cache = build_score_cache(natural.prepared, scores, config)
    report = evaluate_natural_split(natural, scores)
    encoded = json.dumps({"cache": cache, "report": report}, sort_keys=True)

    for forbidden in (
        "PRIVATE NATURAL QUESTION BODY",
        "PRIVATE NATURAL ANSWER BODY",
        "PRIVATE ALPHA INTRODUCTION",
        "PRIVATE GAMMA INTRODUCTION",
        '"Alpha Topic"',
        '"Delta Topic"',
    ):
        assert forbidden not in encoded


def test_file_loader_checks_source_digest_and_streams_selected_qid(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps([natural_example("unused"), natural_example()]), encoding="utf-8"
    )
    manifest = natural_manifest()
    import hashlib

    manifest["provenance"]["source_sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    natural, hashes = prepare_natural_split_from_files(
        manifest_path,
        source_path,
        FakeHFTokenizer(),
        max_length=1024,
        expected_tokenizer_files_sha256=TOKENIZER_FINGERPRINT,
    )
    assert set(natural.prepared.questions) == {"natural-qid"}
    assert hashes["source_sha256"] == manifest["provenance"]["source_sha256"]


def test_natural_model_fingerprint_includes_weight_content(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "pytorch_model.bin").write_bytes(b"weight-one")
    first = fingerprint_model(model)
    (model / "pytorch_model.bin").write_bytes(b"weight-two")
    assert fingerprint_model(model) != first


def test_natural_truncation_is_a_hard_failure() -> None:
    with pytest.raises(EvaluationInvariantError, match="truncation is disabled"):
        prepare_natural_manifest(
            natural_manifest(),
            {"natural-qid": natural_example()},
            FakeHFTokenizer(),
            max_length=8,
        )
