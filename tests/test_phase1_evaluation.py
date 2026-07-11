from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import re
from pathlib import Path

import numpy as np
import pytest

from topocf_rag.baselines import BM25Config
from topocf_rag.evaluation import (
    EvaluationInvariantError,
    PreparedSplit,
    ScoreBundle,
    atomic_json_dump,
    build_score_cache,
    cache_config,
    evaluate_prepared_split,
    load_score_cache,
    prepare_manifest_pairs,
    score_prepared_split,
    sha256_file,
    stream_selected_source_examples,
)
from topocf_rag.pairs import build_pair_manifest, generate_question_pairs
from scripts.evaluate_phase1_baselines import fingerprint_model


class FakeHFTokenizer:
    pattern = re.compile(r"[a-z0-9_]+|->|[^\s]", re.IGNORECASE)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        return_attention_mask: bool,
        return_token_type_ids: bool,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        tokens = ["cls", *self.pattern.findall(text), "sep"]
        if truncation:
            assert max_length is not None
            tokens = tokens[:max_length]
        return {"input_ids": list(range(len(tokens)))}


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(tuple(texts))
        rows = []
        for text in texts:
            rows.append(
                [
                    1.0,
                    float(len(text) % 17 + 1),
                    float(text.count("d0") + 1),
                    float(text.count("title_mention") + 1),
                ]
            )
        return np.asarray(rows, dtype=np.float32)


def example(qid: str) -> dict[str, object]:
    return {
        "_id": qid,
        "question": f"PRIVATE QUESTION BODY {qid}",
        "answer": "PRIVATE ANSWER BODY",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": [["Alpha", 0], ["Beta", 0]],
        "context": [
            ["Alpha", ["PRIVATE GOLD FACT.", "This refers to Beta."]],
            ["Beta", ["PRIVATE SECOND FACT."]],
            ["Gamma", ["This refers to Delta."]],
            ["Delta", ["PRIVATE CONTROL FACT."]],
        ],
    }


def all_observed_example(qid: str) -> dict[str, object]:
    value = example(qid)
    value["context"] = [
        [
            "Alpha",
            ["PRIVATE GOLD FACT refers to Delta.", "This refers to Beta."],
        ],
        ["Beta", ["PRIVATE SECOND FACT."]],
        ["Gamma", ["This refers to Delta and Beta."]],
        ["Delta", ["PRIVATE CONTROL FACT."]],
    ]
    return value


def pairs_for(qid: str):
    return generate_question_pairs(example(qid), [0.9, 0.8, 0.7, 0.6])


def manifest_for(*qids: str, only_one_for_last: bool = False) -> dict[str, object]:
    pairs = []
    for index, qid in enumerate(qids):
        generated = list(pairs_for(qid))
        if only_one_for_last and index == len(qids) - 1:
            generated = generated[:1]
        pairs.extend(generated)
    digest = "0" * 64
    manifest = build_pair_manifest(
        pairs,
        official_split="train",
        source_sha256=digest,
        ids_sha256=digest,
        retrieval_cache_sha256=digest,
        retrieval_top_k=10,
    )
    manifest["frozen_question_count"] = len(qids)
    return manifest


def prepared_for(*qids: str, only_one_for_last: bool = False):
    return prepare_manifest_pairs(
        manifest_for(*qids, only_one_for_last=only_one_for_last),
        {qid: example(qid) for qid in qids},
        FakeHFTokenizer(),
        max_length=4096,
    )


def test_manifest_reconstruction_uses_alias_mapping_and_selected_sentences() -> None:
    manifest = manifest_for("q1")
    prepared = prepare_manifest_pairs(
        manifest, {"q1": example("q1")}, FakeHFTokenizer(), max_length=4096
    )
    record = manifest["pairs"][0]
    pair = prepared.pairs[0]

    assert len(pair.positive.topology.documents) == 4
    for document in pair.positive.topology.documents:
        context_index = record["neutral_alias_mapping"][document.alias]
        source_title, source_sentences = example("q1")["context"][context_index]
        selected = record["sentence_indices"][document.alias]
        assert document.title == source_title
        assert document.sentences == tuple(source_sentences[index] for index in selected)
    assert "PRIVATE QUESTION BODY" not in pair.positive.serialization.text
    assert pair.positive.token_stats.truncated is False


def test_observed_title_mention_requires_selected_source_evidence() -> None:
    manifest = deepcopy(manifest_for("q1"))
    t1 = next(record for record in manifest["pairs"] if record["variant"] == "t1")
    reversed_edge = next(
        edge
        for edge in t1["negative"]["typed_edges"]
        if edge["relation"] == "title_mention" and edge["observed"] is False
    )
    reversed_edge["observed"] = True

    with pytest.raises(EvaluationInvariantError, match="lacks selected sentence evidence"):
        prepare_manifest_pairs(
            manifest,
            {"q1": example("q1")},
            FakeHFTokenizer(),
            max_length=4096,
        )


def test_source_loader_streams_only_manifest_qids(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps([example("unused"), example("selected"), example("later")]),
        encoding="utf-8",
    )
    selected = stream_selected_source_examples(source, ["selected"])
    assert set(selected) == {"selected"}


def test_full_question_bm25_universe_ties_and_dense_encoding_is_deduplicated() -> None:
    prepared = prepared_for("q1")
    embedder = FakeEmbedder()
    scores = score_prepared_split(prepared, embedder)
    report = evaluate_prepared_split(prepared, scores)

    for pair in prepared.pairs:
        positive = scores.bm25_scores[(pair.qid, pair.positive.serialization.sha256)]
        negative = scores.bm25_scores[(pair.qid, pair.negative.serialization.sha256)]
        assert positive == negative
    bm25 = report["overall"]["baselines"]["bm25"]
    assert bm25["pairwise_accuracy"]["value"] == 0.5
    assert bm25["pairwise_accuracy"]["tie_count"] == len(prepared.pairs)
    assert scores.bm25_fit_question_count == 1
    assert scores.dense_question_embedding_count == 1
    assert len(embedder.calls) == 2
    assert len(embedder.calls[0]) == 1
    assert len(embedder.calls[1]) == scores.dense_candidate_embedding_count


def test_combined_t2_group_deduplicates_both_source_strata() -> None:
    manifest = deepcopy(manifest_for("q1"))
    common_t2 = next(
        record
        for record in manifest["pairs"]
        if record["stratum"] == "synthetic_common" and record["variant"] == "t2"
    )
    broad_duplicate = deepcopy(common_t2)
    broad_duplicate["pair_id"] = "pair-derived-t2-duplicate"
    broad_duplicate["stratum"] = "t2_all_observed"
    manifest["pairs"].append(broad_duplicate)
    prepared = prepare_manifest_pairs(
        manifest, {"q1": example("q1")}, FakeHFTokenizer(), max_length=4096
    )
    scores = score_prepared_split(prepared, FakeEmbedder())
    report = evaluate_prepared_split(prepared, scores)

    assert report["by_stratum_variant"]["synthetic_common"]["t2"]["counts"][
        "pair_count"
    ] == 1
    assert report["by_stratum_variant"]["t2_all_observed"]["t2"]["counts"][
        "pair_count"
    ] == 1
    combined = report["derived_groups"]["t2_all_observed_combined"]
    assert combined["counts"]["pair_count"] == 1
    assert combined["baselines"]["bm25"]["pairwise_accuracy"]["value"] == 0.5
    assert combined["baselines"]["bm25"]["expected_recall_at_1"]["value"] == 0.5
    assert all(
        "pairwise_accuracy" in group["baselines"]["bm25"]
        for group in report["by_canonical_class"].values()
    )
    assert all(
        "expected_mrr" in group["baselines"]["bge_m3_dense_cosine"]
        for group in report["by_edge_reality"].values()
    )


def test_combined_t2_rejects_conflicting_duplicate_metadata() -> None:
    manifest = deepcopy(manifest_for("q1"))
    common_t2 = next(record for record in manifest["pairs"] if record["variant"] == "t2")
    duplicate = deepcopy(common_t2)
    duplicate["pair_id"] = "pair-derived-t2-conflict"
    duplicate["stratum"] = "t2_all_observed"
    manifest["pairs"].append(duplicate)
    prepared = prepare_manifest_pairs(
        manifest, {"q1": example("q1")}, FakeHFTokenizer(), max_length=4096
    )
    t2_indices = [
        index for index, pair in enumerate(prepared.pairs) if pair.variant == "t2"
    ]
    assert len(t2_indices) == 2
    changed = list(prepared.pairs)
    changed[t2_indices[-1]] = replace(changed[t2_indices[-1]], base_id="base-conflict")
    conflicting = PreparedSplit(
        official_split=prepared.official_split,
        frozen_question_count=prepared.frozen_question_count,
        questions=prepared.questions,
        pairs=tuple(changed),
    )
    scores = score_prepared_split(conflicting, FakeEmbedder())

    with pytest.raises(EvaluationInvariantError, match="conflicting metadata"):
        evaluate_prepared_split(conflicting, scores)


def test_legacy_all_real_names_are_reported_as_all_observed() -> None:
    source = all_observed_example("q-observed")
    generated = generate_question_pairs(source, [0.9, 0.8, 0.7, 0.6])
    digest = "0" * 64
    manifest = build_pair_manifest(
        generated,
        official_split="train",
        source_sha256=digest,
        ids_sha256=digest,
        retrieval_cache_sha256=digest,
        retrieval_top_k=10,
    )
    manifest["frozen_question_count"] = 1
    legacy = deepcopy(manifest)
    for pair in legacy["pairs"]:
        if pair["stratum"] == "all_observed_rewire":
            pair["stratum"] = "natural_rewire"
            pair["variant"] = "t3_all_real"
    prepared = prepare_manifest_pairs(
        legacy,
        {"q-observed": source},
        FakeHFTokenizer(),
        max_length=4096,
    )
    scores = score_prepared_split(prepared, FakeEmbedder())
    report = evaluate_prepared_split(prepared, scores)
    encoded = json.dumps(report, sort_keys=True)

    assert "all_observed_rewire" in report["by_stratum_variant"]
    assert "t3_all_observed" in report["by_stratum_variant"]["all_observed_rewire"]
    assert "natural_rewire" not in encoded
    assert "t3_all_real" not in encoded


def test_evaluation_pairwise_accuracy_is_question_macro() -> None:
    prepared = prepared_for("many", "one", only_one_for_last=True)
    all_keys = {
        (pair.qid, candidate.serialization.sha256)
        for pair in prepared.pairs
        for candidate in (pair.positive, pair.negative)
    }
    values = {key: 0.0 for key in all_keys}
    for pair in prepared.pairs:
        if pair.qid == "many":
            values[(pair.qid, pair.positive.serialization.sha256)] = 1.0
            values[(pair.qid, pair.negative.serialization.sha256)] = 0.0
        else:
            values[(pair.qid, pair.positive.serialization.sha256)] = 0.0
            values[(pair.qid, pair.negative.serialization.sha256)] = 1.0
    scores = ScoreBundle(values, values, 2, 2, len(all_keys), 0.0)
    report = evaluate_prepared_split(prepared, scores)

    assert len([pair for pair in prepared.pairs if pair.qid == "many"]) > 1
    assert report["overall"]["baselines"]["bm25"]["pairwise_accuracy"][
        "value"
    ] == pytest.approx(0.5)


def test_cache_schema_and_reports_never_persist_source_text(tmp_path: Path) -> None:
    prepared = prepared_for("q1")
    scores = score_prepared_split(prepared, FakeEmbedder())
    digest = "a" * 64
    config = cache_config(
        manifest_sha256=digest,
        source_sha256=digest,
        model_fingerprint_sha256=digest,
        tokenizer_config_sha256=digest,
        max_length=4096,
        bm25_config=BM25Config(),
    )
    cache = build_score_cache(prepared, scores, config)
    report = evaluate_prepared_split(prepared, scores)
    encoded = json.dumps({"cache": cache, "report": report}, sort_keys=True)

    for forbidden in (
        "PRIVATE QUESTION BODY",
        "PRIVATE ANSWER BODY",
        "PRIVATE GOLD FACT",
        "PRIVATE SECOND FACT",
        "PRIVATE CONTROL FACT",
        '"Alpha"',
        '"Beta"',
    ):
        assert forbidden not in encoded
    assert set(cache) == {"schema_version", "config", "counts", "candidates", "pairs"}
    assert set(cache["candidates"][0]) == {
        "qid",
        "candidate_id",
        "serialization_sha256",
        "labels",
        "bm25_score",
        "bge_m3_score",
        "untruncated_token_count",
        "actual_token_count",
        "truncated",
    }

    cache_path = tmp_path / "scores.json"
    atomic_json_dump(cache, cache_path)
    loaded = load_score_cache(cache_path, prepared, config)
    assert loaded is not None
    assert loaded.cache_hit is True
    assert loaded.bm25_scores == scores.bm25_scores
    assert loaded.dense_scores == scores.dense_scores


def test_truncation_fails_by_default() -> None:
    with pytest.raises(EvaluationInvariantError, match="truncation is disabled"):
        prepare_manifest_pairs(
            manifest_for("q1"),
            {"q1": example("q1")},
            FakeHFTokenizer(),
            max_length=8,
        )


def test_report_carries_frozen_and_pair_eligible_question_denominators() -> None:
    manifest = manifest_for("q1")
    manifest["frozen_question_count"] = 10
    prepared = prepare_manifest_pairs(
        manifest, {"q1": example("q1")}, FakeHFTokenizer(), max_length=4096
    )
    report = evaluate_prepared_split(
        prepared, score_prepared_split(prepared, FakeEmbedder())
    )

    assert report["question_coverage"] == {
        "frozen_question_count": 10,
        "pair_eligible_question_count": 1,
        "excluded_no_pair_question_count": 9,
        "pair_eligible_rate": pytest.approx(0.1),
    }


def test_model_fingerprint_changes_when_any_weight_changes(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "pytorch_model.bin").write_bytes(b"dense-v1")
    (model / "sparse_linear.pt").write_bytes(b"sparse-v1")
    (model / "colbert_linear.pt").write_bytes(b"colbert-v1")
    first = fingerprint_model(model)

    (model / "sparse_linear.pt").write_bytes(b"sparse-v2")
    second = fingerprint_model(model)
    (model / "pytorch_model.bin").write_bytes(b"dense-v2")
    third = fingerprint_model(model)

    assert len({first, second, third}) == 3


def test_source_digest_helper_is_streaming_compatible(tmp_path: Path) -> None:
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"phase-one")
    assert sha256_file(path) == __import__("hashlib").sha256(b"phase-one").hexdigest()
