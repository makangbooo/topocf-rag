from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from topocf_rag.graph import build_query_document_graph
from topocf_rag.natural_pairs import (
    NaturalPairInvariantError,
    build_natural_pair_manifest,
    select_natural_retrieval_pair,
    summarize_natural_selections,
)
from topocf_rag.serialization import SERIALIZER_VERSION
from topocf_rag.title_normalization import contains_title_mention
from scripts.materialize_natural_path_pairs import fingerprint_tokenizer_files


def natural_example(qid: str = "fixture-natural-qid") -> dict[str, object]:
    return {
        "_id": qid,
        "question": "Private natural retrieval fixture question?",
        "answer": "Private natural retrieval fixture answer",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": [["Alpha Topic", 2], ["Beta Topic", 2]],
        "context": [
            [
                "Alpha Topic",
                [
                    "Private alpha introduction.",
                    "This article links Beta Topic.",
                    "Private alpha supporting-only sentence.",
                ],
            ],
            [
                "Beta Topic",
                [
                    "Private beta introduction.",
                    "Private beta middle sentence.",
                    "Private beta supporting-only sentence.",
                ],
            ],
            [
                "Gamma Topic",
                [
                    "Private gamma introduction.",
                    "This article links Delta Topic.",
                ],
            ],
            ["Delta Topic", ["Private delta introduction."]],
            [
                "Epsilon Topic",
                [
                    "Private epsilon introduction.",
                    "This article also links Delta Topic.",
                ],
            ],
        ],
    }


SCORES = [0.8, 0.6, 0.79, 0.5, 0.5]
TOKENIZER_FINGERPRINT = "1" * 64


class FakeHFTokenizer:
    def __init__(
        self,
        marker_extra_tokens: dict[str, int] | None = None,
        *,
        constant_count: int | None = None,
    ) -> None:
        self.marker_extra_tokens = marker_extra_tokens or {}
        self.constant_count = constant_count

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int | None = None,
        **_options,
    ) -> dict[str, list[int]]:
        count = self.constant_count or len(text.split())
        count += sum(
            extra
            for marker, extra in self.marker_extra_tokens.items()
            if marker in text
        )
        if truncation:
            assert max_length is not None
            count = min(count, max_length)
        return {"input_ids": list(range(count))}


TOKENIZER = FakeHFTokenizer()


def select(
    example,
    scores,
    *,
    tokenizer=TOKENIZER,
    fingerprint=TOKENIZER_FINGERPRINT,
    max_length=1024,
):
    return select_natural_retrieval_pair(
        example,
        scores,
        tokenizer,
        tokenizer_fingerprint_sha256=fingerprint,
        max_length=max_length,
    )


def selected_pair():
    selection = select(natural_example(), SCORES)
    assert selection.pair is not None
    return selection, selection.pair


def test_positive_and_negative_are_real_two_edge_candidate_paths() -> None:
    example = natural_example()
    selection, pair = selected_pair()
    graph = build_query_document_graph(example, SCORES)
    candidate_indices = {
        (path.source_index, path.target_index) for path in graph.candidate_paths
    }
    gold_indices = {
        (path.source_index, path.target_index) for path in graph.gold_paths
    }
    negative_indices = {
        (path.source_index, path.target_index)
        for path in graph.natural_negative_paths
    }

    assert pair.positive.context_indices in gold_indices
    assert pair.negative.context_indices in negative_indices
    assert pair.positive.context_indices in candidate_indices
    assert pair.negative.context_indices in candidate_indices
    assert selection.positive_candidate_count == 1
    assert selection.negative_candidate_count == 2
    for member in (pair.positive, pair.negative):
        payload = member.to_manifest_record()
        assert len(payload["typed_edges"]) == 2
        assert all(edge["observed"] is True for edge in payload["typed_edges"])


def test_preregistered_distance_selects_best_pair_and_records_gaps() -> None:
    selection, pair = selected_pair()
    assert pair.positive.context_indices == (0, 1)
    assert pair.negative.context_indices == (2, 3)
    assert pair.positive.serialized_token_count > 0
    assert pair.negative.serialized_token_count > 0
    expected_token_gap = abs(
        pair.positive.serialized_token_count
        - pair.negative.serialized_token_count
    )
    expected_relative_token_gap = expected_token_gap / (
        pair.positive.serialized_token_count
    )
    assert pair.absolute_token_gap == expected_token_gap
    assert pair.relative_token_gap == pytest.approx(expected_relative_token_gap)
    assert pair.absolute_score_gap == pytest.approx(0.01)

    expected_std = statistics.pstdev([0.8, 0.79, 0.5])
    expected_distance = 0.01 / expected_std + expected_relative_token_gap
    assert pair.candidate_score_std == pytest.approx(expected_std)
    assert pair.selection_distance == pytest.approx(expected_distance)
    assert selection.candidate_combination_count == 2


def test_stable_tie_break_uses_context_indices() -> None:
    tied_scores = [0.8, 0.6, 0.7, 0.5, 0.7]
    selection = select_natural_retrieval_pair(
        natural_example(),
        tied_scores,
        FakeHFTokenizer(constant_count=100),
        tokenizer_fingerprint_sha256=TOKENIZER_FINGERPRINT,
    )
    assert selection.pair is not None
    assert selection.pair.negative.context_indices == (2, 3)


def test_exact_serialized_tokenizer_can_change_selected_negative() -> None:
    tied_scores = [0.8, 0.6, 0.7, 0.5, 0.7]
    gamma_heavy = select(
        natural_example(),
        tied_scores,
        tokenizer=FakeHFTokenizer({"Gamma Topic": 100}),
        fingerprint="2" * 64,
    )
    epsilon_heavy = select(
        natural_example(),
        tied_scores,
        tokenizer=FakeHFTokenizer({"Epsilon Topic": 100}),
        fingerprint="3" * 64,
    )
    assert gamma_heavy.pair is not None
    assert epsilon_heavy.pair is not None
    assert gamma_heavy.pair.negative.context_indices == (4, 3)
    assert epsilon_heavy.pair.negative.context_indices == (2, 3)


def test_sentence_policy_ignores_supporting_fact_sentence_labels() -> None:
    example = natural_example()
    _selection, pair = selected_pair()
    assert pair.positive.sentence_indices == ((0, 1), (0,))
    assert pair.negative.sentence_indices == ((0, 1), (0,))
    # Both gold supporting-fact labels are sentence 2 and are not selected.
    assert 2 not in pair.positive.sentence_indices[0]
    assert 2 not in pair.positive.sentence_indices[1]

    for path in (pair.positive, pair.negative):
        source_index, target_index = path.context_indices
        source_sentences = example["context"][source_index][1]
        target_title = example["context"][target_index][0]
        assert any(
            contains_title_mention(source_sentences[index], target_title)
            for index in path.sentence_indices[0]
        )


def test_at_most_one_pair_per_question_and_deterministic_output() -> None:
    first = select(natural_example(), SCORES)
    second = select(natural_example(), SCORES)
    assert first == second
    assert first.pair is not None

    digest = "0" * 64
    manifest = build_natural_pair_manifest(
        [first],
        official_split="train",
        source_sha256=digest,
        ids_sha256=digest,
        retrieval_cache_sha256=digest,
        tokenizer_fingerprint_sha256=TOKENIZER_FINGERPRINT,
        retrieval_top_k=10,
        max_length=1024,
    )
    assert len(manifest["pairs"]) == 1
    with pytest.raises(NaturalPairInvariantError, match="at most one pair"):
        build_natural_pair_manifest(
            [first, second],
            official_split="train",
            source_sha256=digest,
            ids_sha256=digest,
            retrieval_cache_sha256=digest,
            tokenizer_fingerprint_sha256=TOKENIZER_FINGERPRINT,
            retrieval_top_k=10,
            max_length=1024,
        )


def test_pair_identity_binds_tokenizer_fingerprint_and_max_length_does_not_truncate() -> None:
    roomy = select(natural_example(), SCORES, max_length=4096)
    diagnostic_limit = select(natural_example(), SCORES, max_length=4)
    changed_fingerprint = select(
        natural_example(), SCORES, fingerprint="4" * 64, max_length=4096
    )
    assert roomy.pair is not None
    assert diagnostic_limit.pair is not None
    assert changed_fingerprint.pair is not None
    assert roomy.pair.pair_id == diagnostic_limit.pair.pair_id
    assert roomy.pair.positive.serialized_token_count == (
        diagnostic_limit.pair.positive.serialized_token_count
    )
    assert roomy.pair.negative.serialized_token_count == (
        diagnostic_limit.pair.negative.serialized_token_count
    )
    assert diagnostic_limit.pair.positive.exceeds_max_length is True
    assert diagnostic_limit.pair.negative.exceeds_max_length is True
    assert roomy.pair.pair_id != changed_fingerprint.pair.pair_id


def test_zero_reasons_and_token_match_count_are_reported() -> None:
    no_positive = natural_example("no-positive")
    no_positive["context"][0][1][1] = "Private sentence without cross-title text."
    no_positive_selection = select(no_positive, SCORES)
    assert no_positive_selection.zero_reason == "no_positive_candidate"

    no_negative = natural_example("no-negative")
    no_negative["context"][2][1][1] = "Private gamma sentence."
    no_negative["context"][4][1][1] = "Private epsilon sentence."
    no_negative_selection = select(no_negative, SCORES)
    assert no_negative_selection.zero_reason == "no_negative_candidate"

    eligible = select(natural_example(), SCORES)
    counts = summarize_natural_selections(
        [no_positive_selection, no_negative_selection, eligible]
    )
    assert counts["frozen_question_count"] == 3
    assert counts["eligible_pair_count"] == 1
    assert counts["zero_pair_reason_counts"] == {
        "no_positive_candidate": 1,
        "no_negative_candidate": 1,
        "unusable_sentence_payload": 0,
    }
    assert counts["selected_pairs_with_exact_token_gap_lte_5_percent"] == int(
        eligible.pair.token_within_five_percent
    )


def test_manifest_is_content_free_and_contains_required_real_path_metadata() -> None:
    selection = select(natural_example(), SCORES)
    digest = "0" * 64
    manifest = build_natural_pair_manifest(
        [selection],
        official_split="dev_distractor",
        source_sha256=digest,
        ids_sha256=digest,
        retrieval_cache_sha256=digest,
        tokenizer_fingerprint_sha256=TOKENIZER_FINGERPRINT,
        retrieval_top_k=10,
        max_length=1024,
    )
    encoded = json.dumps(manifest, sort_keys=True)
    assert manifest["stratum"] == "natural_retrieval_error"
    assert "fixture-natural-qid" in encoded  # qid is explicitly allowed.
    record = manifest["pairs"][0]
    for member in ("positive", "negative"):
        assert set(record[member]) == {
            "context_indices",
            "sentence_indices",
            "typed_edges",
            "retrieval_score",
            "retrieval_rank",
            "serialized_token_count",
            "serialization_sha256",
            "exceeds_max_length",
        }
    assert manifest["schema_version"] == 2
    assert manifest["provenance"]["tokenizer_files_sha256"] == TOKENIZER_FINGERPRINT
    assert manifest["provenance"]["serializer_version"] == SERIALIZER_VERSION
    for forbidden in (
        "Private natural retrieval fixture question",
        "Private natural retrieval fixture answer",
        "Alpha Topic",
        "Beta Topic",
        "Gamma Topic",
        "Delta Topic",
        "Epsilon Topic",
        "Private alpha introduction",
        "Private alpha supporting-only sentence",
    ):
        assert forbidden not in encoded


def test_tokenizer_file_fingerprint_is_deterministic_and_content_sensitive(
    tmp_path: Path,
) -> None:
    (tmp_path / "tokenizer_config.json").write_text("first", encoding="utf-8")
    first = fingerprint_tokenizer_files(tmp_path)
    assert first == fingerprint_tokenizer_files(tmp_path)

    (tmp_path / "tokenizer.json").write_text("vocabulary", encoding="utf-8")
    second = fingerprint_tokenizer_files(tmp_path)
    assert second != first
    (tmp_path / "tokenizer_config.json").write_text("changed", encoding="utf-8")
    assert fingerprint_tokenizer_files(tmp_path) != second
