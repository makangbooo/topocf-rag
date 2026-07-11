from __future__ import annotations

import json
import hashlib

from topocf_rag.graph import build_query_document_graph
from topocf_rag.pairs import (
    build_pair_manifest,
    generate_question_pairs,
    summarize_pairs,
)
from topocf_rag.title_normalization import contains_title_mention


def common_example() -> dict[str, object]:
    return {
        "_id": "fixture-qid",
        "question": "Private fixture question text?",
        "answer": "Private fixture answer",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": [["Alpha", 0], ["Beta", 0]],
        "context": [
            ["Alpha", ["Private gold fact.", "This refers to Beta."]],
            ["Beta", ["Private second gold fact."]],
            ["Gamma", ["This refers to Delta.", "This also refers to Delta."]],
            ["Delta", ["Private control target text."]],
        ],
    }


def all_observed_example() -> dict[str, object]:
    example = common_example()
    example["context"] = [
        [
            "Alpha",
            [
                "Private gold fact.",
                "This refers to Beta.",
                "This cross-refers to Delta.",
            ],
        ],
        ["Beta", ["Private second gold fact."]],
        ["Gamma", ["This refers to Delta.", "This cross-refers to Beta."]],
        ["Delta", ["Private control target text."]],
    ]
    return example


def edge_tuples(record: dict[str, object], member: str) -> set[tuple[object, ...]]:
    topology = record[member]
    assert isinstance(topology, dict)
    edges = topology["typed_edges"]
    assert isinstance(edges, list)
    return {
        (edge["relation"], edge["source"], edge["target"], edge["observed"])
        for edge in edges
    }


def generated_common_records() -> list[dict[str, object]]:
    pairs = generate_question_pairs(common_example(), [0.9, 0.8, 0.7, 0.6])
    return [pair.to_manifest_record() for pair in pairs]


def test_common_base_emits_five_matched_variants_without_duplicate_broad_t2() -> None:
    records = generated_common_records()
    common = [record for record in records if record["stratum"] == "synthetic_common"]
    broad = [record for record in records if record["stratum"] == "t2_all_observed"]

    assert {record["variant"] for record in common} == {
        "t1",
        "t2",
        "t3",
        "fork",
        "double_collider",
    }
    assert len(common) == 5
    assert broad == []
    assert len({record["base_id"] for record in records}) == 1
    assert len({record["pair_id"] for record in records}) == 5


def test_every_pair_has_fixed_four_document_four_edge_schema() -> None:
    records = generated_common_records()
    reference = (
        records[0]["context_indices"],
        records[0]["neutral_alias_mapping"],
        records[0]["sentence_indices"],
    )
    for record in records:
        assert len(record["context_indices"]) == 4
        assert set(record["neutral_alias_mapping"]) == {"d0", "d1", "d2", "d3"}
        assert (
            record["context_indices"],
            record["neutral_alias_mapping"],
            record["sentence_indices"],
        ) == reference
        for member in ("positive", "negative"):
            edges = edge_tuples(record, member)
            assert len(edges) == 4
            assert sum(edge[0] == "retrieval" for edge in edges) == 2
            assert sum(edge[0] == "title_mention" for edge in edges) == 2


def test_t2_is_fully_observed_and_synthetic_edges_are_absent() -> None:
    example = common_example()
    graph = build_query_document_graph(example, [0.9, 0.8, 0.7, 0.6])
    observed_mentions = {
        (edge.source_index, edge.target_index) for edge in graph.mention_edges
    }
    retrieved = {edge.document_index for edge in graph.retrieval_edges}

    for record in generated_common_records():
        alias_to_index = record["neutral_alias_mapping"]
        assert isinstance(alias_to_index, dict)
        negative_edges = edge_tuples(record, "negative")
        if record["variant"] == "t2":
            assert all(edge[3] is True for edge in negative_edges)
        for relation, source, target, observed in negative_edges:
            if relation == "retrieval":
                actual = alias_to_index[target] in retrieved
            else:
                actual = (
                    alias_to_index[source], alias_to_index[target]
                ) in observed_mentions
            assert observed is actual


def test_sentence_selection_is_label_blind_union_with_fallback() -> None:
    records = generated_common_records()
    mapping = records[0]["neutral_alias_mapping"]
    selected = records[0]["sentence_indices"]
    by_context = {mapping[alias]: indices for alias, indices in selected.items()}

    assert by_context == {0: [0, 1], 1: [0], 2: [0, 1], 3: [0]}
    assert all(record["sentence_indices"] == selected for record in records)


def test_all_observed_t3_is_separate_and_all_edges_have_shared_evidence() -> None:
    pairs = generate_question_pairs(
        all_observed_example(), [0.9, 0.8, 0.7, 0.6]
    )
    records = [pair.to_manifest_record() for pair in pairs]
    rewires = [
        record for record in records
        if record["stratum"] == "all_observed_rewire"
    ]
    common = [record for record in records if record["stratum"] == "synthetic_common"]

    assert len(rewires) == 1
    assert rewires[0]["variant"] == "t3_all_observed"
    assert all(edge[3] is True for edge in edge_tuples(rewires[0], "negative"))
    assert common == []
    mapping = rewires[0]["neutral_alias_mapping"]
    selected = rewires[0]["sentence_indices"]
    by_context = {mapping[alias]: indices for alias, indices in selected.items()}
    assert by_context[0] == [0, 1, 2]
    assert by_context[2] == [0, 1]

    broad = [pair for pair in pairs if pair.stratum == "t2_all_observed"]
    assert len(broad) == 1
    assert broad[0].base_id != next(
        pair.base_id for pair in pairs if pair.stratum == "all_observed_rewire"
    )
    assert broad[0].neutral_alias_mapping == next(
        pair.neutral_alias_mapping
        for pair in pairs
        if pair.stratum == "all_observed_rewire"
    )
    assert broad[0].sentence_indices != next(
        pair.sentence_indices
        for pair in pairs
        if pair.stratum == "all_observed_rewire"
    )


def test_reverse_control_excludes_entire_common_generator_universe() -> None:
    example = common_example()
    example["context"][3][1][0] = "Private target refers to Gamma."
    pairs = generate_question_pairs(example, [0.9, 0.8, 0.7, 0.6])
    assert not any(pair.stratum == "synthetic_common" for pair in pairs)
    assert sum(pair.stratum == "t2_all_observed" for pair in pairs) == 2


def test_required_retrieval_anchors_filter_endpoint_operations() -> None:
    pairs = generate_question_pairs(
        common_example(),
        [0.9, 0.1, 0.8, 0.7],
        retrieval_top_k=3,
    )
    assert pairs == ()


def test_generation_and_manifest_are_deterministic_and_content_free() -> None:
    first = generate_question_pairs(common_example(), [0.9, 0.8, 0.7, 0.6])
    second = generate_question_pairs(common_example(), [0.9, 0.8, 0.7, 0.6])
    assert first == second

    digest = "0" * 64
    manifest = build_pair_manifest(
        first,
        official_split="train",
        source_sha256=digest,
        ids_sha256=digest,
        retrieval_cache_sha256=digest,
        retrieval_top_k=10,
    )
    encoded = json.dumps(manifest, sort_keys=True)
    assert json.dumps(manifest, sort_keys=True) == json.dumps(
        build_pair_manifest(
            second,
            official_split="train",
            source_sha256=digest,
            ids_sha256=digest,
            retrieval_cache_sha256=digest,
            retrieval_top_k=10,
        ),
        sort_keys=True,
    )
    assert "fixture-qid" in encoded  # qid is explicitly allowed.
    for forbidden in (
        "Private fixture question text",
        "Private fixture answer",
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "Private gold fact",
        "Private control target text",
    ):
        assert forbidden not in encoded


def test_every_observed_mention_has_selected_source_sentence_evidence() -> None:
    for example in (common_example(), all_observed_example()):
        pairs = generate_question_pairs(example, [0.9, 0.8, 0.7, 0.6])
        for pair in pairs:
            record = pair.to_manifest_record()
            mapping = record["neutral_alias_mapping"]
            selected = record["sentence_indices"]
            for member in ("positive", "negative"):
                topology = record[member]
                for edge in topology["typed_edges"]:
                    if edge["relation"] != "title_mention" or not edge["observed"]:
                        continue
                    source_index = mapping[edge["source"]]
                    target_index = mapping[edge["target"]]
                    target_title = example["context"][target_index][0]
                    source_sentences = example["context"][source_index][1]
                    assert any(
                        contains_title_mention(source_sentences[index], target_title)
                        for index in selected[edge["source"]]
                    )


def test_no_duplicate_pair_payload_crosses_strata() -> None:
    records = generated_common_records()
    records.extend(
        pair.to_manifest_record()
        for pair in generate_question_pairs(
            all_observed_example(), [0.9, 0.8, 0.7, 0.6]
        )
    )
    fingerprints: set[tuple[object, ...]] = set()
    for record in records:
        shared = {
            "context_indices": record["context_indices"],
            "sentence_indices": record["sentence_indices"],
            "neutral_alias_mapping": record["neutral_alias_mapping"],
        }

        def payload_sha(member: str) -> str:
            payload = {**shared, "topology": record[member]}
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode("utf-8")
            ).hexdigest()

        fingerprint = (
            record["qid"],
            payload_sha("positive"),
            payload_sha("negative"),
            record["variant"],
        )
        assert fingerprint not in fingerprints
        fingerprints.add(fingerprint)


def test_combined_t2_counts_common_plus_only_broad_extra() -> None:
    common_pairs = generate_question_pairs(
        common_example(), [0.9, 0.8, 0.7, 0.6]
    )
    common_counts = summarize_pairs(common_pairs)["combined_t2"]
    assert common_counts["pair_count"] == 1
    assert common_counts["source_pair_counts"] == {
        "synthetic_common": 1,
        "t2_all_observed_extra": 0,
    }

    observed_pairs = generate_question_pairs(
        all_observed_example(), [0.9, 0.8, 0.7, 0.6]
    )
    observed_counts = summarize_pairs(observed_pairs)["combined_t2"]
    assert observed_counts["pair_count"] == 1
    assert observed_counts["source_pair_counts"] == {
        "synthetic_common": 0,
        "t2_all_observed_extra": 1,
    }


def test_seed_is_part_of_stable_base_and_pair_identity() -> None:
    first = generate_question_pairs(
        common_example(), [0.9, 0.8, 0.7, 0.6], seed=20260712
    )
    changed_seed = generate_question_pairs(
        common_example(), [0.9, 0.8, 0.7, 0.6], seed=20260713
    )

    assert {pair.base_id for pair in first}.isdisjoint(
        pair.base_id for pair in changed_seed
    )
    assert {pair.pair_id for pair in first}.isdisjoint(
        pair.pair_id for pair in changed_seed
    )
