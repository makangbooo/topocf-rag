from __future__ import annotations

import hashlib
import re

from topocf_rag.serialization import (
    SERIALIZER_VERSION,
    count_serialization_tokens,
    serialize_evidence_topology,
)
from topocf_rag.topology import EvidenceDocument, EvidenceTopology, TypedEdge


class TinyHFTokenizer:
    """A deterministic HF-call-compatible tokenizer for unit tests."""

    _tokens = re.compile(r"[a-z0-9_]+|->|[^\s]", re.IGNORECASE)

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
        assert add_special_tokens is True
        assert padding is False
        assert return_attention_mask is False
        assert return_token_type_ids is False
        tokens = ["[CLS]", *self._tokens.findall(text), "[SEP]"]
        if truncation:
            assert max_length is not None
            tokens = tokens[:max_length]
        return {"input_ids": list(range(len(tokens)))}


def documents(count: int) -> tuple[EvidenceDocument, ...]:
    return tuple(
        EvidenceDocument(
            alias=f"d{index}",
            title=f"Neutral Title {index}",
            sentences=(f"Selected fact {index}a.", f"Selected fact {index}b."),
        )
        for index in range(count)
    )


def topology(
    edges: tuple[TypedEdge, ...],
    *,
    count: int = 2,
    question: str = "QUESTION TEXT MUST NOT APPEAR",
) -> EvidenceTopology:
    return EvidenceTopology(
        question_id="question-id-must-not-appear",
        question=question,
        documents=documents(count),
        edges=edges,
    )


def retrieval(alias: str, *, observed: bool = True) -> TypedEdge:
    return TypedEdge(
        "retrieval",
        "q",
        alias,
        observed,
        ("generator-secret", "answer-role", "bridge-role"),
    )


def mention(source: str, target: str, *, observed: bool = True) -> TypedEdge:
    return TypedEdge(
        "title_mention",
        source,
        target,
        observed,
        ("provenance-secret",),
    )


def test_serialization_is_typed_directed_deterministic_and_hashed() -> None:
    artifact = serialize_evidence_topology(
        topology((retrieval("d1"), mention("d0", "d1")))
    )

    assert artifact.serializer_version == SERIALIZER_VERSION
    assert artifact.sha256 == hashlib.sha256(artifact.text.encode("utf-8")).hexdigest()
    assert "retrieval q -> d1" in artifact.text
    assert "title_mention d0 -> d1" in artifact.text
    assert artifact.text.index("retrieval q -> d1") < artifact.text.index(
        "title_mention d0 -> d1"
    )


def test_document_declaration_order_is_alias_order_not_traversal_order() -> None:
    artifact = serialize_evidence_topology(
        topology((retrieval("d1"), mention("d0", "d1")))
    )

    assert artifact.text.index("[d0]") < artifact.text.index("[d1]")
    assert artifact.text.index('title "Neutral Title 0"') < artifact.text.index(
        'title "Neutral Title 1"'
    )


def test_question_label_and_edge_metadata_cannot_leak() -> None:
    observed = topology((retrieval("d0"), mention("d0", "d1")))
    synthetic = topology(
        (retrieval("d0", observed=False), mention("d0", "d1", observed=False))
    )
    observed_text = serialize_evidence_topology(observed).text
    synthetic_text = serialize_evidence_topology(synthetic).text

    assert observed_text == synthetic_text
    forbidden_values = (
        "question text must not appear",
        "question-id-must-not-appear",
        "generator-secret",
        "provenance-secret",
        "answer-role",
        "bridge-role",
    )
    assert all(value not in observed_text.casefold() for value in forbidden_values)
    structural_words = set(re.findall(r"[a-z_]+", observed_text.casefold()))
    assert structural_words.isdisjoint(
        {"question", "label", "generator", "observed", "provenance", "bridge", "answer", "source", "target"}
    )


def test_t1_t2_keep_alias_payload_fixed_and_change_only_explicit_edges() -> None:
    positive = topology((retrieval("d0"), mention("d0", "d1")))
    t1 = topology((retrieval("d0"), mention("d1", "d0", observed=False)))
    t2 = topology((retrieval("d1"), mention("d0", "d1")))

    positive_artifact = serialize_evidence_topology(positive)
    t1_artifact = serialize_evidence_topology(t1)
    t2_artifact = serialize_evidence_topology(t2)
    assert len({positive_artifact.sha256, t1_artifact.sha256, t2_artifact.sha256}) == 3
    assert "title_mention d1 -> d0" in t1_artifact.text
    assert "retrieval q -> d1" in t2_artifact.text
    for artifact in (positive_artifact, t1_artifact, t2_artifact):
        assert artifact.text.index("[d0]") < artifact.text.index("[d1]")


def test_t3_serializes_four_fixed_documents_and_both_switched_edges() -> None:
    positive = topology(
        (
            retrieval("d0"),
            retrieval("d2"),
            mention("d0", "d1"),
            mention("d2", "d3"),
        ),
        count=4,
    )
    switched = topology(
        (
            retrieval("d0"),
            retrieval("d2"),
            mention("d0", "d3", observed=False),
            mention("d2", "d1", observed=False),
        ),
        count=4,
    )
    positive_artifact = serialize_evidence_topology(positive)
    switched_artifact = serialize_evidence_topology(switched)

    assert positive_artifact.sha256 != switched_artifact.sha256
    assert "title_mention d0 -> d3" in switched_artifact.text
    assert "title_mention d2 -> d1" in switched_artifact.text
    positions = [switched_artifact.text.index(f"[d{index}]") for index in range(4)]
    assert positions == sorted(positions)


def test_hf_token_counts_are_exact_and_report_truncation() -> None:
    artifact = serialize_evidence_topology(
        topology((retrieval("d0"), mention("d0", "d1")))
    )
    tokenizer = TinyHFTokenizer()
    full = tokenizer(
        artifact.text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    stats = count_serialization_tokens(artifact, tokenizer, max_length=12)

    assert stats.untruncated_token_count == len(full)
    assert stats.actual_token_count == 12
    assert stats.truncated is True
    assert stats.max_length == 12


def test_matched_relation_multisets_have_identical_token_budgets() -> None:
    tokenizer = TinyHFTokenizer()
    pairs = [
        (
            topology((retrieval("d0"), mention("d0", "d1"))),
            topology((retrieval("d0"), mention("d1", "d0", observed=False))),
        ),
        (
            topology((retrieval("d0"), mention("d0", "d1"))),
            topology((retrieval("d1"), mention("d0", "d1"))),
        ),
        (
            topology(
                (
                    retrieval("d0"),
                    retrieval("d2"),
                    mention("d0", "d1"),
                    mention("d2", "d3"),
                ),
                count=4,
            ),
            topology(
                (
                    retrieval("d0"),
                    retrieval("d2"),
                    mention("d0", "d3", observed=False),
                    mention("d2", "d1", observed=False),
                ),
                count=4,
            ),
        ),
    ]

    for positive, negative in pairs:
        assert positive.relation_multiset == negative.relation_multiset
        positive_stats = count_serialization_tokens(
            serialize_evidence_topology(positive), tokenizer, max_length=4096
        )
        negative_stats = count_serialization_tokens(
            serialize_evidence_topology(negative), tokenizer, max_length=4096
        )
        assert positive_stats == negative_stats
        assert positive_stats.truncated is False
