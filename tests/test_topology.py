from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from topocf_rag.topology import (
    EvidenceDocument,
    EvidenceTopology,
    TopologyInvariantError,
    TypedEdge,
    validate_matched_pair,
)


def documents(count: int = 2) -> tuple[EvidenceDocument, ...]:
    return tuple(
        EvidenceDocument(
            alias=f"d{index}",
            title=f"Document {index}",
            sentences=(f"Fixed sentence {index}.",),
        )
        for index in range(count)
    )


def topology(
    edges: tuple[TypedEdge, ...],
    *,
    evidence_documents: tuple[EvidenceDocument, ...] | None = None,
) -> EvidenceTopology:
    return EvidenceTopology(
        question_id="question-id",
        question="Which document completes the chain?",
        documents=evidence_documents or documents(),
        edges=edges,
    )


def retrieval(target: str, *, observed: bool = True) -> TypedEdge:
    return TypedEdge("retrieval", "q", target, observed)


def mention(
    source: str, target: str, *, observed: bool = True
) -> TypedEdge:
    return TypedEdge("title_mention", source, target, observed)


def test_t1_and_t2_have_same_label_independent_collider_signature() -> None:
    positive = topology((retrieval("d0"), mention("d0", "d1")))
    t1 = topology((retrieval("d0"), mention("d1", "d0", observed=False)))
    t2 = topology((retrieval("d1"), mention("d0", "d1")))

    assert t1.canonical_signature == t2.canonical_signature
    assert t1.canonical_signature != positive.canonical_signature
    assert validate_matched_pair(
        positive,
        t1,
        positive_token_count=100,
        negative_token_count=105,
    ) == pytest.approx(0.05)


def test_canonical_signature_ignores_observation_provenance() -> None:
    observed = topology(
        (
            TypedEdge("retrieval", "q", "d0", True, ("rank:1",)),
            TypedEdge("title_mention", "d0", "d1", True, ("sentence:0",)),
        )
    )
    counterfactual = topology(
        (
            TypedEdge("retrieval", "q", "d0", False, ("synthetic",)),
            TypedEdge("title_mention", "d0", "d1", False, ("synthetic",)),
        )
    )

    assert observed.canonical_signature == counterfactual.canonical_signature


def test_directed_t3_switch_preserves_relations_and_per_node_degrees() -> None:
    docs = documents(4)
    positive = topology(
        (
            retrieval("d0"),
            retrieval("d2"),
            mention("d0", "d1"),
            mention("d2", "d3"),
        ),
        evidence_documents=docs,
    )
    switched = topology(
        (
            retrieval("d0"),
            retrieval("d2"),
            mention("d0", "d3", observed=False),
            mention("d2", "d1", observed=False),
        ),
        evidence_documents=docs,
    )

    assert len(positive.edges) == len(switched.edges)
    assert positive.relation_multiset == switched.relation_multiset
    assert positive.degree_profile == switched.degree_profile
    assert positive.canonical_signature == switched.canonical_signature
    validate_matched_pair(positive, switched)


def test_text_change_and_token_limit_break_matching() -> None:
    positive = topology((retrieval("d0"), mention("d0", "d1")))
    changed_documents = (
        EvidenceDocument("d0", "Document 0", ("Changed sentence.",)),
        documents()[1],
    )
    changed_text = topology(
        (retrieval("d0"), mention("d1", "d0", observed=False)),
        evidence_documents=changed_documents,
    )

    with pytest.raises(TopologyInvariantError, match="document-text multisets"):
        validate_matched_pair(positive, changed_text)
    with pytest.raises(TopologyInvariantError, match="token difference"):
        validate_matched_pair(
            positive,
            topology((retrieval("d0"), mention("d1", "d0", observed=False))),
            positive_token_count=100,
            negative_token_count=106,
        )


def test_document_payload_is_immutable_and_exactly_hashed() -> None:
    document = documents(1)[0]
    changed = EvidenceDocument("d0", document.title, ("Different.",))

    assert document.text_sha256 != changed.text_sha256
    with pytest.raises(FrozenInstanceError):
        document.title = "Mutated"  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory, message",
    [
        (
            lambda: TypedEdge("retrieval", "d0", "d1", True),
            "from q",
        ),
        (
            lambda: TypedEdge("title_mention", "q", "d0", True),
            "two document aliases",
        ),
        (
            lambda: TypedEdge("title_mention", "d0", "d0", True),
            "self-loop",
        ),
        (
            lambda: EvidenceDocument("bridge", "Title", ("Sentence.",)),
            "neutral form",
        ),
    ],
)
def test_invalid_edge_shapes_and_role_bearing_aliases_are_rejected(
    factory, message: str
) -> None:
    with pytest.raises(TopologyInvariantError, match=message):
        factory()


def test_topology_rejects_missing_endpoints_duplicates_and_noncanonical_order() -> None:
    with pytest.raises(TopologyInvariantError, match="not declared"):
        topology((retrieval("d0"), mention("d0", "d2")))

    duplicate = mention("d0", "d1")
    with pytest.raises(TopologyInvariantError, match="duplicate"):
        topology((retrieval("d0"), duplicate, duplicate))

    with pytest.raises(TopologyInvariantError, match="canonical.*order"):
        topology((mention("d0", "d1"), retrieval("d0")))

    with pytest.raises(TopologyInvariantError, match="consecutive aliases"):
        EvidenceTopology(
            question_id="question-id",
            question="Question?",
            documents=(documents()[1], documents()[0]),
            edges=(),
        )


def test_exact_token_counts_must_be_supplied_as_a_pair() -> None:
    positive = topology((retrieval("d0"), mention("d0", "d1")))
    negative = topology((retrieval("d0"), mention("d1", "d0", observed=False)))

    with pytest.raises(TopologyInvariantError, match="supplied together"):
        validate_matched_pair(positive, negative, positive_token_count=100)
