from __future__ import annotations

from scripts.audit_phase1_design import (
    audit_directed_edges,
    canonical_overlap_audit,
)


GOLD = (0, 1)
RETRIEVED = {0, 1, 2, 3, 4}


def test_t3_deduplicates_repeated_directed_endpoint_pairs() -> None:
    audit = audit_directed_edges(
        [(0, 1), (2, 3), (2, 3)],
        gold_indices=GOLD,
        retrieved_indices=RETRIEVED,
    )
    assert audit.t1_operations == 1
    assert audit.t2_operations == 1
    assert audit.t3_disjoint_original_edge_pairs == 1
    assert audit.t3_synthetic_operations == 1


def test_t3_requires_four_distinct_document_vertices() -> None:
    audit = audit_directed_edges(
        {(0, 1), (1, 2), (2, 0)},
        gold_indices=GOLD,
        retrieved_indices=RETRIEVED,
    )
    assert audit.t3_disjoint_original_edge_pairs == 0
    assert audit.t3_synthetic_operations == 0
    assert audit.t3_natural_operations == 0


def test_t3_excludes_synthetic_switch_when_new_edge_exists() -> None:
    partially_real = audit_directed_edges(
        {(0, 1), (2, 3), (0, 3)},
        gold_indices=GOLD,
        retrieved_indices=RETRIEVED,
    )
    assert partially_real.t3_disjoint_original_edge_pairs == 1
    assert partially_real.t3_synthetic_operations == 0
    assert partially_real.t3_natural_operations == 0
    assert partially_real.t3_partially_real_rejected == 1

    both_real = audit_directed_edges(
        {(0, 1), (2, 3), (0, 3), (2, 1)},
        gold_indices=GOLD,
        retrieved_indices=RETRIEVED,
    )
    assert both_real.t3_disjoint_original_edge_pairs == 1
    assert both_real.t3_synthetic_operations == 0
    assert both_real.t3_natural_operations == 1
    assert both_real.t3_partially_real_rejected == 0


def test_bidirectional_gold_edge_is_not_eligible() -> None:
    audit = audit_directed_edges(
        {(0, 1), (1, 0), (2, 3)},
        gold_indices=GOLD,
        retrieved_indices=RETRIEVED,
    )
    assert audit.gold_bidirectional_mention_edge == 1
    assert audit.t1_operations == 0
    assert audit.t2_operations == 0
    assert audit.t3_disjoint_original_edge_pairs == 0


def test_t2_requires_both_real_retrieval_anchors() -> None:
    audit = audit_directed_edges(
        {(0, 1), (2, 3)},
        gold_indices=GOLD,
        retrieved_indices={0, 2, 3},
    )
    assert audit.gold_asymmetric_mention_edge == 1
    assert audit.t1_operations == 0
    assert audit.t2_operations == 0


def test_t3_requires_both_original_edge_sources_to_be_retrieved() -> None:
    missing_other_anchor = audit_directed_edges(
        {(0, 1), (2, 3)},
        gold_indices=GOLD,
        retrieved_indices={0, 1, 3},
    )
    assert missing_other_anchor.t1_operations == 1
    assert missing_other_anchor.t2_operations == 1
    assert missing_other_anchor.t3_disjoint_original_edge_pairs == 0
    assert missing_other_anchor.t3_synthetic_operations == 0

    missing_gold_source_anchor = audit_directed_edges(
        {(0, 1), (2, 3)},
        gold_indices=GOLD,
        retrieved_indices={1, 2, 3},
    )
    assert missing_gold_source_anchor.t3_disjoint_original_edge_pairs == 0
    assert missing_gold_source_anchor.t3_synthetic_operations == 0


def test_t1_and_t2_negative_topologies_have_same_canonical_signature() -> None:
    audit = canonical_overlap_audit()
    assert audit["signatures_overlap"] is True
    assert audit["t1_negative_signature"] == audit["t2_negative_signature"]
    assert "title_mention:" in audit["t1_negative_signature"]
    assert all(
        not edge.startswith("mention:")
        for edge in audit["t1_negative_signature"].split("|")
    )


def test_t3_positive_and_negative_have_same_canonical_signature() -> None:
    audit = canonical_overlap_audit()
    assert audit["t3_signatures_overlap"] is True
    assert audit["t3_positive_signature"] == audit["t3_negative_signature"]
    assert audit["t3_positive_signature"].count("title_mention:") == 2
