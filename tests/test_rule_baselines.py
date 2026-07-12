from __future__ import annotations

import json

import pytest

from topocf_rag.rule_baselines import (
    QuestionBalancedLookup,
    RuleAuditInvariantError,
    alias_topology_key,
    build_rule_leakage_audit,
    canonical_topology_key,
    composable_two_hop_fraction,
    observed_edge_fraction,
    parse_rule_manifest,
    root_reachable_fraction,
)


def edge(relation, source, target, observed=True):
    return {
        "relation": relation,
        "source": source,
        "target": target,
        "observed": observed,
    }


POSITIVE = [
    edge("retrieval", "q", "d0"),
    edge("retrieval", "q", "d1"),
    edge("title_mention", "d0", "d2"),
    edge("title_mention", "d1", "d3"),
]
COLLIDER = [
    edge("retrieval", "q", "d0"),
    edge("retrieval", "q", "d1"),
    edge("title_mention", "d2", "d0"),
    edge("title_mention", "d3", "d1"),
]
SYNTHETIC_COLLIDER = [
    edge("retrieval", "q", "d0"),
    edge("retrieval", "q", "d1"),
    edge("title_mention", "d2", "d0", False),
    edge("title_mention", "d3", "d1", False),
]


def record(qid, pair_id, negative, *, stratum="synthetic_common", variant="t1"):
    return {
        "qid": qid,
        "pair_id": pair_id,
        "stratum": stratum,
        "variant": variant,
        "positive": {"typed_edges": POSITIVE},
        "negative": {"typed_edges": negative},
    }


def manifest(records, *, schema_version=1, split="train"):
    return {
        "schema_version": schema_version,
        "official_split": split,
        "pairs": records,
    }


def parsed(records, *, schema_version=1, split="train"):
    return parse_rule_manifest(
        manifest(records, schema_version=schema_version, split=split)
    )[1]


def test_structural_rules_detect_noncomposable_colliders() -> None:
    pair = parsed([record("q1", "p1", COLLIDER)])[0]
    assert root_reachable_fraction(pair.positive, observed_only=False) == 1.0
    assert root_reachable_fraction(pair.negative, observed_only=False) == 0.5
    assert composable_two_hop_fraction(pair.positive, observed_only=False) == 1.0
    assert composable_two_hop_fraction(pair.negative, observed_only=False) == 0.0


def test_observation_rule_exposes_synthetic_edge_metadata() -> None:
    pair = parsed([record("q1", "p1", SYNTHETIC_COLLIDER)])[0]
    assert observed_edge_fraction(pair.positive) == 1.0
    assert observed_edge_fraction(pair.negative) == 0.5


def test_canonical_key_ignores_document_alias_permutations() -> None:
    first = parsed([record("q1", "p1", COLLIDER)])[0].negative
    relabeled = [
        edge("retrieval", "q", "d2"),
        edge("retrieval", "q", "d3"),
        edge("title_mention", "d0", "d2"),
        edge("title_mention", "d1", "d3"),
    ]
    second = parsed([record("q2", "p2", relabeled)])[0].negative
    assert alias_topology_key(first) != alias_topology_key(second)
    assert canonical_topology_key(first) == canonical_topology_key(second)


def test_lookup_can_exclude_the_scored_training_question() -> None:
    pairs = parsed(
        [
            record("q1", "p1", COLLIDER),
            record("q2", "p2", COLLIDER),
        ]
    )
    lookup = QuestionBalancedLookup(alias_topology_key).fit(pairs)
    positive_score = lookup.score(pairs[0].positive, exclude_qid="q1")
    negative_score = lookup.score(pairs[0].negative, exclude_qid="q1")
    assert positive_score > 0.5
    assert negative_score < 0.5


def test_full_audit_fails_gate_when_a_simple_rule_solves_dev() -> None:
    synthetic_train = parsed(
        [record("q1", "p1", COLLIDER), record("q2", "p2", COLLIDER)]
    )
    synthetic_dev = parsed(
        [record("q3", "p3", COLLIDER)], split="dev_distractor"
    )
    natural_train = parsed(
        [record("n1", "n1-p", POSITIVE, stratum="natural", variant="natural")],
        schema_version=2,
    )
    natural_dev = parsed(
        [record("n2", "n2-p", POSITIVE, stratum="natural", variant="natural")],
        schema_version=2,
        split="dev_distractor",
    )
    report = build_rule_leakage_audit(
        synthetic_train=synthetic_train,
        synthetic_dev=synthetic_dev,
        natural_train=natural_train,
        natural_dev=natural_dev,
    )
    gate = report["synthetic_dev_gate"]
    assert gate["passed"] is False
    assert gate["best_question_macro_pairwise_accuracy"] == 1.0
    encoded = json.dumps(report)
    assert "q1" not in encoded
    assert "p1" not in encoded


def test_natural_identical_structures_tie() -> None:
    pairs = parsed(
        [record("n1", "pair", POSITIVE, stratum="natural", variant="natural")],
        schema_version=2,
    )
    report = build_rule_leakage_audit(
        synthetic_train=pairs,
        synthetic_dev=pairs,
        natural_train=pairs,
        natural_dev=pairs,
    )
    assert all(
        metrics["value"] == 0.5
        for metrics in report["natural"]["dev_distractor"]["rules"].values()
    )


def test_invalid_relation_is_rejected() -> None:
    bad = [edge("unsupported", "q", "d0")]
    with pytest.raises(RuleAuditInvariantError, match="unsupported relation"):
        parsed([record("q1", "p1", bad)])
