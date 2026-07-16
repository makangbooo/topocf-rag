"""Content-free readiness gate for the TopoCF binding-and-repair method.

The gate intentionally operates on pair manifests and aggregate reports only.
It does not read questions, titles, sentences, answers, or source datasets.
Its job is to prevent a structurally trivial diagnostic split from silently
becoming the primary verifier task.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from .rule_baselines import (
    RulePair,
    canonical_topology_key,
    degree_profile_key,
    parse_rule_manifest,
)


METHOD_READINESS_SCHEMA_VERSION = 1


class MethodReadinessInvariantError(ValueError):
    """Raised when a frozen input or method contract is inconsistent."""


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MethodReadinessInvariantError(f"invalid JSON: {path}") from exc


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MethodReadinessInvariantError(f"{field} must be an object")
    return value


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise MethodReadinessInvariantError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    digest = _require_nonempty_string(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise MethodReadinessInvariantError(
            f"{field} must be lowercase SHA256 hexadecimal"
        )
    return digest


def _resolve_path(config_path: Path, value: Any, field: str) -> Path:
    raw = _require_nonempty_string(value, field)
    path = Path(raw)
    if not path.is_absolute():
        path = (config_path.parent.parent.parent / path).resolve()
    if not path.is_file():
        raise MethodReadinessInvariantError(f"{field} is not a file: {path}")
    return path


def _load_bound_artifact(
    config_path: Path,
    artifact: Any,
    field: str,
) -> tuple[Path, str, Any]:
    payload = _require_mapping(artifact, field)
    path = _resolve_path(config_path, payload.get("path"), f"{field}.path")
    expected = _require_sha256(payload.get("sha256"), f"{field}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise MethodReadinessInvariantError(
            f"{field} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return path, actual, _load_json(path)


def _selector(value: Any, field: str) -> tuple[str, str]:
    payload = _require_mapping(value, field)
    return (
        _require_nonempty_string(payload.get("stratum"), f"{field}.stratum"),
        _require_nonempty_string(payload.get("variant"), f"{field}.variant"),
    )


def _selected_pairs(
    pairs: Sequence[RulePair], selector: tuple[str, str]
) -> tuple[RulePair, ...]:
    stratum, variant = selector
    return tuple(
        pair
        for pair in pairs
        if pair.stratum == stratum and pair.variant == variant
    )


def _raw_records_by_id(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = manifest.get("pairs")
    if not isinstance(records, list):
        raise MethodReadinessInvariantError("manifest pairs must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        payload = _require_mapping(record, "manifest pair")
        pair_id = _require_nonempty_string(payload.get("pair_id"), "pair_id")
        if pair_id in result:
            raise MethodReadinessInvariantError("manifest pair IDs must be unique")
        result[pair_id] = payload
    return result


def _member_relation_multiset(pair: RulePair, role: str) -> Counter[str]:
    member = pair.positive if role == "positive" else pair.negative
    return Counter(edge.relation for edge in member.edges)


def _selector_contract(
    manifest: Mapping[str, Any],
    pairs: Sequence[RulePair],
    selector: tuple[str, str],
) -> dict[str, Any]:
    selected = _selected_pairs(pairs, selector)
    if not selected:
        raise MethodReadinessInvariantError(
            f"selector {selector[0]}/{selector[1]} matched no pairs"
        )
    raw_by_id = _raw_records_by_id(manifest)
    checks = Counter()
    negative_observed_histogram: Counter[str] = Counter()
    canonical_pair_classes: Counter[str] = Counter()
    positive_keys: set[Any] = set()
    negative_keys: set[Any] = set()

    for pair in selected:
        raw = raw_by_id[pair.pair_id]
        aliases = raw.get("neutral_alias_mapping")
        context_indices = raw.get("context_indices")
        sentence_indices = raw.get("sentence_indices")
        if (
            isinstance(aliases, Mapping)
            and tuple(aliases) == ("d0", "d1", "d2", "d3")
            and isinstance(context_indices, list)
            and len(context_indices) == 4
            and len(set(context_indices)) == 4
            and isinstance(sentence_indices, Mapping)
            and tuple(sentence_indices) == ("d0", "d1", "d2", "d3")
        ):
            checks["fixed_four_document_payload"] += 1

        if len(pair.positive.edges) == len(pair.negative.edges) == 4:
            checks["four_edges_each"] += 1
        if (
            _member_relation_multiset(pair, "positive")
            == _member_relation_multiset(pair, "negative")
            == {"retrieval": 2, "title_mention": 2}
        ):
            checks["relation_multiset_matched"] += 1
        if canonical_topology_key(pair.positive) == canonical_topology_key(
            pair.negative
        ):
            checks["canonical_topology_matched"] += 1
        if degree_profile_key(pair.positive) == degree_profile_key(pair.negative):
            checks["per_node_relation_degree_matched"] += 1
        if all(edge.observed for edge in pair.positive.edges):
            checks["positive_all_observed"] += 1

        observed = sum(edge.observed for edge in pair.negative.edges)
        negative_observed_histogram[f"{observed}_of_{len(pair.negative.edges)}"] += 1
        positive_key = canonical_topology_key(pair.positive)
        negative_key = canonical_topology_key(pair.negative)
        positive_keys.add(positive_key)
        negative_keys.add(negative_key)
        canonical_pair_classes[repr((positive_key, negative_key))] += 1

    pair_count = len(selected)
    check_report = {
        name: {
            "count": checks[name],
            "rate": checks[name] / pair_count,
            "passed": checks[name] == pair_count,
        }
        for name in (
            "fixed_four_document_payload",
            "four_edges_each",
            "relation_multiset_matched",
            "canonical_topology_matched",
            "per_node_relation_degree_matched",
            "positive_all_observed",
        )
    }
    return {
        "selector": {"stratum": selector[0], "variant": selector[1]},
        "counts": {
            "pair_count": pair_count,
            "question_count": len({pair.qid for pair in selected}),
            "base_count": len(
                {
                    (pair.qid, raw_by_id[pair.pair_id].get("base_id"))
                    for pair in selected
                }
            ),
        },
        "checks": check_report,
        "negative_observed_edge_histogram": dict(
            sorted(negative_observed_histogram.items())
        ),
        "canonical_summary": {
            "positive_signature_count": len(positive_keys),
            "negative_signature_count": len(negative_keys),
            "pair_class_count": len(canonical_pair_classes),
        },
        "passed": all(check["passed"] for check in check_report.values()),
    }


def _rule_slice(
    rule_report: Mapping[str, Any],
    official_split: str,
    selector: tuple[str, str],
) -> Mapping[str, Any]:
    try:
        return rule_report["synthetic"][official_split][
            "by_stratum_variant"
        ][f"{selector[0]}/{selector[1]}"]
    except (KeyError, TypeError) as exc:
        raise MethodReadinessInvariantError(
            "rule report is missing a selected split/stratum/variant"
        ) from exc


def _rule_control(
    rule_report: Mapping[str, Any],
    official_split: str,
    selector: tuple[str, str],
    *,
    visible_rules: Sequence[str],
    hidden_rules: Sequence[str],
    max_visible_accuracy: float,
) -> dict[str, Any]:
    selected = _rule_slice(rule_report, official_split, selector)
    rules = _require_mapping(selected.get("rules"), "selected rule metrics")

    def values(names: Sequence[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        for name in names:
            metric = _require_mapping(rules.get(name), f"rule {name}")
            value = metric.get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise MethodReadinessInvariantError(f"rule {name} value is invalid")
            result[name] = float(value)
        return result

    visible = values(visible_rules)
    hidden = values(hidden_rules)
    alias_metric = _require_mapping(
        rules.get("alias_topology_lookup"), "rule alias_topology_lookup"
    ).get("value")
    if not isinstance(alias_metric, (int, float)) or isinstance(alias_metric, bool):
        raise MethodReadinessInvariantError("alias topology rule value is invalid")
    best_visible_name, best_visible_value = max(
        visible.items(), key=lambda item: (item[1], item[0])
    )
    return {
        "question_count": selected["counts"]["question_count"],
        "pair_count": selected["counts"]["pair_count"],
        "model_visible_permutation_invariant_rules": visible,
        "best_model_visible_permutation_invariant_rule": {
            "name": best_visible_name,
            "value": best_visible_value,
        },
        "max_allowed_accuracy": max_visible_accuracy,
        "passed": best_visible_value <= max_visible_accuracy,
        "alias_sensitive_lookup_accuracy": float(alias_metric),
        "alias_control": (
            "random S4 relabeling during training and exact 24-permutation score "
            "averaging during dev/test are mandatory"
        ),
        "forbidden_hidden_metadata_rules": hidden,
        "hidden_metadata_interpretation": (
            "observed/provenance metadata is excluded from the serializer; these "
            "scores audit the consequence of accidentally exposing it"
        ),
    }


def _baseline_slice(
    baseline_report: Mapping[str, Any],
    official_split: str,
    selector: tuple[str, str],
) -> dict[str, Any]:
    try:
        selected = baseline_report["splits"][official_split][
            "by_stratum_variant"
        ][selector[0]][selector[1]]
    except (KeyError, TypeError) as exc:
        raise MethodReadinessInvariantError(
            "baseline report is missing a selected split/stratum/variant"
        ) from exc
    baselines = _require_mapping(selected.get("baselines"), "baseline metrics")
    pairwise: dict[str, float] = {}
    for name, metrics in baselines.items():
        payload = _require_mapping(metrics, f"baseline {name}")
        pair_metric = _require_mapping(
            payload.get("pairwise_accuracy"), f"baseline {name} pairwise_accuracy"
        )
        value = pair_metric.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise MethodReadinessInvariantError(
                f"baseline {name} pairwise value is invalid"
            )
        pairwise[str(name)] = float(value)
    return {
        "counts": dict(selected["counts"]),
        "question_macro_pairwise_accuracy": dict(sorted(pairwise.items())),
    }


def _validate_threshold(value: Any, field: str, *, lower: float, upper: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not lower <= float(value) <= upper
    ):
        raise MethodReadinessInvariantError(
            f"{field} must be between {lower} and {upper}"
        )
    return float(value)


def build_method_readiness_report(config_path: str | Path) -> dict[str, Any]:
    """Validate frozen artifacts and return an aggregate-only gate report."""

    resolved_config = Path(config_path).resolve()
    config = _require_mapping(_load_json(resolved_config), "config")
    if config.get("schema_version") != METHOD_READINESS_SCHEMA_VERSION:
        raise MethodReadinessInvariantError("unsupported method config schema")

    artifacts = _require_mapping(config.get("artifacts"), "artifacts")
    loaded: dict[str, tuple[Path, str, Any]] = {
        name: _load_bound_artifact(resolved_config, artifact, f"artifacts.{name}")
        for name, artifact in artifacts.items()
    }
    required_artifacts = {
        "synthetic_train_manifest",
        "synthetic_dev_manifest",
        "phase1_baselines",
        "rule_leakage",
    }
    if set(loaded) != required_artifacts:
        raise MethodReadinessInvariantError(
            "artifacts must contain exactly the four frozen Phase 1 inputs"
        )

    train_manifest = _require_mapping(
        loaded["synthetic_train_manifest"][2], "synthetic train manifest"
    )
    dev_manifest = _require_mapping(
        loaded["synthetic_dev_manifest"][2], "synthetic dev manifest"
    )
    train_split, train_pairs = parse_rule_manifest(train_manifest)
    dev_split, dev_pairs = parse_rule_manifest(dev_manifest)
    if train_split != "train" or dev_split != "dev_distractor":
        raise MethodReadinessInvariantError("official train/dev split names changed")
    overlap = {pair.qid for pair in train_pairs}.intersection(
        pair.qid for pair in dev_pairs
    )
    if overlap:
        raise MethodReadinessInvariantError("train and dev question IDs overlap")

    selectors = _require_mapping(config.get("selectors"), "selectors")
    development_selector = _selector(
        selectors.get("development"), "selectors.development"
    )
    confirmatory_selector = _selector(
        selectors.get("confirmatory"), "selectors.confirmatory"
    )
    excluded = selectors.get("diagnostic_only")
    if not isinstance(excluded, list) or not excluded:
        raise MethodReadinessInvariantError(
            "selectors.diagnostic_only must be a non-empty list"
        )
    diagnostic_selectors = [
        _selector(value, f"selectors.diagnostic_only[{index}]")
        for index, value in enumerate(excluded)
    ]
    if development_selector == confirmatory_selector:
        raise MethodReadinessInvariantError(
            "development and confirmatory selectors must differ"
        )
    if (
        len(set(diagnostic_selectors)) != len(diagnostic_selectors)
        or development_selector in diagnostic_selectors
        or confirmatory_selector in diagnostic_selectors
    ):
        raise MethodReadinessInvariantError(
            "diagnostic selectors must be unique and disjoint from primary roles"
        )
    for selector_value in diagnostic_selectors:
        if not _selected_pairs(train_pairs, selector_value) or not _selected_pairs(
            dev_pairs, selector_value
        ):
            raise MethodReadinessInvariantError(
                f"diagnostic selector {selector_value[0]}/{selector_value[1]} "
                "must exist in train and dev"
            )

    thresholds = _require_mapping(config.get("thresholds"), "thresholds")
    max_visible = _validate_threshold(
        thresholds.get("max_visible_structural_rule_accuracy"),
        "thresholds.max_visible_structural_rule_accuracy",
        lower=0.5,
        upper=1.0,
    )
    minimum_train_questions = thresholds.get("minimum_development_train_questions")
    minimum_dev_questions = thresholds.get("minimum_development_dev_questions")
    minimum_confirmatory_questions = thresholds.get(
        "minimum_confirmatory_dev_questions"
    )
    for name, value in (
        ("minimum_development_train_questions", minimum_train_questions),
        ("minimum_development_dev_questions", minimum_dev_questions),
        ("minimum_confirmatory_dev_questions", minimum_confirmatory_questions),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise MethodReadinessInvariantError(f"thresholds.{name} must be positive")

    visible_rules = config.get("model_visible_permutation_invariant_rules")
    hidden_rules = config.get("forbidden_hidden_metadata_rules")
    if not isinstance(visible_rules, list) or not visible_rules:
        raise MethodReadinessInvariantError(
            "model_visible_permutation_invariant_rules must be non-empty"
        )
    if not isinstance(hidden_rules, list) or not hidden_rules:
        raise MethodReadinessInvariantError(
            "forbidden_hidden_metadata_rules must be non-empty"
        )
    if any(not isinstance(name, str) or not name for name in (*visible_rules, *hidden_rules)):
        raise MethodReadinessInvariantError("rule names must be non-empty strings")

    rule_report = _require_mapping(loaded["rule_leakage"][2], "rule report")
    baseline_report = _require_mapping(
        loaded["phase1_baselines"][2], "baseline report"
    )
    manifest_hashes = {
        "train": loaded["synthetic_train_manifest"][1],
        "dev_distractor": loaded["synthetic_dev_manifest"][1],
    }
    for split_name, digest in manifest_hashes.items():
        try:
            bound = baseline_report["splits"][split_name]["input_hashes"][
                "manifest_sha256"
            ]
        except (KeyError, TypeError) as exc:
            raise MethodReadinessInvariantError(
                "baseline report is missing a manifest hash"
            ) from exc
        if bound != digest:
            raise MethodReadinessInvariantError(
                f"baseline {split_name} manifest hash does not match"
            )

    train_development = _selector_contract(
        train_manifest, train_pairs, development_selector
    )
    dev_development = _selector_contract(
        dev_manifest, dev_pairs, development_selector
    )
    train_confirmatory = _selector_contract(
        train_manifest, train_pairs, confirmatory_selector
    )
    dev_confirmatory = _selector_contract(
        dev_manifest, dev_pairs, confirmatory_selector
    )

    controls = {
        "development_train": _rule_control(
            rule_report,
            "train",
            development_selector,
            visible_rules=visible_rules,
            hidden_rules=hidden_rules,
            max_visible_accuracy=max_visible,
        ),
        "development_dev": _rule_control(
            rule_report,
            "dev_distractor",
            development_selector,
            visible_rules=visible_rules,
            hidden_rules=hidden_rules,
            max_visible_accuracy=max_visible,
        ),
        "confirmatory_train": _rule_control(
            rule_report,
            "train",
            confirmatory_selector,
            visible_rules=visible_rules,
            hidden_rules=hidden_rules,
            max_visible_accuracy=max_visible,
        ),
        "confirmatory_dev": _rule_control(
            rule_report,
            "dev_distractor",
            confirmatory_selector,
            visible_rules=visible_rules,
            hidden_rules=hidden_rules,
            max_visible_accuracy=max_visible,
        ),
    }
    baseline_summary = {
        "development_train": _baseline_slice(
            baseline_report, "train", development_selector
        ),
        "development_dev": _baseline_slice(
            baseline_report, "dev_distractor", development_selector
        ),
        "confirmatory_train": _baseline_slice(
            baseline_report, "train", confirmatory_selector
        ),
        "confirmatory_dev": _baseline_slice(
            baseline_report, "dev_distractor", confirmatory_selector
        ),
    }

    def counts_match(contract: Mapping[str, Any], key: str) -> bool:
        baseline_counts = baseline_summary[key]["counts"]
        rule_counts = controls[key]
        return (
            baseline_counts["pair_count"] == contract["counts"]["pair_count"]
            and baseline_counts["question_count"]
            == contract["counts"]["question_count"]
            and rule_counts["pair_count"] == contract["counts"]["pair_count"]
            and rule_counts["question_count"]
            == contract["counts"]["question_count"]
        )

    development_checks = {
        "train_pair_contract": train_development["passed"],
        "dev_pair_contract": dev_development["passed"],
        "train_dev_question_disjoint": not overlap,
        "baseline_hash_bound": True,
        "train_aggregate_counts_match": counts_match(
            train_development, "development_train"
        ),
        "dev_aggregate_counts_match": counts_match(
            dev_development, "development_dev"
        ),
        "train_visible_rule_gate": controls["development_train"]["passed"],
        "dev_visible_rule_gate": controls["development_dev"]["passed"],
        "train_question_capacity": (
            train_development["counts"]["question_count"]
            >= minimum_train_questions
        ),
        "dev_question_capacity": (
            dev_development["counts"]["question_count"] >= minimum_dev_questions
        ),
        "s4_alias_permutation_control_required": True,
        "observed_and_provenance_metadata_hidden_required": True,
    }
    confirmatory_checks = {
        "train_pair_contract": train_confirmatory["passed"],
        "dev_pair_contract": dev_confirmatory["passed"],
        "train_visible_rule_gate": controls["confirmatory_train"]["passed"],
        "dev_visible_rule_gate": controls["confirmatory_dev"]["passed"],
        "train_aggregate_counts_match": counts_match(
            train_confirmatory, "confirmatory_train"
        ),
        "dev_aggregate_counts_match": counts_match(
            dev_confirmatory, "confirmatory_dev"
        ),
        "dev_question_capacity": (
            dev_confirmatory["counts"]["question_count"]
            >= minimum_confirmatory_questions
        ),
        "negative_edges_all_observed": set(
            dev_confirmatory["negative_observed_edge_histogram"]
        )
        == {"4_of_4"},
    }
    development_authorized = all(development_checks.values())
    confirmatory_ready = all(confirmatory_checks.values())

    return {
        "schema_version": METHOD_READINESS_SCHEMA_VERSION,
        "config_sha256": sha256_file(resolved_config),
        "method": {
            "name": _require_nonempty_string(config.get("method_name"), "method_name"),
            "claim": _require_nonempty_string(config.get("claim"), "claim"),
            "training_objective": config.get("training_objective"),
            "permutation_protocol": config.get("permutation_protocol"),
        },
        "content_contract": (
            "aggregate manifest metadata, rule metrics, counts, and hashes only; "
            "no question, answer, title, sentence, or source text"
        ),
        "artifacts": {
            name: {
                "path": _require_mapping(
                    artifacts[name], f"artifacts.{name}"
                )["path"],
                "sha256": digest,
            }
            for name, (_path, digest, _payload) in sorted(loaded.items())
        },
        "split_integrity": {
            "train_question_count": len({pair.qid for pair in train_pairs}),
            "dev_question_count": len({pair.qid for pair in dev_pairs}),
            "overlap_count": len(overlap),
            "passed": not overlap,
        },
        "primary_development": {
            "train": train_development,
            "dev_distractor": dev_development,
        },
        "confirmatory_all_observed": {
            "train": train_confirmatory,
            "dev_distractor": dev_confirmatory,
        },
        "rule_controls": controls,
        "semantic_baselines": baseline_summary,
        "diagnostic_only_selectors": [
            {"stratum": stratum, "variant": variant}
            for stratum, variant in diagnostic_selectors
        ],
        "gates": {
            "development_training": {
                "checks": development_checks,
                "passed": development_authorized,
                "status": (
                    "authorized_for_bounded_method_development"
                    if development_authorized
                    else "blocked_before_training"
                ),
            },
            "confirmatory_evaluation": {
                "checks": confirmatory_checks,
                "passed": confirmatory_ready,
                "status": (
                    "ready"
                    if confirmatory_ready
                    else "underpowered_hotpot_all_observed_stratum"
                ),
            },
            "paper_claim": {
                "passed": False,
                "status": "not_authorized_by_readiness_audit",
                "remaining_requirements": [
                    "implement and verify exact S4 score invariance",
                    "train the counterfactual ranking plus structured repair objective",
                    "expand the all-observed confirmatory stratum to the frozen minimum",
                    "build a schema-matched natural-retrieval-error evaluation",
                    "replicate the frozen method on a second official dataset",
                ],
            },
        },
        "interpretation": {
            "why_t1_t2_are_not_primary": (
                "their labels are recoverable from explicit reachability/canonical "
                "topology, so they measure graph parsing rather than semantic binding"
            ),
            "why_t3_is_primary": (
                "positive and negative match canonical typed topology and every node's "
                "per-relation directed degree; the signal is document-node binding"
            ),
            "why_training_is_bounded": (
                "the synthetic T3 split supports development, but the all-observed "
                "HotpotQA confirmatory split is below the preregistered sample minimum"
            ),
        },
    }
