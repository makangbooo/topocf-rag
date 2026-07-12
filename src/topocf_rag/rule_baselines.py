"""Content-free structural leakage rules for Phase 1 pair manifests.

The audit deliberately consumes only manifest metadata. It never reads HotpotQA
question, answer, title, sentence, or document text. Learned lookup rules are
fit on train and evaluated on dev; train values use leave-one-question-out
counts so a question cannot label itself.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import permutations
import math
from typing import Any


_RELATIONS = ("retrieval", "title_mention")
_RELATION_ORDER = {name: index for index, name in enumerate(_RELATIONS)}


class RuleAuditInvariantError(ValueError):
    """Raised when a content-free pair manifest violates the audit contract."""


@dataclass(frozen=True, slots=True)
class RuleEdge:
    relation: str
    source: str
    target: str
    observed: bool

    def __post_init__(self) -> None:
        if self.relation not in _RELATION_ORDER:
            raise RuleAuditInvariantError("unsupported relation")
        if not isinstance(self.source, str) or not isinstance(self.target, str):
            raise RuleAuditInvariantError("edge endpoints must be strings")
        if not isinstance(self.observed, bool):
            raise RuleAuditInvariantError("edge observed flag must be bool")
        if self.relation == "retrieval":
            if self.source != "q" or not _is_alias(self.target):
                raise RuleAuditInvariantError(
                    "retrieval edges must be directed from q to a document alias"
                )
        elif not _is_alias(self.source) or not _is_alias(self.target):
            raise RuleAuditInvariantError(
                "title-mention edges must connect document aliases"
            )
        if self.source == self.target:
            raise RuleAuditInvariantError("self-loops are not allowed")

    @property
    def structural_key(self) -> tuple[str, str, str]:
        return (self.relation, self.source, self.target)


@dataclass(frozen=True, slots=True)
class RuleMember:
    edges: tuple[RuleEdge, ...]
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RulePair:
    pair_id: str
    qid: str
    stratum: str
    variant: str
    positive: RuleMember
    negative: RuleMember


@dataclass(frozen=True, slots=True)
class PairwiseRuleReport:
    value: float | None
    question_count: int
    pair_count: int
    win_count: int
    tie_count: int
    loss_count: int


def _is_alias(value: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("d")
        and value[1:].isdigit()
        and str(int(value[1:])) == value[1:]
    )


def _alias_index(value: str) -> int:
    if not _is_alias(value):
        raise RuleAuditInvariantError("invalid document alias")
    return int(value[1:])


def _edge_sort_key(edge: RuleEdge) -> tuple[int, int, int, bool]:
    source = -1 if edge.source == "q" else _alias_index(edge.source)
    return (
        _RELATION_ORDER[edge.relation],
        source,
        _alias_index(edge.target),
        edge.observed,
    )


def _parse_member(value: Any) -> RuleMember:
    if not isinstance(value, Mapping) or not isinstance(value.get("typed_edges"), list):
        raise RuleAuditInvariantError("pair member must contain typed_edges")
    edges: list[RuleEdge] = []
    identities: set[tuple[str, str, str]] = set()
    aliases: set[str] = set()
    for payload in value["typed_edges"]:
        if not isinstance(payload, Mapping):
            raise RuleAuditInvariantError("typed edge must be an object")
        edge = RuleEdge(
            relation=payload.get("relation"),
            source=payload.get("source"),
            target=payload.get("target"),
            observed=payload.get("observed"),
        )
        if edge.structural_key in identities:
            raise RuleAuditInvariantError("duplicate typed edge")
        identities.add(edge.structural_key)
        edges.append(edge)
        aliases.add(edge.target)
        if edge.source != "q":
            aliases.add(edge.source)
    if not edges:
        raise RuleAuditInvariantError("pair member must contain at least one edge")
    ordered_aliases = tuple(sorted(aliases, key=_alias_index))
    if ordered_aliases != tuple(f"d{index}" for index in range(len(ordered_aliases))):
        raise RuleAuditInvariantError("document aliases must be consecutive from d0")
    return RuleMember(tuple(sorted(edges, key=_edge_sort_key)), ordered_aliases)


def parse_rule_manifest(payload: Any) -> tuple[str, tuple[RulePair, ...]]:
    """Validate schema-v1 synthetic or schema-v2 natural pair metadata."""

    if not isinstance(payload, Mapping):
        raise RuleAuditInvariantError("manifest must be an object")
    if payload.get("schema_version") not in {1, 2}:
        raise RuleAuditInvariantError("unsupported manifest schema version")
    official_split = payload.get("official_split")
    records = payload.get("pairs")
    if not isinstance(official_split, str) or not official_split:
        raise RuleAuditInvariantError("manifest must name an official split")
    if not isinstance(records, list):
        raise RuleAuditInvariantError("manifest must contain a pairs list")

    pairs: list[RulePair] = []
    seen: set[str] = set()
    default_stratum = payload.get("stratum", "unspecified")
    for record in records:
        if not isinstance(record, Mapping):
            raise RuleAuditInvariantError("pair record must be an object")
        pair_id = record.get("pair_id")
        qid = record.get("qid")
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
            raise RuleAuditInvariantError("pair IDs must be unique non-empty strings")
        if not isinstance(qid, str) or not qid:
            raise RuleAuditInvariantError("qid must be a non-empty string")
        seen.add(pair_id)
        stratum = record.get("stratum", default_stratum)
        variant = record.get("variant", "natural")
        if not isinstance(stratum, str) or not stratum:
            raise RuleAuditInvariantError("stratum must be a non-empty string")
        if not isinstance(variant, str) or not variant:
            raise RuleAuditInvariantError("variant must be a non-empty string")
        pairs.append(
            RulePair(
                pair_id=pair_id,
                qid=qid,
                stratum=stratum,
                variant=variant,
                positive=_parse_member(record.get("positive")),
                negative=_parse_member(record.get("negative")),
            )
        )
    return official_split, tuple(pairs)


def observed_edge_fraction(member: RuleMember) -> float:
    return sum(edge.observed for edge in member.edges) / len(member.edges)


def observed_title_mention_fraction(member: RuleMember) -> float:
    mentions = [edge for edge in member.edges if edge.relation == "title_mention"]
    return sum(edge.observed for edge in mentions) / len(mentions) if mentions else 0.0


def root_reachable_fraction(member: RuleMember, *, observed_only: bool) -> float:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in member.edges:
        if observed_only and not edge.observed:
            continue
        adjacency[edge.source].append(edge.target)
    reached = {"q"}
    queue = deque(["q"])
    while queue:
        source = queue.popleft()
        for target in adjacency.get(source, []):
            if target not in reached:
                reached.add(target)
                queue.append(target)
    return sum(alias in reached for alias in member.aliases) / len(member.aliases)


def composable_two_hop_fraction(member: RuleMember, *, observed_only: bool) -> float:
    retrieval_targets = {
        edge.target
        for edge in member.edges
        if edge.relation == "retrieval" and (edge.observed or not observed_only)
    }
    mentions = [
        edge
        for edge in member.edges
        if edge.relation == "title_mention" and (edge.observed or not observed_only)
    ]
    if not mentions:
        return 0.0
    return sum(edge.source in retrieval_targets for edge in mentions) / len(mentions)


def alias_topology_key(member: RuleMember) -> Hashable:
    return tuple(edge.structural_key for edge in member.edges)


def full_metadata_key(member: RuleMember) -> Hashable:
    return tuple(
        (edge.relation, edge.source, edge.target, edge.observed)
        for edge in member.edges
    )


def edge_reality_key(member: RuleMember) -> Hashable:
    return tuple(
        (
            relation,
            sum(edge.relation == relation and edge.observed for edge in member.edges),
            sum(
                edge.relation == relation and not edge.observed
                for edge in member.edges
            ),
        )
        for relation in _RELATIONS
    )


def degree_profile_key(member: RuleMember) -> Hashable:
    nodes = ("q", *member.aliases)
    return tuple(
        (
            node,
            tuple(
                (
                    relation,
                    sum(
                        edge.relation == relation and edge.target == node
                        for edge in member.edges
                    ),
                    sum(
                        edge.relation == relation and edge.source == node
                        for edge in member.edges
                    ),
                )
                for relation in _RELATIONS
            ),
        )
        for node in nodes
    )


def canonical_topology_key(member: RuleMember) -> Hashable:
    best: tuple[tuple[str, str, str], ...] | None = None
    for aliases_in_new_order in permutations(member.aliases):
        relabel = {
            old: f"d{new_index}"
            for new_index, old in enumerate(aliases_in_new_order)
        }
        encoded = tuple(
            sorted(
                (
                    edge.relation,
                    edge.source if edge.source == "q" else relabel[edge.source],
                    relabel[edge.target],
                )
                for edge in member.edges
            )
        )
        if best is None or encoded < best:
            best = encoded
    if best is None:
        raise RuleAuditInvariantError("canonicalization requires document aliases")
    return (len(member.aliases), best)


FeatureFunction = Callable[[RuleMember], Hashable]


class QuestionBalancedLookup:
    """Laplace-smoothed label lookup with one total vote per question/key."""

    def __init__(self, feature: FeatureFunction) -> None:
        self.feature = feature
        self._totals: dict[Hashable, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self._question: dict[tuple[str, Hashable], tuple[float, float]] = {}

    def fit(self, pairs: Sequence[RulePair]) -> "QuestionBalancedLookup":
        raw: Counter[tuple[str, Hashable, bool]] = Counter()
        for pair in pairs:
            raw[(pair.qid, self.feature(pair.positive), True)] += 1
            raw[(pair.qid, self.feature(pair.negative), False)] += 1
        question_keys = {(qid, key) for qid, key, _label in raw}
        for qid, key in sorted(
            question_keys, key=lambda item: (item[0], repr(item[1]))
        ):
            positive = raw[(qid, key, True)]
            negative = raw[(qid, key, False)]
            total = positive + negative
            contribution = (positive / total, negative / total)
            self._question[(qid, key)] = contribution
            self._totals[key][0] += contribution[0]
            self._totals[key][1] += contribution[1]
        return self

    @property
    def feature_cardinality(self) -> int:
        return len(self._totals)

    def score(self, member: RuleMember, *, exclude_qid: str | None = None) -> float:
        key = self.feature(member)
        positive, negative = self._totals.get(key, [0.0, 0.0])
        if exclude_qid is not None:
            held_out = self._question.get((exclude_qid, key), (0.0, 0.0))
            positive -= held_out[0]
            negative -= held_out[1]
        if positive < -1e-12 or negative < -1e-12:
            raise RuleAuditInvariantError("lookup exclusion produced a negative count")
        return (max(0.0, positive) + 1.0) / (
            max(0.0, positive) + max(0.0, negative) + 2.0
        )


def _negative_edge_reality(pair: RulePair) -> str:
    observed = sum(edge.observed for edge in pair.negative.edges)
    total = len(pair.negative.edges)
    if observed == total:
        return "all_observed"
    if observed == 0:
        return "all_counterfactual"
    return f"mixed_{observed}_observed_{total - observed}_counterfactual"


def macro_pairwise_rule(
    pairs: Sequence[RulePair],
    scorer: Callable[[RuleMember, str], float],
) -> PairwiseRuleReport:
    grouped: dict[str, list[float]] = defaultdict(list)
    wins = ties = losses = 0
    for pair in pairs:
        positive = float(scorer(pair.positive, pair.qid))
        negative = float(scorer(pair.negative, pair.qid))
        if not math.isfinite(positive) or not math.isfinite(negative):
            raise RuleAuditInvariantError("rule scores must be finite")
        if positive > negative:
            outcome = 1.0
            wins += 1
        elif positive == negative:
            outcome = 0.5
            ties += 1
        else:
            outcome = 0.0
            losses += 1
        grouped[pair.qid].append(outcome)
    question_values = [sum(values) / len(values) for values in grouped.values()]
    return PairwiseRuleReport(
        value=(
            sum(question_values) / len(question_values)
            if question_values
            else None
        ),
        question_count=len(question_values),
        pair_count=len(pairs),
        win_count=wins,
        tie_count=ties,
        loss_count=losses,
    )


_DETERMINISTIC_RULES: dict[str, Callable[[RuleMember], float]] = {
    "observed_edge_fraction": observed_edge_fraction,
    "observed_title_mention_fraction": observed_title_mention_fraction,
    "root_reachable_fraction_all_edges": lambda member: root_reachable_fraction(
        member, observed_only=False
    ),
    "root_reachable_fraction_observed_edges": lambda member: root_reachable_fraction(
        member, observed_only=True
    ),
    "composable_two_hop_fraction_all_edges": lambda member: composable_two_hop_fraction(
        member, observed_only=False
    ),
    "composable_two_hop_fraction_observed_edges": (
        lambda member: composable_two_hop_fraction(member, observed_only=True)
    ),
}

_LOOKUP_FEATURES: dict[str, FeatureFunction] = {
    "canonical_topology_lookup": canonical_topology_key,
    "alias_topology_lookup": alias_topology_key,
    "degree_profile_lookup": degree_profile_key,
    "edge_reality_lookup": edge_reality_key,
    "full_structural_metadata_lookup": full_metadata_key,
}


def _slice_report(
    pairs: Sequence[RulePair],
    lookups: Mapping[str, QuestionBalancedLookup],
    *,
    leave_one_question_out: bool,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for name, rule in _DETERMINISTIC_RULES.items():
        reports[name] = asdict(
            macro_pairwise_rule(pairs, lambda member, _qid, fn=rule: fn(member))
        )
    for name, lookup in lookups.items():
        reports[name] = asdict(
            macro_pairwise_rule(
                pairs,
                lambda member, qid, model=lookup: model.score(
                    member,
                    exclude_qid=qid if leave_one_question_out else None,
                ),
            )
        )
        reports[name]["feature_cardinality"] = lookup.feature_cardinality
    return dict(sorted(reports.items()))


def _split_report(
    pairs: Sequence[RulePair],
    lookups: Mapping[str, QuestionBalancedLookup],
    *,
    leave_one_question_out: bool,
) -> dict[str, Any]:
    by_stratum_variant: dict[tuple[str, str], list[RulePair]] = defaultdict(list)
    by_edge_reality: dict[str, list[RulePair]] = defaultdict(list)
    for pair in pairs:
        by_stratum_variant[(pair.stratum, pair.variant)].append(pair)
        by_edge_reality[_negative_edge_reality(pair)].append(pair)
    return {
        "counts": {
            "question_count": len({pair.qid for pair in pairs}),
            "pair_count": len(pairs),
        },
        "rules": _slice_report(
            pairs, lookups, leave_one_question_out=leave_one_question_out
        ),
        "by_stratum_variant": {
            f"{stratum}/{variant}": {
                "counts": {
                    "question_count": len({pair.qid for pair in selected}),
                    "pair_count": len(selected),
                },
                "rules": _slice_report(
                    selected,
                    lookups,
                    leave_one_question_out=leave_one_question_out,
                ),
            }
            for (stratum, variant), selected in sorted(by_stratum_variant.items())
        },
        "by_negative_edge_reality": {
            reality: {
                "counts": {
                    "question_count": len({pair.qid for pair in selected}),
                    "pair_count": len(selected),
                },
                "rules": _slice_report(
                    selected,
                    lookups,
                    leave_one_question_out=leave_one_question_out,
                ),
            }
            for reality, selected in sorted(by_edge_reality.items())
        },
    }


def _fit_lookups(pairs: Sequence[RulePair]) -> dict[str, QuestionBalancedLookup]:
    return {
        name: QuestionBalancedLookup(feature).fit(pairs)
        for name, feature in _LOOKUP_FEATURES.items()
    }


def build_rule_leakage_audit(
    *,
    synthetic_train: Sequence[RulePair],
    synthetic_dev: Sequence[RulePair],
    natural_train: Sequence[RulePair],
    natural_dev: Sequence[RulePair],
    gate_threshold: float = 0.65,
) -> dict[str, Any]:
    """Build a text-free leakage report with an explicit synthetic-dev gate."""

    if not 0.5 <= gate_threshold <= 1.0:
        raise RuleAuditInvariantError("gate threshold must be between 0.5 and 1")
    synthetic_lookups = _fit_lookups(synthetic_train)
    natural_lookups = _fit_lookups(natural_train)
    synthetic_train_report = _split_report(
        synthetic_train, synthetic_lookups, leave_one_question_out=True
    )
    synthetic_dev_report = _split_report(
        synthetic_dev, synthetic_lookups, leave_one_question_out=False
    )
    natural_train_report = _split_report(
        natural_train, natural_lookups, leave_one_question_out=True
    )
    natural_dev_report = _split_report(
        natural_dev, natural_lookups, leave_one_question_out=False
    )
    dev_values = {
        name: metrics["value"]
        for name, metrics in synthetic_dev_report["rules"].items()
        if metrics["value"] is not None
    }
    best_name, best_value = max(dev_values.items(), key=lambda item: (item[1], item[0]))
    return {
        "schema_version": 1,
        "content_contract": (
            "manifest metadata only; no question, answer, title, sentence, "
            "or document text"
        ),
        "lookup_protocol": (
            "question-balanced train lookup; leave-one-question-out on train; "
            "train-fitted lookup on dev; Laplace smoothing"
        ),
        "synthetic": {
            "train": synthetic_train_report,
            "dev_distractor": synthetic_dev_report,
        },
        "natural": {
            "train": natural_train_report,
            "dev_distractor": natural_dev_report,
        },
        "synthetic_dev_gate": {
            "threshold": gate_threshold,
            "best_rule": best_name,
            "best_question_macro_pairwise_accuracy": best_value,
            "passed": best_value <= gate_threshold,
        },
    }
