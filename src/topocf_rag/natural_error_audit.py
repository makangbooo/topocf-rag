"""Controlled retrieval-failure audit for certificate-v1.

This module deliberately separates a cheap, deterministic screening proxy from
the human-validated G0 motivation gate.  It ranks only the official HotpotQA
context documents supplied with each question; it does not claim full-corpus or
end-to-end RAG retrieval realism.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence

from .graph import supporting_title_order, validate_hotpot_example
from .title_normalization import normalize_title


AUDIT_SCHEMA_VERSION = 2
AUDIT_SCOPE = "hotpot_official_context_pool_top_k"
TARGET_PROXY = "exactly_one_gold_support_document_retrieved"


class NaturalErrorAuditInvariantError(ValueError):
    """Raised when an audit input violates the frozen screening contract."""


@dataclass(frozen=True, slots=True)
class TopKOutcome:
    """Question-level outcome for one deterministic retrieval budget."""

    top_k: int
    retrieved_indices: tuple[int, ...]
    gold_support_document_count: int
    retained_supporting_fact_count: int
    answer_string_check_eligible: bool
    answer_string_present: bool | None

    @property
    def complete(self) -> bool:
        return self.gold_support_document_count == 2

    @property
    def failure(self) -> bool:
        return not self.complete

    @property
    def partial_gold_evidence_proxy(self) -> bool:
        return (
            self.gold_support_document_count == 1
            and self.retained_supporting_fact_count >= 1
        )


@dataclass(frozen=True, slots=True)
class QuestionAudit:
    """Internal question record; text never enters the public report."""

    qid: str
    ranked_indices: tuple[int, ...]
    retrieval_scores: tuple[float, ...]
    gold_support_indices: tuple[int, int]
    outcomes: tuple[TopKOutcome, ...]

    def outcome_for(self, top_k: int) -> TopKOutcome:
        for outcome in self.outcomes:
            if outcome.top_k == top_k:
                return outcome
        raise NaturalErrorAuditInvariantError(
            f"question audit does not contain top_k={top_k}"
        )


def _normalized_search_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _answer_occurrence_eligible(answer: str) -> bool:
    normalized = _normalized_search_text(answer)
    return len(normalized) >= 3 and normalized not in {"yes", "no"}


def _validate_top_ks(top_ks: Sequence[int], primary_top_k: int) -> tuple[int, ...]:
    if not top_ks:
        raise NaturalErrorAuditInvariantError("top_ks must not be empty")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in top_ks
    ):
        raise NaturalErrorAuditInvariantError("top_ks must contain positive integers")
    normalized = tuple(sorted(set(top_ks)))
    if len(normalized) != len(top_ks):
        raise NaturalErrorAuditInvariantError("top_ks must be unique")
    if primary_top_k not in normalized:
        raise NaturalErrorAuditInvariantError("primary_top_k must occur in top_ks")
    return normalized


def audit_question(
    example: Mapping[str, Any],
    retrieval_scores: Sequence[float],
    *,
    top_ks: Sequence[int],
    primary_top_k: int,
) -> QuestionAudit:
    """Audit one bridge question using stable score/index ranking."""

    validate_hotpot_example(example)
    if example.get("type") != "bridge":
        raise NaturalErrorAuditInvariantError("audit requires bridge questions")
    qid = example.get("_id")
    if not isinstance(qid, str) or not qid:
        raise NaturalErrorAuditInvariantError("question ID must be a non-empty string")

    normalized_top_ks = _validate_top_ks(top_ks, primary_top_k)
    context = example["context"]
    if len(retrieval_scores) != len(context):
        raise NaturalErrorAuditInvariantError(
            "retrieval score count must equal context document count"
        )
    scores: list[float] = []
    for score in retrieval_scores:
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise NaturalErrorAuditInvariantError("retrieval scores must be numeric")
        numeric = float(score)
        if not math.isfinite(numeric):
            raise NaturalErrorAuditInvariantError("retrieval scores must be finite")
        scores.append(numeric)

    support_titles = supporting_title_order(example)
    if len(support_titles) != 2:
        raise NaturalErrorAuditInvariantError(
            "controlled bridge audit requires exactly two supporting documents"
        )
    context_index_by_title = {
        normalize_title(document[0]): index for index, document in enumerate(context)
    }
    try:
        support_indices = tuple(context_index_by_title[title] for title in support_titles)
    except KeyError as error:
        raise NaturalErrorAuditInvariantError(
            "supporting document is missing from official context"
        ) from error
    if len(set(support_indices)) != 2:
        raise NaturalErrorAuditInvariantError(
            "supporting documents must map to distinct context indices"
        )

    support_fact_count_by_index: dict[int, int] = {}
    for title, _sentence_index in example["supporting_facts"]:
        context_index = context_index_by_title[normalize_title(title)]
        support_fact_count_by_index[context_index] = (
            support_fact_count_by_index.get(context_index, 0) + 1
        )

    ranked_indices = tuple(
        sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    )
    answer = str(example.get("answer", ""))
    answer_eligible = _answer_occurrence_eligible(answer)
    normalized_answer = _normalized_search_text(answer)

    outcomes: list[TopKOutcome] = []
    for top_k in normalized_top_ks:
        retrieved = ranked_indices[: min(top_k, len(ranked_indices))]
        retrieved_set = set(retrieved)
        retained_support_indices = retrieved_set.intersection(support_indices)
        retained_fact_count = sum(
            support_fact_count_by_index.get(index, 0)
            for index in retained_support_indices
        )
        answer_present: bool | None = None
        if answer_eligible:
            retrieved_text = " ".join(
                " ".join((str(context[index][0]), *map(str, context[index][1])))
                for index in retrieved
            )
            answer_present = normalized_answer in _normalized_search_text(retrieved_text)
        outcomes.append(
            TopKOutcome(
                top_k=top_k,
                retrieved_indices=tuple(retrieved),
                gold_support_document_count=len(retained_support_indices),
                retained_supporting_fact_count=retained_fact_count,
                answer_string_check_eligible=answer_eligible,
                answer_string_present=answer_present,
            )
        )

    return QuestionAudit(
        qid=qid,
        ranked_indices=ranked_indices,
        retrieval_scores=tuple(scores),
        gold_support_indices=(support_indices[0], support_indices[1]),
        outcomes=tuple(outcomes),
    )


def wilson_interval(success_count: int, total_count: int) -> dict[str, float] | None:
    """Return a two-sided 95% Wilson interval for a binomial proportion."""

    if total_count == 0:
        return None
    if not 0 <= success_count <= total_count:
        raise NaturalErrorAuditInvariantError("invalid binomial counts")
    z = 1.959963984540054
    proportion = success_count / total_count
    denominator = 1.0 + (z * z / total_count)
    center = (proportion + z * z / (2.0 * total_count)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total_count
            + z * z / (4.0 * total_count * total_count)
        )
        / denominator
    )
    return {"low": max(0.0, center - radius), "high": min(1.0, center + radius)}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _aggregate_outcomes(
    audits: Sequence[QuestionAudit], top_k: int
) -> dict[str, Any]:
    outcomes = [audit.outcome_for(top_k) for audit in audits]
    complete_count = sum(outcome.complete for outcome in outcomes)
    partial_count = sum(
        outcome.partial_gold_evidence_proxy for outcome in outcomes
    )
    no_gold_count = sum(
        outcome.gold_support_document_count == 0 for outcome in outcomes
    )
    failure_count = len(outcomes) - complete_count
    answer_eligible = [
        outcome for outcome in outcomes if outcome.answer_string_check_eligible
    ]
    answer_present_count = sum(
        outcome.answer_string_present is True for outcome in answer_eligible
    )

    def answer_summary(subset: Sequence[TopKOutcome]) -> dict[str, Any]:
        eligible_subset = [
            outcome for outcome in subset if outcome.answer_string_check_eligible
        ]
        present_count = sum(
            outcome.answer_string_present is True for outcome in eligible_subset
        )
        return {
            "eligible_count": len(eligible_subset),
            "present_count": present_count,
            "absent_count": len(eligible_subset) - present_count,
            "present_rate": _rate(present_count, len(eligible_subset)),
        }

    return {
        "counts": {
            "eligible_question_count": len(outcomes),
            "complete_gold_evidence_count": complete_count,
            "retrieval_failure_count": failure_count,
            "partial_gold_evidence_proxy_count": partial_count,
            "no_gold_support_document_count": no_gold_count,
            "answer_string_check_eligible_count": len(answer_eligible),
            "answer_string_present_count": answer_present_count,
        },
        "rates": {
            "complete_gold_evidence_rate": _rate(complete_count, len(outcomes)),
            "retrieval_failure_rate": _rate(failure_count, len(outcomes)),
            "partial_proxy_rate_among_retrieval_failures": _rate(
                partial_count, failure_count
            ),
            "no_gold_rate_among_retrieval_failures": _rate(
                no_gold_count, failure_count
            ),
            "answer_string_present_rate_when_check_eligible": _rate(
                answer_present_count, len(answer_eligible)
            ),
        },
        "partial_proxy_wilson_95": wilson_interval(partial_count, failure_count),
        "answer_string_by_retrieval_status": {
            "all": answer_summary(outcomes),
            "complete_gold_evidence": answer_summary(
                [outcome for outcome in outcomes if outcome.complete]
            ),
            "retrieval_failure": answer_summary(
                [outcome for outcome in outcomes if outcome.failure]
            ),
            "partial_gold_evidence_proxy": answer_summary(
                [
                    outcome
                    for outcome in outcomes
                    if outcome.partial_gold_evidence_proxy
                ]
            ),
            "no_gold_support_document": answer_summary(
                [
                    outcome
                    for outcome in outcomes
                    if outcome.gold_support_document_count == 0
                ]
            ),
        },
    }


def _stable_failure_sample(
    audits: Sequence[QuestionAudit],
    *,
    primary_top_k: int,
    sample_size: int,
    sample_seed: int,
) -> tuple[QuestionAudit, ...]:
    if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size < 1:
        raise NaturalErrorAuditInvariantError("sample_size must be a positive integer")
    if not isinstance(sample_seed, int) or isinstance(sample_seed, bool):
        raise NaturalErrorAuditInvariantError("sample_seed must be an integer")
    failures = [
        audit for audit in audits if audit.outcome_for(primary_top_k).failure
    ]

    def sample_key(audit: QuestionAudit) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"{sample_seed}\0{audit.qid}".encode("utf-8")
        ).hexdigest()
        return digest, audit.qid

    return tuple(sorted(failures, key=sample_key)[:sample_size])


def _selection_sha256(audits: Sequence[QuestionAudit]) -> str:
    encoded = json.dumps(
        [audit.qid for audit in audits],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_public_audit_report(
    audits: Sequence[QuestionAudit],
    *,
    audit_id: str,
    dataset: str,
    official_split: str,
    top_ks: Sequence[int],
    primary_top_k: int,
    sample_size: int,
    sample_seed: int,
    screening_threshold: float,
    source_sha256: str,
    ids_sha256: str,
    retrieval_cache_sha256: str,
    config_sha256: str,
) -> tuple[dict[str, Any], tuple[QuestionAudit, ...]]:
    """Build an aggregate-only public report and the deterministic sample IDs."""

    normalized_top_ks = _validate_top_ks(top_ks, primary_top_k)
    if not audits:
        raise NaturalErrorAuditInvariantError("audit question list must not be empty")
    if len({audit.qid for audit in audits}) != len(audits):
        raise NaturalErrorAuditInvariantError("audit question IDs must be unique")
    if not 0.0 <= screening_threshold <= 1.0:
        raise NaturalErrorAuditInvariantError(
            "screening_threshold must lie in [0, 1]"
        )
    sample = _stable_failure_sample(
        audits,
        primary_top_k=primary_top_k,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )
    by_top_k = {
        str(top_k): _aggregate_outcomes(audits, top_k)
        for top_k in normalized_top_ks
    }
    primary_value = by_top_k[str(primary_top_k)]["rates"][
        "partial_proxy_rate_among_retrieval_failures"
    ]
    passed = primary_value is not None and primary_value >= screening_threshold
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": audit_id,
        "audit_scope": AUDIT_SCOPE,
        "dataset": dataset,
        "official_split": official_split,
        "content_contract": (
            "aggregate numeric metadata and cryptographic hashes only; no question, "
            "answer, title, sentence, document text, or question IDs"
        ),
        "scope_warning": (
            "Scores rank only the official HotpotQA context pool. This controlled "
            "screening audit is not full-corpus retrieval and is not an end-to-end "
            "RAG failure evaluation."
        ),
        "target_proxy": TARGET_PROXY,
        "input_hashes": {
            "source": source_sha256,
            "frozen_ids": ids_sha256,
            "retrieval_cache": retrieval_cache_sha256,
            "config": config_sha256,
        },
        "protocol": {
            "ranking": "descending retrieval score; context index breaks exact ties",
            "top_ks": list(normalized_top_ks),
            "primary_top_k": primary_top_k,
            "failure_definition": "fewer than two gold supporting documents retrieved",
            "proxy_definition": (
                "exactly one of two gold supporting documents retrieved and at least "
                "one annotated supporting fact retained"
            ),
        },
        "by_top_k": by_top_k,
        "screening_gate": {
            "name": "pre-human-audit candidate prevalence screen",
            "formal_g0": False,
            "metric": "partial_proxy_rate_among_retrieval_failures",
            "threshold": screening_threshold,
            "value": primary_value,
            "passed": passed,
            "status": (
                "proceed_to_human_validation"
                if passed
                else "stop_or_redesign_before_human_validation"
            ),
            "interpretation": (
                "Passing authorizes private human or independent-judge validation "
                "only; it does not establish formal G0 or natural full-corpus "
                "retrieval prevalence."
            ),
        },
        "private_sample": {
            "selection_policy": "ascending sha256(sample_seed NUL question_id)",
            "sample_seed": sample_seed,
            "requested_size": sample_size,
            "actual_size": len(sample),
            "question_id_sequence_sha256": _selection_sha256(sample),
            "must_not_be_committed": True,
        },
    }
    return report, sample


def build_private_sample_record(
    example: Mapping[str, Any],
    audit: QuestionAudit,
    *,
    primary_top_k: int,
) -> dict[str, Any]:
    """Create one text-bearing adjudication record for private storage only."""

    if str(example.get("_id")) != audit.qid:
        raise NaturalErrorAuditInvariantError("example and audit question IDs disagree")
    context = example["context"]
    outcome = audit.outcome_for(primary_top_k)
    if not outcome.failure:
        raise NaturalErrorAuditInvariantError(
            "private sample may contain only primary-top-k retrieval failures"
        )
    rank_by_index = {
        context_index: rank
        for rank, context_index in enumerate(audit.ranked_indices, start=1)
    }
    retrieved_documents = []
    for context_index in outcome.retrieved_indices:
        title, sentences = context[context_index]
        retrieved_documents.append(
            {
                "context_index": context_index,
                "retrieval_rank": rank_by_index[context_index],
                "retrieval_score": audit.retrieval_scores[context_index],
                "title": title,
                "sentences": list(sentences),
            }
        )
    retrieved_set = set(outcome.retrieved_indices)
    gold_facts = []
    context_index_by_title = {
        normalize_title(document[0]): index for index, document in enumerate(context)
    }
    for title, sentence_index in example["supporting_facts"]:
        context_index = context_index_by_title[normalize_title(title)]
        gold_facts.append(
            {
                "context_index": context_index,
                "retrieved": context_index in retrieved_set,
                "title": context[context_index][0],
                "sentence_index": sentence_index,
                "sentence": context[context_index][1][sentence_index],
            }
        )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "privacy": "private_text_bearing_artifact_do_not_commit",
        "question_id": audit.qid,
        "question": example["question"],
        "answer": example["answer"],
        "primary_top_k": primary_top_k,
        "proxy": {
            "gold_support_document_count": outcome.gold_support_document_count,
            "retained_supporting_fact_count": outcome.retained_supporting_fact_count,
            "partial_gold_evidence_proxy": outcome.partial_gold_evidence_proxy,
            "answer_string_check_eligible": outcome.answer_string_check_eligible,
            "answer_string_present": outcome.answer_string_present,
        },
        "retrieved_documents": retrieved_documents,
        "gold_supporting_facts_for_adjudication": gold_facts,
        "human_annotation": {
            "label": None,
            "allowed_labels": [
                "locally_true_globally_incomplete",
                "missing_all_gold_evidence",
                "complete_via_alternative_proof",
                "ambiguous_or_annotation_issue",
                "other",
            ],
            "retrieved_local_facts_true": None,
            "globally_sufficient": None,
            "alternative_proof_present": None,
            "notes": "",
        },
    }
