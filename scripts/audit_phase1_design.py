#!/usr/bin/env python3
"""Audit Phase 1 perturbation feasibility without emitting dataset content.

The report contains only aggregate counts, definitions, and input provenance.
Questions, answers, titles, sentences, supporting facts, and selected IDs are
used only in memory and are never written or printed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Hashable, Iterable, Mapping, Sequence

import ijson

from topocf_rag.graph import build_query_document_graph


REPORT_SCHEMA_VERSION = 1
DEFAULT_RETRIEVAL_TOP_K = 10

DirectedEdge = tuple[int, int]
TypedEdge = tuple[str, Hashable, Hashable]


@dataclass(frozen=True, slots=True)
class QuestionAudit:
    """Content-free feasibility counts for one frozen question."""

    eligible_supporting_title_pair: int = 0
    gold_no_mention_edge: int = 0
    gold_asymmetric_mention_edge: int = 0
    gold_bidirectional_mention_edge: int = 0
    t1_operations: int = 0
    t2_operations: int = 0
    t3_disjoint_original_edge_pairs: int = 0
    t3_synthetic_operations: int = 0
    t3_natural_operations: int = 0
    t3_partially_real_rejected: int = 0


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("ids") if isinstance(payload, Mapping) else None
    if not isinstance(ids, list) or any(
        not isinstance(question_id, str) or not question_id for question_id in ids
    ):
        raise ValueError("frozen ID manifest must contain a list of non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("frozen ID manifest contains duplicate IDs")
    return set(ids)


def load_retrieval_records(
    path: Path, *, expected_ids_sha256: str
) -> Mapping[str, Mapping[str, Sequence[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("retrieval cache must be a JSON object")
    if payload.get("ids_sha256") != expected_ids_sha256:
        raise ValueError("retrieval cache does not match the frozen ID manifest")
    records = payload.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("retrieval cache must contain a records object")
    return records


def canonical_typed_edge_signature(edges: Iterable[TypedEdge]) -> str:
    """Canonicalize a small typed directed graph while keeping ``q`` fixed."""

    edge_tuple = tuple(edges)
    document_nodes = sorted(
        {
            node
            for _relation, source, target in edge_tuple
            for node in (source, target)
            if node != "q"
        },
        key=repr,
    )
    signatures: list[str] = []
    for permutation in itertools.permutations(range(len(document_nodes))):
        renamed: dict[Hashable, str] = {
            node: f"d{label}"
            for node, label in zip(document_nodes, permutation, strict=True)
        }

        def label(node: Hashable) -> str:
            return "q" if node == "q" else renamed[node]

        canonical_edges = sorted(
            (relation, label(source), label(target))
            for relation, source, target in edge_tuple
        )
        signatures.append(
            "|".join(
                f"{relation}:{source}>{target}"
                for relation, source, target in canonical_edges
            )
        )
    return min(signatures, default="")


def canonical_overlap_audit() -> dict[str, Any]:
    # T1 keeps q->u and reverses u->v. T2 keeps u->v and moves q's anchor.
    t1_negative = canonical_typed_edge_signature(
        (("retrieval", "q", "u"), ("title_mention", "v", "u"))
    )
    t2_negative = canonical_typed_edge_signature(
        (("retrieval", "q", "v"), ("title_mention", "u", "v"))
    )
    t3_positive = canonical_typed_edge_signature(
        (
            ("retrieval", "q", "u"),
            ("retrieval", "q", "a"),
            ("title_mention", "u", "v"),
            ("title_mention", "a", "b"),
        )
    )
    t3_negative = canonical_typed_edge_signature(
        (
            ("retrieval", "q", "u"),
            ("retrieval", "q", "a"),
            ("title_mention", "u", "b"),
            ("title_mention", "a", "v"),
        )
    )
    return {
        "node_labels_ignored": "document labels only; q and relation types are fixed",
        "t1_negative_signature": t1_negative,
        "t2_negative_signature": t2_negative,
        "signatures_overlap": t1_negative == t2_negative,
        "t1_t2_implication": (
            "T1 and T2 share the same typed unlabeled collider topology, so holding "
            "one out does not test an unseen topology family"
        ),
        "t3_positive_signature": t3_positive,
        "t3_negative_signature": t3_negative,
        "t3_signatures_overlap": t3_positive == t3_negative,
        "t3_implication": (
            "The degree-preserving 2-switch keeps the same typed unlabeled "
            "two-branch topology; its counterfactual signal depends on fixed "
            "document identities or attributes rather than an unlabeled graph motif"
        ),
    }


def audit_directed_edges(
    mention_edges: Iterable[DirectedEdge],
    *,
    gold_indices: tuple[int, int],
    retrieved_indices: Iterable[int],
) -> QuestionAudit:
    """Audit T1/T2/T3 operations for one content-free directed graph.

    Each mention edge is reduced to its endpoint pair, so repeated evidence
    sentences cannot duplicate an operation. A T3 operation orders the gold
    edge first and swaps targets with one vertex-disjoint real mention edge.
    """

    edges = set(mention_edges)
    retrieved = set(retrieved_indices)
    first, second = gold_indices
    if first == second:
        raise ValueError("gold document indices must be distinct")

    forward = (first, second) in edges
    reverse = (second, first) in edges
    if not (forward or reverse):
        return QuestionAudit(
            eligible_supporting_title_pair=1,
            gold_no_mention_edge=1,
        )
    if forward and reverse:
        return QuestionAudit(
            eligible_supporting_title_pair=1,
            gold_bidirectional_mention_edge=1,
        )

    gold_edge = (first, second) if forward else (second, first)
    gold_source, gold_target = gold_edge
    both_gold_retrieved = {gold_source, gold_target}.issubset(retrieved)

    disjoint_pairs = 0
    synthetic = 0
    natural = 0
    partially_real = 0
    for other_source, other_target in sorted(edges):
        other_edge = (other_source, other_target)
        if other_edge == gold_edge:
            continue
        if len({gold_source, gold_target, other_source, other_target}) != 4:
            continue
        if gold_source not in retrieved or other_source not in retrieved:
            continue
        disjoint_pairs += 1
        switched_first = (gold_source, other_target)
        switched_second = (other_source, gold_target)
        switched_real_count = int(switched_first in edges) + int(
            switched_second in edges
        )
        if switched_real_count == 0:
            synthetic += 1
        elif switched_real_count == 2:
            natural += 1
        else:
            partially_real += 1

    return QuestionAudit(
        eligible_supporting_title_pair=1,
        gold_asymmetric_mention_edge=1,
        t1_operations=int(both_gold_retrieved),
        t2_operations=int(both_gold_retrieved),
        t3_disjoint_original_edge_pairs=disjoint_pairs,
        t3_synthetic_operations=synthetic,
        t3_natural_operations=natural,
        t3_partially_real_rejected=partially_real,
    )


class SplitAccumulator:
    def __init__(self) -> None:
        self.question_count = 0
        self.totals = {field: 0 for field in QuestionAudit.__dataclass_fields__}
        self.questions_with = {
            "t1": 0,
            "t2": 0,
            "t3_disjoint_original_edge_pair": 0,
            "t3_synthetic": 0,
            "t3_natural": 0,
            "t3_partially_real_rejected": 0,
        }

    def add(self, audit: QuestionAudit) -> None:
        self.question_count += 1
        values = asdict(audit)
        for key, value in values.items():
            self.totals[key] += value
        self.questions_with["t1"] += int(audit.t1_operations > 0)
        self.questions_with["t2"] += int(audit.t2_operations > 0)
        self.questions_with["t3_disjoint_original_edge_pair"] += int(
            audit.t3_disjoint_original_edge_pairs > 0
        )
        self.questions_with["t3_synthetic"] += int(
            audit.t3_synthetic_operations > 0
        )
        self.questions_with["t3_natural"] += int(audit.t3_natural_operations > 0)
        self.questions_with["t3_partially_real_rejected"] += int(
            audit.t3_partially_real_rejected > 0
        )

    def report(self) -> dict[str, Any]:
        return {
            "question_count": self.question_count,
            "gold_mention_structure": {
                "eligible_two_supporting_title_questions": self.totals[
                    "eligible_supporting_title_pair"
                ],
                "no_edge_questions": self.totals["gold_no_mention_edge"],
                "asymmetric_edge_count": self.totals[
                    "gold_asymmetric_mention_edge"
                ],
                "asymmetric_edge_questions": self.totals[
                    "gold_asymmetric_mention_edge"
                ],
                "bidirectional_questions": self.totals[
                    "gold_bidirectional_mention_edge"
                ],
            },
            "t1": {
                "operations": self.totals["t1_operations"],
                "questions": self.questions_with["t1"],
            },
            "t2": {
                "operations": self.totals["t2_operations"],
                "questions": self.questions_with["t2"],
            },
            "t3_four_document_subgraph": {
                "disjoint_original_edge_pairs": {
                    "operations": self.totals[
                        "t3_disjoint_original_edge_pairs"
                    ],
                    "questions": self.questions_with[
                        "t3_disjoint_original_edge_pair"
                    ],
                },
                "synthetic_both_switched_edges_absent": {
                    "operations": self.totals["t3_synthetic_operations"],
                    "questions": self.questions_with["t3_synthetic"],
                },
                "natural_both_switched_edges_real": {
                    "operations": self.totals["t3_natural_operations"],
                    "questions": self.questions_with["t3_natural"],
                },
                "rejected_exactly_one_switched_edge_real": {
                    "operations": self.totals["t3_partially_real_rejected"],
                    "questions": self.questions_with[
                        "t3_partially_real_rejected"
                    ],
                },
            },
        }


def audit_split(
    *,
    source_path: Path,
    ids_path: Path,
    retrieval_cache_path: Path,
    retrieval_top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ids_sha256 = sha256_file(ids_path)
    wanted = load_frozen_ids(ids_path)
    retrieval_records = load_retrieval_records(
        retrieval_cache_path, expected_ids_sha256=ids_sha256
    )
    accumulator = SplitAccumulator()
    seen: set[str] = set()

    with source_path.open("rb") as stream:
        for record in ijson.items(stream, "item"):
            question_id = record.get("_id") if isinstance(record, Mapping) else None
            if question_id not in wanted:
                continue
            if question_id in seen:
                raise ValueError("source contains a duplicate frozen ID")
            seen.add(question_id)
            if record.get("type") != "bridge":
                raise ValueError("frozen split contains a non-bridge record")

            retrieval_record = retrieval_records.get(question_id)
            if not isinstance(retrieval_record, Mapping):
                raise ValueError("retrieval cache is missing a frozen record")
            scores = retrieval_record.get("scores")
            if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
                raise ValueError("retrieval cache scores must be a sequence")
            numeric_scores = [float(score) for score in scores]
            top_k = min(retrieval_top_k, len(record["context"]))
            graph = build_query_document_graph(
                record, numeric_scores, retrieval_top_k=top_k
            )
            if len(graph.supporting_title_order) != 2:
                accumulator.add(QuestionAudit())
                continue
            index_by_title = {
                document.normalized_title: document.index
                for document in graph.documents
            }
            gold_indices = tuple(
                index_by_title[title] for title in graph.supporting_title_order
            )
            mention_edges = (
                (edge.source_index, edge.target_index) for edge in graph.mention_edges
            )
            retrieved_indices = (
                edge.document_index for edge in graph.retrieval_edges
            )
            accumulator.add(
                audit_directed_edges(
                    mention_edges,
                    gold_indices=(gold_indices[0], gold_indices[1]),
                    retrieved_indices=retrieved_indices,
                )
            )

    missing_count = len(wanted.difference(seen))
    if missing_count:
        raise ValueError(f"source is missing {missing_count} frozen records")
    if accumulator.question_count != len(wanted):
        raise RuntimeError("audited question count does not match frozen split size")

    provenance = {
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "ids_path": str(ids_path.resolve()),
        "ids_sha256": ids_sha256,
        "frozen_question_count": len(wanted),
        "retrieval_top_k": retrieval_top_k,
    }
    return accumulator.report(), provenance


def build_report(
    *,
    split_inputs: Mapping[str, tuple[Path, Path, Path]],
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> dict[str, Any]:
    if retrieval_top_k < 1:
        raise ValueError("retrieval_top_k must be positive")

    split_reports: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for split_name, (source_path, ids_path, retrieval_cache_path) in split_inputs.items():
        split_report, split_provenance = audit_split(
            source_path=source_path,
            ids_path=ids_path,
            retrieval_cache_path=retrieval_cache_path,
            retrieval_top_k=retrieval_top_k,
        )
        split_reports[split_name] = split_report
        provenance[split_name] = split_provenance

    overlap = canonical_overlap_audit()
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "inputs": provenance,
        "definitions": {
            "t1": (
                "exactly one real directed mention edge exists between the two gold "
                "documents; both endpoints have real top-k retrieval edges; the "
                "counterfactual reverses the mention direction"
            ),
            "t2": (
                "for asymmetric real gold edge u->v, positive edges are q->u and "
                "u->v; the counterfactual keeps u->v and moves the real top-k "
                "retrieval anchor to q->v"
            ),
            "t3_synthetic": (
                "combine asymmetric gold edge u->v with a real mention edge a->b "
                "over four distinct documents; q->u and q->a must both be real "
                "top-k retrieval anchors; target-swap to u->b and a->v, both of "
                "which must be absent"
            ),
            "t3_natural": (
                "the same four-document target-swap, but both switched mention "
                "edges already exist in the unmodified graph"
            ),
            "operation_deduplication": (
                "directed endpoint pairs are sets; one gold-edge/other-edge pair is "
                "counted once regardless of the number of mentioning sentences"
            ),
        },
        "splits": split_reports,
        "single_path_t3": {
            "operations": 0,
            "questions": 0,
            "definition": (
                "a q->d_i->d_j scoring unit has only two document nodes and one "
                "mention edge, while a degree-preserving 2-switch requires two "
                "edges over four distinct document nodes"
            ),
        },
        "canonical_topology_audit": overlap,
        "design_gate": {
            "passed": False,
            "status": "blocked_before_formal_pair_generation_and_lopo",
            "blocked_actions": [
                "formal_synthetic_pair_generation",
                "leave_one_perturbation_out_training_or_evaluation",
            ],
            "reasons": [
                "T1 and T2 have overlapping canonical typed negative topology",
                "T3 is incompatible with the current two-document single-path unit",
                (
                    "four-document T3 and current two-document T1/T2 require "
                    "different verifier input schemas"
                ),
            ],
        },
    }


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-source",
        type=Path,
        default=Path("/file_system/datasets/hotpotqa/hotpot_train_v1.1.json"),
    )
    parser.add_argument(
        "--train-ids",
        type=Path,
        default=Path("data/splits/hotpot_train_bridge_1000_ids.json"),
    )
    parser.add_argument(
        "--train-retrieval-cache",
        type=Path,
        default=Path("reports/title_graph/cache/train_bge_retrieval.json"),
    )
    parser.add_argument(
        "--dev-source",
        type=Path,
        default=Path("/file_system/datasets/hotpotqa/hotpot_dev_distractor_v1.json"),
    )
    parser.add_argument(
        "--dev-ids",
        type=Path,
        default=Path("data/splits/hotpot_dev_bridge_500_ids.json"),
    )
    parser.add_argument(
        "--dev-retrieval-cache",
        type=Path,
        default=Path("reports/title_graph/cache/dev_distractor_bge_retrieval.json"),
    )
    parser.add_argument(
        "--retrieval-top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/phase1/design_audit.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        split_inputs={
            "train": (
                args.train_source,
                args.train_ids,
                args.train_retrieval_cache,
            ),
            "dev_distractor": (
                args.dev_source,
                args.dev_ids,
                args.dev_retrieval_cache,
            ),
        },
        retrieval_top_k=args.retrieval_top_k,
    )
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "design_gate": report["design_gate"]["status"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
