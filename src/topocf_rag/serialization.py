"""Leakage-resistant serialization of validated evidence topologies.

The serializer emits only a typed directed-edge list and immutable document
payloads selected upstream.  Question text, labels, generation metadata, edge
observation flags, provenance, and inferred document roles are intentionally
outside this representation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sized
from dataclasses import dataclass
from typing import Any, Final

from .topology import EvidenceTopology, TypedEdge


SERIALIZER_VERSION: Final[str] = "evidence-topology-neutral-v1"

_RELATION_ORDER: Final[dict[str, int]] = {
    "retrieval": 0,
    "title_mention": 1,
}
_DOCUMENT_ALIAS: Final[re.Pattern[str]] = re.compile(r"d(0|[1-9][0-9]*)\Z")


@dataclass(frozen=True, slots=True)
class SerializedTopology:
    """An in-memory serialization and its content identity."""

    text: str
    serializer_version: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SerializationTokenStats:
    """Token budget measured on the exact serialized string."""

    untruncated_token_count: int
    actual_token_count: int
    truncated: bool
    max_length: int


def _endpoint_order(endpoint: str) -> tuple[int, int]:
    if endpoint == "q":
        return (0, -1)
    match = _DOCUMENT_ALIAS.fullmatch(endpoint)
    if match is None:  # EvidenceTopology normally rejects this first.
        raise ValueError("topology contains a non-neutral edge endpoint")
    return (1, int(match.group(1)))


def _edge_order(edge: TypedEdge) -> tuple[int, tuple[int, int], tuple[int, int]]:
    return (
        _RELATION_ORDER[edge.relation],
        _endpoint_order(edge.source),
        _endpoint_order(edge.target),
    )


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def serialize_evidence_topology(topology: EvidenceTopology) -> SerializedTopology:
    """Serialize topology without question- or label-bearing metadata.

    Document blocks always follow their declared neutral alias order.  They are
    never reordered from an edge traversal.  ``EvidenceDocument.sentences`` is
    treated as the complete set of necessary sentences already selected by an
    upstream, label-blind policy.
    """

    if not isinstance(topology, EvidenceTopology):
        raise TypeError("topology must be an EvidenceTopology")

    lines = ["[typed_edges]"]
    for edge in sorted(topology.edges, key=_edge_order):
        # observed and provenance are deliberately not read here.
        lines.append(f"{edge.relation} {edge.source} -> {edge.target}")

    lines.append("[documents]")
    for document in topology.documents:
        lines.append(f"[{document.alias}]")
        lines.append(f"title {_json_string(document.title)}")
        for sentence_index, sentence in enumerate(document.sentences):
            lines.append(f"sentence {sentence_index} {_json_string(sentence)}")

    text = "\n".join(lines) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return SerializedTopology(
        text=text,
        serializer_version=SERIALIZER_VERSION,
        sha256=digest,
    )


def _token_count(encoded: Any) -> int:
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError("tokenizer output must map input_ids to a flat token sequence")
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, (str, bytes)) or not isinstance(input_ids, Sized):
        raise TypeError("tokenizer input_ids must be a flat token sequence")
    if len(input_ids) and isinstance(input_ids[0], (list, tuple)):
        raise TypeError("tokenizer must return an unbatched token sequence")
    return len(input_ids)


def count_serialization_tokens(
    serialization: SerializedTopology,
    tokenizer: Any,
    *,
    max_length: int,
) -> SerializationTokenStats:
    """Measure full and effective lengths with a Hugging Face tokenizer.

    Both calls use the exact same serialization, special-token policy, and no
    padding.  The second call differs only by enabling truncation at
    ``max_length``.
    """

    if not isinstance(serialization, SerializedTopology):
        raise TypeError("serialization must be a SerializedTopology")
    if not callable(tokenizer):
        raise TypeError("tokenizer must be callable")
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise ValueError("max_length must be a positive integer")

    common_options = {
        "add_special_tokens": True,
        "padding": False,
        "return_attention_mask": False,
        "return_token_type_ids": False,
    }
    full = tokenizer(
        serialization.text,
        truncation=False,
        **common_options,
    )
    effective = tokenizer(
        serialization.text,
        truncation=True,
        max_length=max_length,
        **common_options,
    )
    untruncated_token_count = _token_count(full)
    actual_token_count = _token_count(effective)
    if actual_token_count > untruncated_token_count:
        raise ValueError("truncated token count cannot exceed the full token count")
    return SerializationTokenStats(
        untruncated_token_count=untruncated_token_count,
        actual_token_count=actual_token_count,
        truncated=actual_token_count < untruncated_token_count,
        max_length=max_length,
    )
