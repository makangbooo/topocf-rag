# Phase 1 Design Gate

Date: 2026-07-12

## Status

Phase 1a is a representation and perturbation-design gate. Its purpose is to
determine whether T1, T2, and T3 are distinct, implementable topology
interventions under the frozen matching contract before materializing formal
pairs, scoring them with bge-m3, or training a verifier.

This gate is not a verifier experiment. No formal Phase 1 pairs have been
frozen, no Phase 1 bge-m3 baseline has been run, and no verifier has been
trained.

## Formal objects

Let `R(q,A)` be a directed retrieval edge from the implicit question node `q`
to document `A`. Let `M(A,B)` be a directed title-mention edge from document
`A` to document `B`. Document aliases are neutral and carry no bridge, answer,
source, or target role.

For an observed asymmetric gold connection, the two-document positive is:

```text
Positive = {R(q,A), M(A,B)}
```

The title-mention direction is determined by the observed edge, not by the
first-appearance order of HotpotQA supporting facts.

### T1: title-mention direction reversal

T1 keeps the retrieval edge fixed and reverses only the asymmetric mention
edge:

```text
T1 = {R(q,A), M(B,A)}
```

Eligibility requires observed `M(A,B)` and absent `M(B,A)`. The reversed edge
is synthetic. The result is the collider `q -> A <- B`, not a traversable
directed path.

### T2: query-entry role swap

T2 moves the query entry edge to the other document and keeps the real mention
edge fixed:

```text
T2 = {R(q,B), M(A,B)}
```

The result is the collider `q -> B <- A`. Both component edges are observed:
`R(q,B)` is produced by retrieval and `M(A,B)` is present in document text.
T2 is therefore the most important hard negative in the current design: each
edge is real, but the directed edges do not compose into a question-rooted
reasoning chain.

### T3: directed two-switch

A standard non-self-loop directed two-switch cannot be performed on the two
adjacent edges of one `q -> A -> B` path. Switching their heads produces
`q -> B` and the self-loop `A -> A`. The number of valid single-path T3
operations is exactly zero.

The minimum clean, relation-preserving representation uses four distinct
documents and two question-rooted paths:

```text
T3 positive = {
  R(q,A), R(q,C),
  M(A,B), M(C,D)
}

T3 switched = {
  R(q,A), R(q,C),
  M(A,D), M(C,B)
}
```

`A`, `B`, `C`, and `D` must be distinct. This switch preserves the document
set, edge count, relation multiset, and every node's per-relation directed
in/out degree.

Synthetic T3 requires both switched mention edges to be absent from the
observed graph. The stronger natural-switched stratum requires both switched
mention edges to be observed and changes only which real edges are selected as
the evidence topology.

## Feasibility counts

Counts use the frozen 1,000 train and 500 dev bridge questions. `Operations`
counts eligible perturbation instances; `Questions` counts distinct question
IDs with at least one instance.

| Perturbation stratum | Train operations | Train questions | Dev operations | Dev questions |
|---|---:|---:|---:|---:|
| T1 asymmetric gold | 747 | 747 | 340 | 340 |
| T2 query-entry swap | 747 | 747 | 340 | 340 |
| T3 synthetic four-document switch | 450 | 184 | 187 | 74 |
| T3 natural four-document switch | 65 | 27 | 28 | 12 |
| T3 single two-document path | 0 | 0 | 0 | 0 |

The natural T3 stratum is too small to support a primary held-out claim. The
synthetic dev T3 stratum covers only 74 questions, so any later result must use
question-level macro averaging and question-level bootstrap confidence
intervals rather than treating 187 operations as independent samples.

## Canonical topology audit

The canonical colored-graph signature fixes `q`, treats documents as same-color
nodes, preserves edge direction and relation type, and ignores document aliases,
edge observation flags, and provenance.

Under this definition:

- T1 and T2 have the same collider signature after swapping `A` and `B`.
- Both T1 and T2 differ from the positive chain signature.
- T3 positive and T3 switched have the same unlabeled signature: each is two
  disjoint `q`-rooted two-hop chains. T3 changes which immutable document texts
  occupy the endpoints, not the unlabeled graph motif.

Consequently, the three generators do not define three disjoint topology
classes. Holding out one generator does not establish generalization to an
unseen canonical topology. Calling the current protocol
`leave-one-perturbation-out` would overstate what is held out.

## Gate decision

Formal pair materialization, Phase 1 bge-m3 scoring, verifier training, and the
planned LOPO experiment remain paused for the following reasons:

1. The two-document `CandidatePath` representation cannot encode T1 or the
   all-real T2 collider without pretending that a non-path subgraph is a path.
2. T1 and T2 overlap canonically, so a held-out result can reflect a motif seen
   during training rather than a new topology.
3. T3 is impossible on one two-edge path and requires a different four-document
   evidence budget.
4. Synthetic T1 and T3 can be solved by checking edge-to-text consistency. They
   must not be allowed to hide performance on the all-observed T2 and natural
   error strata.
5. Serialization and exact token matching cannot be frozen until one common
   evidence object and budget are selected.

## Decision options

### Option A: unified EvidenceTopology (recommended)

Use a common four-document, two-path `EvidenceTopology` budget for all formal
pairs. Embed the T1 or T2 intervention in one path component and retain a fixed
control path in the other component. Use the four-document switch directly for
T3. Keep the question, immutable document-text multiset, document declaration
order, edge count, relation multiset, and exact-token difference within the
preregistered limit.

Rename the evaluation to `leave-one-generator-out`. Report results separately
for canonical-signature overlap versus non-overlap, and separately for
observed-edge versus synthetic-edge negatives. T2 must remain a dedicated
all-observed hard-negative stratum and must never be pooled away by the larger
synthetic strata.

This option gives every generator one schema and one evidence budget while
making the actual generalization claim explicit.

### Option B: retain the two-document path abstraction

If every example must remain a two-document traversable path, merge T1 and T2
because both reduce to the same reversed ordered path under that abstraction.
Remove T3 because a valid directed two-switch has zero support on the two
adjacent edges. A new perturbation must then be designed and shown to have a
non-isomorphic canonical signature before any LOPO-style claim is restored.

This option is simpler, but it discards the all-real T2 collider and weakens the
central claim that individually valid edges can fail to compose.

## Implemented validation support

`src/topocf_rag/topology.py` now provides:

- immutable `EvidenceDocument`, `TypedEdge`, and `EvidenceTopology` objects;
- implicit fixed `q` and consecutive neutral document aliases;
- exact document-text SHA256 identities;
- strict endpoint, self-loop, duplicate-edge, and canonical-order validation;
- exhaustive label-independent canonical signatures for up to six documents;
- relation multisets and per-node, per-relation directed degree profiles;
- matched-pair validation for question identity, alias/text identity, edge and
  relation budgets, and exact-token difference.

The focused and full test commands are:

```bash
.venv/bin/python -m pytest -q tests/test_topology.py
.venv/bin/python -m pytest -q
```

At design-gate handoff, the focused suite passed 11 tests and the current full
suite passed 55 tests.
