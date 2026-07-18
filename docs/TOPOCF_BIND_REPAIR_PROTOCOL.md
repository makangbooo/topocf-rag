# TopoCF-Bind-Repair v1 protocol

Frozen: 2026-07-16

Stopped: 2026-07-18. The frozen independent-edge baseline reached three-seed
official-dev mean pairwise accuracy `0.999145`, triggering the preregistered
`STOP_CURRENT_METHOD` decision. The global binding/repair model described below
must not be trained for `synthetic_common/t3`. The successor feasibility audit
is `TOPOCF_ALL_OBSERVED_V2_PROTOCOL.md`.

## Decision

The original mixed T1/T2/T3 task is not a valid primary learning problem. A
content-free rule that sees only manifest graph metadata reaches 0.9912
question-macro pairwise accuracy on HotpotQA dev. T1, T2, fork, and
double-collider negatives expose their label through reachability or canonical
topology. They remain useful diagnostic controls, but they are excluded from
primary training and headline results.

The primary method-development task is the degree-preserving T3 target swap:

```text
positive: q->A, q->C, A->B, C->D
negative: q->A, q->C, A->D, C->B
```

Both members use exactly the same question, four immutable documents, selected
sentences, two retrieval edges, two mention edges, canonical typed topology,
and per-node/per-relation directed degrees. Only the document-to-node binding
changes. Observation and provenance flags are never model inputs.

## Frozen data roles

| Role | Manifest selector | Train | Dev | Use |
|---|---|---:|---:|---|
| Development | `synthetic_common/t3` | 161 questions / 368 pairs | 65 / 150 | train and bounded kill-test dev |
| Confirmatory | `all_observed_rewire/t3_all_observed` | 27 / 65 | 12 / 28 | descriptive only until expanded |
| Diagnostic | T1, T2, fork, double-collider | not applicable | not applicable | report rule and model behavior; never pool into primary |

The all-observed HotpotQA stratum is below the frozen minimum of 50 dev
questions. It cannot support the paper's confirmatory claim by itself.

### Train-only model-selection split

The 161 synthetic-T3 train questions are divided by question ID, never by
pair. Pair-count buckets `1`, `2`, `3-4`, and `5+` are allocated with the
Hamilton method, followed by SHA256 ranking with seed `20260718`.

| Inner role | Questions | Pairs | Permutation use |
|---|---:|---:|---|
| Fit | 129 | 294 | one shared deterministic S4 permutation per pair and epoch |
| Inner validation | 32 | 74 | exact 24-permutation score average |

Only fit examples may contribute gradients. Inner validation selects the loss
weight, learning rate, epoch, and checkpoint. The 65-question synthetic T3
official dev split is evaluated once after the configuration is frozen. The
all-observed official-dev stratum is neither a training nor a selection set.

## Method: permutation-equivariant binding energy

Let the four document nodes be `d0...d3`. A text encoder produces
question-conditioned document representations. A relation-conditioned binding
module then produces a compatibility matrix for candidate typed edges. For the
two mention sources and two mention targets, a Sinkhorn-normalized matrix gives
a soft one-to-one binding. The candidate topology energy combines:

1. question-to-retrieval-root compatibility;
2. source-text-to-target-document edge grounding;
3. global question/evidence sufficiency; and
4. the constrained binding score.

Training has two losses:

```text
L = L_pairwise_margin + lambda_repair * L_binding_repair
```

`L_pairwise_margin` ranks the positive binding above its target-swapped
counterfactual. `L_binding_repair` recovers the positive endpoint assignment
from the negative candidate. `lambda_repair` is selected using a train-only
inner split. Dev is not used for architecture, threshold, loss-weight, epoch,
or checkpoint selection.

This is not presented as a generic binary verifier. The method's testable claim
is narrower: counterfactual binding supervision and a constrained repair head
improve question-conditioned evidence binding beyond flat semantic or
independent-edge scoring.

## Mandatory permutation control

Neutral aliases and serialization positions are not semantic. During training,
sample one deterministic epoch-dependent permutation from the 24 elements of
`S4` for each matched pair and apply it identically to both members. At dev and
test, score all 24 simultaneous
document/edge relabelings and average the 24 logits before ranking.

The repository implements the relabeling orbit in
`src/topocf_rag/serialization.py`. Document text and typed-edge endpoints move
together. The rule audit motivates this control: the train-fitted
alias-sensitive lookup reaches 0.7139 on the small all-observed dev stratum,
whereas canonical topology, degree, all-edge reachability, and all-edge
composability rules are exactly 0.5.

## Frozen baselines and ablations

Every primary result must report question-macro matched-pair accuracy, AUROC,
mean/median margin, and question-level bootstrap 95% intervals. Required
comparisons are:

1. BM25 and bge-m3 cosine from the frozen Phase 1 report;
2. content-free canonical topology, degree, reachability, and composability;
3. a flat question-plus-serialized-topology cross-encoder;
4. independent local edge scoring without the constrained binding matrix;
5. the full counterfactual ranking plus repair model;
6. full model without repair loss;
7. full model without the 24-permutation evaluation ensemble; and
8. training with the diagnostic T1/T2 examples added, reported only as a
   shortcut-sensitivity ablation.

The observed/provenance rules are reported as forbidden-metadata audits, not as
legitimate model-visible baselines.

## Gates

Development training is authorized only if `audit_method_readiness.py` passes.
It currently passes for the synthetic T3 development task. This authorizes one
bounded implementation and hyperparameter study, not a paper claim.

The confirmatory and paper gates require all of the following:

- exact score invariance under the 24 alias permutations;
- at least 50 untouched all-observed dev questions under the common schema;
- a schema-matched natural retrieval-error evaluation;
- improvement over both the flat cross-encoder and independent-edge ablation;
- a positive structured repair result, not only binary ranking accuracy; and
- replication under a frozen adapter on a second official dataset.

If the full model wins only on synthetic T3, if the gain disappears on
all-observed pairs, or if independent title-mention checking matches the full
model, stop the method claim rather than adding more synthetic generators.

## Reproduction

```bash
python scripts/audit_method_readiness.py
python scripts/prepare_method_inner_split.py
python -m pytest -q tests/test_method_readiness.py tests/test_serialization.py
python -m pytest -q tests/test_method_data.py
```

The frozen configuration is
`configs/certificate_v1/topocf_bind_repair_v1.json`; the aggregate report is
`reports/phase1/method_readiness.json`. Both are bound to the exact Phase 1
manifests, semantic baseline report, and structural leakage report by SHA256.
The train-only split is `data/splits/topocf_t3_train_inner_v1.json`, and its
aggregate-only materialization report is
`reports/phase1/method_inner_split.json`. The model-facing data contract and
all relevant artifact hashes are frozen in
`configs/certificate_v1/topocf_data_v1.json`.
