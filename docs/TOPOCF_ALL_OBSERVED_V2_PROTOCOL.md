# TopoCF all-observed v2 feasibility protocol

Frozen: 2026-07-18

## Why the predecessor stopped

The `synthetic_common/t3` task is closed. A question-conditioned local-edge
reranker reached pairwise accuracy `1.0000`, `1.0000`, and `0.9974` on the
65-question official-dev split with three hash-frozen adapters. The mean
`0.999145` exceeded the preregistered stop gate. A global one-to-one binding
module cannot establish necessity on a task already solved by checking each
swapped edge independently.

The aggregate evidence is frozen in
`reports/phase1/text_baseline_dev_gate.json`. No Sinkhorn/binding model may be
trained for the stopped task.

## Replacement candidate

The repository already defines a stricter `all_observed_rewire` construction:

```text
positive: q->A, q->C, A->B, C->D
negative: q->A, q->C, A->D, C->B
```

All four mention relations `A->B`, `C->D`, `A->D`, and `C->B` must occur in
the source documents. Positive and negative candidates share the identical
four-document payload and the union of evidence sentences for all four
relations. Thus every local candidate edge is observed and grounded; edge
existence alone cannot reveal the label.

This is only a necessary condition for a global-composition task, not proof
that independent edge scoring will fail after learning. Population size is
audited before any new split, training, or model score is produced.

## Full-population census

`scripts/audit_all_observed_population.py` streams all official HotpotQA train
and dev-distractor records, filters to `type == bridge`, and treats every
document supplied in the record as a controlled context pool. It does not call
a retriever and must not be described as full-corpus retrieval.

Source-level schema validation is applied to every record before the bridge
filter. Graph-level validation is applied only to bridge questions. A bridge
question whose context cannot satisfy the frozen unique-normalized-title node
identity is excluded rather than silently merging documents; the exclusion and
its content-free reason are counted in the public aggregate report. The
all-observed prevalence keeps all bridge questions as its denominator. This
handling does not change the frozen eligible-question thresholds.

Only aggregate counts and histograms are written. No IDs, questions, answers,
titles, contexts, or sentences are persisted. The frozen feasibility gate is:

- at least 500 eligible official-train questions;
- at least 100 eligible official-dev questions; and
- zero non-observed edges in either member of every candidate pair.

Passing authorizes split design only. It does not authorize verifier training.
Failure requires a different official dataset or graph construction; the
all-observed constraint must not be weakened after seeing the count.

Run:

```bash
python scripts/audit_all_observed_population.py --no-fail-on-gate
```

The frozen inputs, source hashes, thresholds, and post-audit rules are in
`configs/certificate_v1/topocf_all_observed_v2.json`.
