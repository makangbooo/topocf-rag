# MuSiQue receiving-degree shortcut audit protocol

## Purpose and source

The Hotpot graph method beat a train-selected degree-only control, whereas the
2Wiki tuned graph method and degree-only control were effectively identical.
MuSiQue is therefore used as a third official dataset to test whether GraphRAG
retrieval gains reflect query-conditioned propagation or benchmark structural
centrality.

Only the official MuSiQue v1.0 answerable train and dev JSONL files are in
scope. Their expected hashes are frozen in code. MuSiQue-Full unanswerable
instances and the server's external Wikipedia corpus are excluded. The data
format follows the official repository:
<https://github.com/StonyBrookNLP/musique>.

## Stage A: schema and eligibility audit

The first run performs no sampling, retrieval scoring, model inference, or
hyperparameter selection. It streams both files and reports aggregate schema
counts only.

A record is structurally eligible when:

1. it is answerable and has a non-empty ID and question;
2. paragraph indices are unique, contiguous, and equal to list order;
3. every paragraph has a non-empty title and text plus a Boolean
   `is_supporting` label;
4. normalized titles are unique within the candidate pool;
5. every decomposition step has a valid ID, question, answer, and resolvable
   `paragraph_support_idx`; and
6. the set of decomposition support indices exactly equals the set of
   paragraphs marked supporting.

```bash
python scripts/audit_musique_schema.py \
  --output reports/musique/schema_audit.json
```

Passing Stage A verifies source identity and structural auditability only.
Hop-count, paragraph-count, and exclusion histograms must be inspected before
sample sizes or retrieval budgets are frozen. No dev result may influence a
retrieval method or hyperparameter.

## Stage B: duplicate-title and label-ambiguity audit

Stage A showed that strict unique-title identity would remove roughly half of
both official splits. Those records must not be silently discarded. Stage B
separates duplicate titles with distinct paragraph text from exact-text
duplicates that mix supporting and non-supporting labels.

```bash
python scripts/audit_musique_duplicates.py \
  --schema-report reports/musique/schema_audit.json \
  --output reports/musique/duplicate_title_audit.json
```

Before observing Stage B counts, the planned primary pool is frozen as records
with exactly 20 paragraphs that satisfy every structural contract except title
uniqueness and contain no exact-text duplicate-title group with mixed support
labels. Paragraph `idx` is the node identity. A normalized title mention fans
out deterministically to every matching paragraph occurrence. The exact-20,
strict-unique-title pool is retained as a sensitivity analysis rather than the
main experiment. No retrieval scoring occurs in Stage B.

## Stage C: frozen ID manifests

Stage B found no exact-text duplicate-title group, so every structurally valid
record is label-unambiguous under paragraph-occurrence identity. The primary
pool is restricted to the official 20-paragraph instances.

Train is selected with seed `20260715` using nine cells: 2/3/4 hops crossed
with unique titles, distractor-only title collisions, and supporting-title
collisions. Exactly 200 records are selected per cell, for 1,800 total. Dev is
not sampled: every eligible 20-paragraph record is retained in official JSONL
order. The primary diagnostic aggregate is a macro average over the nine cells;
official-distribution micro and the unique-title cells are secondary analyses.

```bash
python scripts/prepare_musique_splits.py
```

Only ID manifests and content-free aggregate audit reports are written. Passing
Stage C authorizes local bge-m3 context scoring but not graph method selection
on dev.

## Stage D: frozen dense retrieval caches

Local bge-m3 scores every frozen question against its 20 official paragraph
occurrences. The serialization is `title + newline + paragraph_text` and the
official paragraph order is preserved. Cache rows contain only question IDs,
paragraph indices, cosine scores, and truncated token lengths. Duplicate titles
remain separate paragraph-occurrence nodes.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/score_musique_contexts.py
```

The scoring script verifies the Stage C gate, both manifest hashes, both source
hashes, the frozen Stage C report hash, the local model and tokenizer
fingerprints, all nine hop-collision allocations, and every cache alignment
before it writes the aggregate report. Gold support flags, answers, and
decompositions do not enter the score computation. No retrieval metric or graph
configuration is selected in Stage D.

## Stage E: occurrence-aware graph and shortcut controls

Paragraph `idx` remains the graph node identity. A boundary-safe literal title
mention in one paragraph body creates a directed edge to every other paragraph
occurrence with that normalized title; same-title occurrences are never merged
and self-links are excluded.

The same predeclared 36-configuration grid is used for the real graph and the
degree-only control. The strongest non-graph baseline, real graph configuration,
and degree-only configuration are selected independently on the balanced train
split using the nine-cell macro metric. Dev is evaluated once. The frozen
Hotpot configuration is also reported as a zero-shot transfer result.

Top-K is the question's gold paragraph count plus an extra budget of 0, 1, or
3. A positive mechanism gate requires the selected real graph to beat both the
train-selected non-graph baseline and independently train-selected degree-only
control, outperform 200 question-local node-identity permutations, and produce
significant paired gains at every budget. Failure is a scientific result and
forbids a query-conditioned graph-gain claim.

```bash
python scripts/evaluate_musique_graph_retrieval.py \
  --no-fail-on-gate
```

Stage E selected `one_hop_max__dense__undirected__w0.75` on train and passed
the frozen dev gate. The result supports a narrow query-conditioned graph
alignment claim inside the official 20-paragraph pools. It does not establish
directed topology reasoning, uniform gains across hop/collision cells, QA
generation gains, or full-corpus GraphRAG performance.

## Stage F: post-primary robustness and claim-boundary audit

Stage F was declared after observing Stage E and is explicitly not
preregistered evidence. It binds the exact Stage E report SHA256 and first
reproduces all selected train/dev metrics. It then runs three deterministic,
nine-cell-stratified paired bootstraps comparing the frozen graph against:

1. the train-selected dense baseline;
2. the independently train-selected degree-only control; and
3. a degree-only control using exactly the graph method's family, seed,
   direction, and weight.

The bootstrap resamples questions with replacement within every frozen cell,
macros equally over cells, and reports percentile 95% intervals for each
budget and their mean. The primary post-primary robustness gate requires the
mean-delta lower bound to exceed zero for all three comparisons.

Two train-only diagnostics do not enter that gate. A 500-replicate stratified
bootstrap measures graph-configuration selection stability. A nine-fold
leave-one-cell-out analysis selects the graph, baseline, and degree control on
eight balanced train cells and evaluates the ninth. Direction-specific train
selection and fixed-config direction variants are reported as sensitivity
analyses; dev never selects a configuration.

```bash
python scripts/validate_musique_graph_retrieval.py \
  --no-fail-on-gate
```

Even if Stage F passes, the authorized claim remains limited to a small
query-conditioned retrieval-alignment gain in the frozen MuSiQue candidate
pools. Because the primary method is undirected and subgroup effects are
heterogeneous, directed-reasoning and uniform-improvement claims remain
forbidden.

## Stage G0: untouched answerable-test readiness audit

Stage F passed all three paired-bootstrap controls. Before any test retrieval
score is computed, Stage G0 binds the exact Stage F report and the official
answerable-test source hash. It streams the test file to audit schema,
duplicate-title policies, nine-cell coverage, and overlap with the frozen
train/dev manifests. It performs no sampling, embedding, retrieval evaluation,
or configuration selection.

The following test hypotheses are frozen before the audit output is inspected:

1. the train-selected undirected graph has a positive nine-cell paired-
   bootstrap 95% lower bound against dense;
2. it has a positive lower bound against the same-config degree prior; and
3. outgoing propagation exceeds incoming propagation by at least `0.02` in
   macro mean complete-evidence rate and has a positive paired-bootstrap lower
   bound.

Outgoing versus undirected is descriptive only and can never select the test
method. Subgroup estimates remain non-confirmatory.

```bash
python scripts/audit_musique_test_readiness.py
```

Passing Stage G0 authorizes only review of aggregate counts and deterministic
materialization of an all-eligible test ID manifest. Test scoring remains
paused until those counts are reviewed and the manifest hash is frozen.

Stage G0 did not pass. The official answerable-test file contains only `id`,
`question`, and `paragraphs`; it has no answerability flag, decomposition, or
paragraph support labels. Consequently, complete-evidence retrieval cannot be
evaluated and the official test file must never be scored under this protocol.
The failed audit is retained as a frozen scientific result rather than
reinterpreted or rerun.

## Stage G1: unused official-train holdout readiness

The Stage C train manifest used only 1,800 deterministically selected records
from a larger structurally eligible official-train pool. Records outside that
manifest have never been embedded, scored, evaluated, or used for graph
selection. Stage G1 audits this labeled remainder as a transparent
within-dataset replication holdout.

Before any remainder scoring, the audit binds the exact Stage C report and the
failed Stage G0 report, reconstructs the Stage C candidate pool, removes every
selected train ID, verifies no overlap with selected dev, and requires at least
100 untouched records in each of the nine cells. The pool rule is all remaining
eligible records in official source order; no sampling or balancing is allowed.
The Stage G0 hypotheses are carried forward unchanged.

```bash
python scripts/audit_musique_unused_holdout.py
```

Passing Stage G1 authorizes manifest materialization only. This is explicitly
a post-primary untouched within-dataset replication, not an official test-set
result, independent dataset result, or preregistered external validation.

Stage G1 did not pass. After removing the 1,800 selected train records, the
untouched eligible remainder contained 18,117 records, but the frozen
`4-hop + distractor` cell contained only 24. The preregistered minimum was 100
per cell. That minimum must not be reduced after inspecting the audit, and no
remainder embedding or retrieval score may be computed. The failed report is
retained as a hash-bound stopping result.

The next admissible check is the untouched official 2Wiki test readiness
audit described in `docs/2WIKI_ADAPTER_PROTOCOL.md`. It carries the MuSiQue
hypotheses forward without using 2Wiki test labels for method selection.

## Planned later stages

No MuSiQue test or unused-remainder scoring stage is currently authorized.
Gold labels remain restricted to aggregate evaluation and frozen strata; they
never enter scores, graph edges, propagation, or tie breaking.
