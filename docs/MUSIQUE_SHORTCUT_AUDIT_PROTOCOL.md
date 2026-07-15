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

## Planned later stages

After Stage E is reviewed, the next experiment is chosen from the frozen gate
outcome. Gold labels remain restricted to aggregate evaluation and frozen
strata; they never enter scores, graph edges, propagation, or tie breaking.
