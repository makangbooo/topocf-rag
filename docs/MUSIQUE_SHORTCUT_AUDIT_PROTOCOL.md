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

## Planned later stages

After the schema output is reviewed, a separate commit will freeze a
hop-stratified train/dev ID sample. Retrieval scoring will reuse local bge-m3
and compare dense, query-independent receiving degree, and query-conditioned
graph propagation under equal extra-evidence budgets. Gold labels will be used
only for aggregate evaluation and never for scores, graphs, or tie breaking.
