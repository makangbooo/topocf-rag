# 2WikiMultiHopQA graph-retrieval adapter protocol

## Purpose

The HotpotQA graph retrieval kill test and its robustness controls passed, so
the next experiment is a cross-dataset reproduction on the official
2WikiMultiHopQA train and dev splits. This remains a controlled ten-document
context-pool experiment, not full-corpus GraphRAG.

## Frozen structural eligibility rule

The official files contain a small number of supporting-fact indices that do
not address an available sentence, plus duplicate context titles that make a
title-identity graph ambiguous. The main experiment therefore includes a
record only when all of the following hold:

1. the context contains exactly ten documents;
2. context titles are unique under the project's deterministic title
   normalization;
3. every supporting fact resolves by **exact title** to exactly one context
   occurrence with an in-range sentence index; and
4. the number of unique gold documents is four for `bridge_comparison` and two
   for `comparison`, `compositional`, and `inference`.

Title normalization is not used to repair or reinterpret gold labels. No
out-of-range index is clamped, shifted, or silently discarded. Exclusion
counts and overlaps are reported before retrieval is scored.

## Frozen sampling rule

- Official train and dev remain separate.
- Seed: `20260714`.
- Train: 1,000 eligible records, 250 from each question type.
- Dev: 500 eligible records, 125 from each question type.
- Within each type, IDs are ordered by a SHA256 rank over the seed, split,
  type, and ID; context order never affects selection.

Balanced sampling makes later per-type comparisons adequately powered and
prevents the large compositional stratum from choosing every configuration.
The retrieval report must show per-type metrics; a sample-wide aggregate is a
type-macro result, not an estimate of the official population distribution.

## Reproduction

```bash
source /home/mkb524/miniconda3/etc/profile.d/conda.sh
conda activate topocf-rag-cert-v1
cd /home/mkb524/topocf-rag

python scripts/prepare_2wiki_splits.py
```

The two ID manifests contain IDs and provenance metadata only. The public
audit contains aggregate counts, paths, and hashes and never embeds dataset
text.

## Frozen bge-m3 cache

After the split audit passes, score each selected question against its ten
official context documents with the local bge-m3 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/score_2wiki_contexts.py
```

The serialization and dense-scoring contract is identical to the Hotpot kill
test: `title + newline + joined sentences`, maximum length 512, FP16 model
execution, L2-normalized dense vectors, and cosine similarity. Gold supporting
facts are not read by the scorer. Cache reuse requires exact source, ID-file,
model-path, serialization-version, and maximum-length metadata matches.

## Frozen graph-retrieval evaluation

The primary cross-dataset result transfers the Hotpot-selected graph method
without using 2Wiki for selection: one-hop maximum propagation, dense/BM25
RRF seed, direction ignored, and graph weight 0.75. A separate secondary
result selects one of the same 36 declared configurations on frozen 2Wiki
train and evaluates it once on dev.

Two-gold and four-gold questions are compared using the same **extra evidence
budget** rather than the same absolute K. Extra budgets are 0, 1, and 3, giving
K=2/3/5 for `comparison`, `compositional`, and `inference`, and K=4/5/7 for
`bridge_comparison`. The primary aggregate is a macro average across the four
balanced question types.

```bash
python scripts/evaluate_2wiki_graph_retrieval.py \
  --output reports/graph_retrieval/2wiki_graph_retrieval.json
```

The zero-shot reproduction gate requires at least +0.02 absolute macro mean
complete-evidence retrieval over the matching RRF baseline, with no more than
0.01 absolute macro regression at any frozen extra budget. This gate is fixed
before observing 2Wiki retrieval outcomes.

## Post-primary robustness validation

Because the primary 2Wiki outcome is known before these controls are declared,
the validation report explicitly marks itself as post-primary rather than
preregistered evidence. It compares the frozen Hotpot transfer against the
strongest non-graph baseline selected on 2Wiki train, a separately
train-selected degree-only control, 200 deterministic node-label
permutations, and paired exact McNemar tests at every extra budget.

```bash
python scripts/validate_2wiki_graph_retrieval.py \
  --base-report reports/graph_retrieval/2wiki_graph_retrieval.json \
  --output reports/graph_retrieval/2wiki_graph_retrieval_validation.json
```

The validation gate requires at least +0.02 over both the strongest non-graph
and degree-only controls, no more than 0.01 regression at any macro budget
against the strongest baseline, permutation `p <= 0.01` with the observed
score above the null maximum, and positive exact-McNemar gains with
`p <= 0.01` at every budget.
