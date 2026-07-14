# Graph-aware retrieval kill test

## Question

The existing Phase 0 graph is constructed after dense retrieval and therefore
does not establish that graph structure improves retrieval.  This kill test
asks a narrower question before any verifier training or cross-dataset work:

> Can the literal HotpotQA title-mention graph improve complete two-document
> evidence retrieval over strong label-free baselines inside the official
> context pool?

This remains a controlled ten-document-pool diagnostic.  Passing it does not
establish full-corpus GraphRAG performance.

## Leakage controls

- Graph nodes and edges use only official context documents and literal
  cross-document title mentions.
- Dense and BM25 scores use the question and document text only.
- Gold supporting facts are consumed only by aggregate evaluation metrics.
- The 36 graph configurations are declared in code before observing results.
- Hyperparameters and the strongest non-graph comparator are selected using
  the frozen train 1,000 questions only.
- Frozen dev 500 questions are evaluated only after selection.  Dev metrics do
  not select a direction, propagation family, seed, or weight.
- Exact score ties are resolved by ascending official context index.

The non-graph baselines are dense bge-m3, question-local BM25, and dense/BM25
reciprocal-rank fusion.  Graph candidates combine dense or fused seeds with
one-hop maximum propagation or personalized PageRank over outgoing, incoming,
or direction-ignored title-mention edges.

## Metrics and gate

For `K in {2, 3, 5}`, the primary retrieval outcome is whether both gold
supporting documents occur in the top K.  The report also includes supporting
document recall, at-least-one-gold rate, no-gold rate, full-evidence MRR, graph
coverage, and the share of dense partial failures whose missing gold document
is connected to the retrieved gold document.

The graph feasibility gate passes only when, against the train-selected
non-graph baseline on dev:

1. the mean complete-evidence rate over K=2,3,5 improves by at least 0.02
   absolute; and
2. the graph method does not regress by more than 0.01 absolute at any K.

Failure means redesign the graph retrieval representation before integrating
2WikiMultiHopQA or MuSiQue.  Passing authorizes cross-dataset reproduction; it
does not authorize a final paper claim.

## Reproduction

The command reuses the immutable bge-m3 caches created by
`scripts/analyze_title_graph.py`; it makes no model or API calls.

```bash
source /home/mkb524/miniconda3/etc/profile.d/conda.sh
conda activate topocf-rag-cert-v1
cd /home/mkb524/topocf-rag

python scripts/evaluate_graph_retrieval.py \
  --output reports/graph_retrieval/hotpot_kill_test.json
```

The public report contains aggregate metrics, configurations, paths, and
cryptographic hashes only.  It contains no question IDs or dataset text.

## Robustness validation

After the primary gate is frozen, run a separate validation rather than
modifying or overwriting its report.  The validation reuses the train-selected
graph configuration and adds four controls:

1. comparison against the exact non-graph seed used by the selected method;
2. a degree-only control whose seed, direction, and weight are independently
   selected on frozen train, with no query-conditioned neighbor message;
3. 200 deterministic question-local node-label permutations that preserve the
   full graph isomorphism and degree sequence but break graph/document
   alignment; and
4. per-question win/loss transitions with two-sided exact McNemar tests.

The robustness gate requires at least 0.02 absolute mean improvement over both
the matched non-graph seed and degree-only control, empirical permutation
`p <= 0.01` with the observed score above every null replicate, and positive
paired gains with exact McNemar `p <= 0.01` at every K.  Direction ablations
choose their configuration independently on frozen train; dev never selects a
direction.

```bash
python scripts/validate_graph_retrieval.py \
  --base-report reports/graph_retrieval/hotpot_kill_test.json \
  --output reports/graph_retrieval/hotpot_kill_test_validation.json
```

Passing shows that the observed graph/document alignment, rather than merely
BM25 fusion, graph density, or an arbitrary graph topology, drives the
controlled-pool gain.  It still does not establish semantic direction
reasoning, full-corpus performance, or the final topology-certificate claim.
