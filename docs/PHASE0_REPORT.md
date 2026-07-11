# Phase 0 Report

Date: 2026-07-11

## Decision

Phase 0 completed without training a verifier or calling the closed-source
API. The preregistered ordered-directed title-mention coverage is 77.6% on the
train diagnostic split and 70.6% on dev-distractor. Both exceed the 60% hard
gate, so the graph-coverage decision is **continue**.

This decision has an important limitation: ordered-directed coverage uses the
first appearance of each unique title in `supporting_facts` as `title0 ->
title1`. HotpotQA does not document this order as semantic reasoning direction.
It is a reproducible operational proxy, not evidence that the graph captures
causal or semantic direction.

## Environment

| Item | Recorded value |
| --- | --- |
| Python | 3.11.14 |
| Python executable | `/file_system/mkb524/topocf-rag/.venv/bin/python` |
| PyTorch | 2.9.1+cu126 |
| CUDA reported by PyTorch | 12.6 |
| GPU | NVIDIA A100-SXM4-80GB, visible device 0 |
| transformers | 4.57.6 |
| FlagEmbedding | 1.4.0 |
| Resolved bge-m3 | `/file_system/models/embedding_models/bge-m3` |

The requested `/file_system/models/bge-m3` directory was absent. The complete
dependency snapshot is in `reports/environment/pip-freeze.txt`; CUDA and Python
metadata are in `reports/environment/runtime.json`. `pip check` reported no
broken requirements.

## HotpotQA inspection and frozen splits

Both sources are top-level JSON arrays. Every inspected record has `_id`,
`question`, `answer`, `context`, `supporting_facts`, `type`, and `level`.
`question` and `answer` are strings; `context` is a list of `[title,
list-of-sentences]`; `supporting_facts` is a list of `[title, sentence-index]`.
No record text was written to reports.

| Official split | Records | Bridge pool | Frozen | Selected levels |
| --- | ---: | ---: | ---: | --- |
| train | 90,447 | 72,991 | 1,000 | easy 198, medium 631, hard 171 |
| dev_distractor | 7,405 | 5,918 | 500 | hard 500 |

Sampling used seed `20260711`, Hamilton proportional allocation by `level`,
and a stable SHA256 rank within each stratum. Dev-distractor bridge records are
all `hard`, so its stratification correctly degenerates to one stratum. The
official train/dev sources were not mixed and the selected ID sets have zero
overlap.

### Required SHA256 values

| Artifact | SHA256 |
| --- | --- |
| `hotpot_train_v1.1.json` | `26650cf50234ef5fb2e664ed70bbecdfd87815e6bffc257e068efea5cf7cd316` |
| `hotpot_dev_distractor_v1.json` | `4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea` |
| `hotpot_train_bridge_1000_ids.json` | `a06aeb4b654bfe615d572788dea4d7032ff82a4b4f3bd0c8c3382e9714026c57` |
| `hotpot_dev_bridge_500_ids.json` | `501affbe3d35fdc51be29f0cf8012718d9b47b4063ddd229da3efebf862b8bbb` |

## bge-m3 smoke test

The smoke test encoded exactly 20 fixed synthetic English texts on CUDA device
0. The output shape was `[20, 1024]` with `float16` output. Elapsed time,
including model load, was 4.823 seconds. Peak PyTorch allocated GPU memory was
1,148,672,512 bytes (about 1.07 GiB). All values were finite.

## Title-mention graph

Each question's provided HotpotQA distractor context is the retrieval pool.
bge-m3 cosine scores rank full title-plus-document text, with `top_k=10`; this
is reranking inside the provided pool, not corpus-wide Wikipedia retrieval.
Mention edges use conservative NFKC, underscore-to-space, whitespace collapse,
case-folding, and Unicode word-boundary matching. Punctuation and parenthetical
qualifiers remain significant, and self-links are excluded.

| Coverage | Train (n=1,000) | Dev (n=500) |
| --- | ---: | ---: |
| Ordered directed | 776 (77.6%) | 353 (70.6%) |
| Any direction / direction ignored | 792 (79.2%) | 358 (71.6%) |
| Reverse-only | 16 (1.6%) | 5 (1.0%) |
| Bidirectional | 45 (4.5%) | 18 (3.6%) |

For two gold document nodes, "a real edge in either direction" and "an edge
after ignoring direction" have the same existence count. They are therefore
not independent evidence.

## Natural path candidates

Natural negatives are real `q -> d_i -> d_j` candidates: `d_i` is in the bge
retrieval pool, the title-mention edge is observed in document text, and the
path does not cover both supporting titles. No edge was edited.

| Statistic | Train | Dev |
| --- | ---: | ---: |
| Total natural negative paths | 2,848 | 1,335 |
| Mean per question | 2.848 | 2.670 |
| Median | 1 | 1 |
| P75 / P90 | 4.25 / 8 | 4 / 8 |
| Maximum | 23 | 23 |
| Questions with zero | 379 (37.9%) | 231 (46.2%) |
| Questions with both gold and natural paths | 504 | 202 |
| Questions with a natural path within 5% token length | 161 | 66 |

The full integer histograms and retrieval-score matching diagnostics are in
`reports/title_graph/analysis.json`. The strict 5% row is diagnostic only: the
5% constraint is mandatory for synthetic topology-only pairs, while natural
errors are required to be matched as closely as possible and reported with
their residual gaps.

## Verification

```text
.venv/bin/python -m pytest -q
20 passed in 0.77s

.venv/bin/python -m pip check
No broken requirements found.
```

The exercised tests cover streaming schema validation, deterministic
stratified sampling, official split separation, title normalization and
boundaries, edge direction, real-edge candidate paths, retrieval top-k, and
invalid data rejection.

## API status

No closed-source request was sent. Key rotation has not been confirmed and the
required endpoint/model configuration is incomplete. The one-call smoke test
remains blocked until rotation is explicitly confirmed.

## Recommendation

Proceed to Phase 1 graph/pair materialization, but keep natural candidates as a
separate evaluation set. Before verifier training, quantify the nearest
retrieval-score and token-length gaps and freeze the matching policy. The
current provided-context pool yields only 202 dev questions with both a gold
path and a natural negative, and 66 with a natural negative inside a strict 5%
token window. If this is too small for stable confidence intervals, expand to
corpus-level retrieval or repeated seeded diagnostic subsets before changing
the graph semantics. Do not treat the ordered-versus-undirected difference as
a direction-reasoning result.
