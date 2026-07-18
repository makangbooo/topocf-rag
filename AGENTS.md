# Repository Instructions

These instructions apply to the entire repository.

## Research scope

TopoCF-RAG studies topology-counterfactual evidence path verification for
English multi-hop GraphRAG. The central test is whether a verifier can reject a
path whose documents are individually relevant and factually valid but whose
direction, bridge connection, or graph topology does not support the question.

Do not present random edge deletion, arbitrary relation corruption, or random
negative sampling as the main contribution. Generator-aware leave-one-out
utility may only be used as an auxiliary label, filter, or ablation.

Phase 0 is a diagnostic stage. Do not train a verifier during this phase.

## Phase 1 design gate

Read `docs/PHASE1_DESIGN_GATE.md` before generating synthetic pairs. Formal
pair materialization, bge-m3 baseline scoring, and verifier training are
paused until one common evidence representation is selected. The current
two-document path cannot represent a legal T3 two-switch, and T1/T2 share the
same canonical typed collider signature. Do not describe the current protocol
as generalization to three unseen topology classes.

The fixed four-document, two-path `EvidenceTopology` option was selected and
implemented. A subsequent content-free audit showed that mixed T1/T2/T3
training is structurally shortcut-solvable. Follow
`docs/TOPOCF_BIND_REPAIR_PROTOCOL.md`: only degree- and topology-matched T3 is
the primary method-development task; T1/T2/fork/double-collider are diagnostic
only. All-observed T3 remains a separate, currently underpowered confirmation
stratum.

The readiness gate authorizes bounded method development only. Fitting must
use the frozen question-disjoint train-only inner split in
`data/splits/topocf_t3_train_inner_v1.json`. Official dev and all-observed pairs
must not be used for loss-weight, hyperparameter, epoch, or checkpoint
selection. Synchronized training-time S4 relabeling and exact 24-permutation
score averaging for validation are mandatory.

## Environment and paths

- Repository: `~/topocf-rag`
- Isolated Conda environment:
  `/home/mkb524/miniconda3/envs/topocf-rag-cert-v1`
- HotpotQA root: `/file_system/datasets/hotpotqa`
- Requested model path: `/file_system/models/bge-m3` (not present at bootstrap)
- Resolved bge-m3 path:
  `/file_system/models/embedding_models/bge-m3`
- Primary GPU: the available NVIDIA A100-SXM4-80GB; do not search for a 4090.

Use Python 3.11 in the existing `topocf-rag-cert-v1` Conda environment. Never
install project dependencies into the base environment or recreate a project
`.venv` unless the user explicitly changes the environment decision.

## Data rules

- Read large JSON files with a streaming parser. Never print full examples or
  recursively dump dataset contents.
- Use official splits only; never mix or reshuffle train and dev.
- Phase 0 train diagnostic split: 1,000 bridge questions from official train,
  seed `20260711`, stratified by `level`.
- Phase 0 evaluation diagnostic split: 500 bridge questions from official
  `dev_distractor`, seed `20260711`, stratified by `level`.
- Store only selected IDs in:
  `data/splits/hotpot_train_bridge_1000_ids.json` and
  `data/splits/hotpot_dev_bridge_500_ids.json`.
- Record SHA256 digests for both source files and both selected-ID files.
- Public `answer` and `supporting_facts` fields are allowed. Do not add human
  annotations.
- 2WikiMultiHopQA is the second dataset, but must not be integrated until the
  HotpotQA kill test passes.

Sampling must be deterministic. Tests must verify sample size, uniqueness,
bridge-only membership, official-split membership, stratification behavior, and
stable output for the fixed seed.

## Graph contract

The first graph is a minimal query-document graph:

- Nodes: one question node `q` and Hotpot context document nodes `d_i`.
- Retrieval edges: `q -> d_i`, scored by local bge-m3.
- Mention edges: `d_i -> d_j` only when the body of `d_i` mentions the
  deterministically normalized title of `d_j`.
- Candidate paths: `q -> d_i -> d_j`.

Title normalization must be deterministic and covered by tests. Normalize
Unicode consistently, map underscores to spaces, collapse whitespace, and
compare case-insensitively. Do not remove punctuation or parenthetical text in
a way that merges distinct titles. Reject self-links unless an experiment
explicitly defines and reports them.

A positive path covers both supporting titles and contains a real
title-mention connection. A natural negative must come from actual retrieval,
contain a real title-mention edge, fail to cover both supporting titles, have
the same path length, and be matched as closely as possible on bge-m3 retrieval
score and token length. Never manually edit an edge for a natural negative.

Report directed and direction-ignored gold-title mention coverage, plus the
per-question natural-negative count distribution. State the exact denominator
and document-order rule used for directed coverage.

**Hard gate:** if directed gold title-mention coverage is below 60%, stop before
path-verifier training, report the result, and redesign the graph. Do not bypass
or reinterpret this threshold after seeing results.

## Synthetic topology pairs

Every positive/negative pair must keep the same question, exactly the same
document-text multiset, path length, edge count, and relation-type multiset.
Token-length difference may not exceed 5%.

- T1 reverses an explicitly asymmetric title-mention edge.
- T2 swaps bridge-document roles while retaining the same question and nodes.
- T3 performs degree-preserving 2-switch rewiring.

Do not use the old leave-one-perturbation-out plan as a primary claim: the
generators are not disjoint topology classes and several are solved by explicit
reachability. Natural negatives are always evaluated separately and never
enter synthetic training data.

## Metrics and serialization

Path serialization must preserve document order, directed arrows, relation
markers, document titles, and only the necessary sentences. Baselines are
BM25 and bge-m3 cosine similarity between the question and serialized path.

For each positive/negative pair, pairwise accuracy is `1` when
`score_pos > score_neg`, `0.5` for a tie, and `0` otherwise. Average within each
question first, then macro-average across questions. Also report mean and median
score margin, AUROC, MRR, Recall@K, official HotpotQA Answer EM/F1, Supporting
Fact EM/F1, and mean input-token count.

## API safety

Do not call the closed-source generator until the user confirms that the API
key has been rotated. Read `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and
`OPENAI_MODEL` from the environment only. Never hardcode, print, log, persist,
or commit credentials or request headers.

After rotation, make exactly one deterministic, single-concurrency smoke call
to `{OPENAI_BASE_URL-without-trailing-slash}/chat/completions` with a 60-second
timeout, `temperature=0`, `max_tokens=8`, and the prompt `Return exactly OK.`.
Validate `choices[0].message.content`. Record only status code, latency, usage,
response shape, and an exact-match boolean.

## Verification and reporting

Keep generated manifests and reports free of dataset content, secrets, and request
headers. (Dataset content must not be embedded in repository artifacts.) Run
focused tests after each change and the complete test suite before handoff.
Record the Python path, package freeze, PyTorch/CUDA versions, GPU name, bge-m3
smoke output shape, elapsed time, and peak allocated GPU memory.
