# Project Handoff

Updated: 2026-07-16

## Current status

Phase 0 passed its graph-coverage gate. Phase 1 Option A was implemented with a
fixed four-document `EvidenceTopology`, frozen pair manifests, and BM25/bge-m3
baselines. A content-free structural audit then showed that the mixed
T1/T2/T3 task is shortcut-solvable: the best metadata rule reaches 0.9912 on
dev. Do not train on the mixed generator pool.

The current continuation is `TopoCF-Bind-Repair-v1`. Train only on
`synthetic_common/t3`; reserve `all_observed_rewire/t3_all_observed` as an
untouched confirmatory stratum; use T1/T2/fork/double-collider only as
diagnostics. Exact S4 alias-permutation control, counterfactual ranking, and
structured binding repair are mandatory. See
`TOPOCF_BIND_REPAIR_PROTOCOL.md` and run `scripts/audit_method_readiness.py`
before training.

Current primary feasibility:

- synthetic T3 development: 368 pairs/161 train questions and 150/65 dev;
- all-observed T3 confirmation: 65/27 train and 28/12 dev;
- model-visible permutation-invariant structural rules: 0.5 on both T3 strata;
- all-observed HotpotQA dev remains underpowered relative to the minimum 50.

The synthetic-T3 official-train population is now frozen into 129 fit
questions/294 pairs and 32 train-only inner-validation questions/74 pairs.
Fit and validation are question-disjoint and stratified by per-question T3 pair
count. Official dev remains untouched until model and checkpoint selection are
complete. Use `topocf_data_v1.json` as the model-facing artifact contract.

Run the design audit with:

```bash
source /home/mkb524/miniconda3/etc/profile.d/conda.sh
conda activate topocf-rag-cert-v1
python scripts/audit_phase1_design.py
python scripts/audit_method_readiness.py
python scripts/prepare_method_inner_split.py
```

## Mission

TopoCF-RAG is an English-only multi-hop knowledge-base QA experiment focused on
topology-counterfactual evidence paths. Its claim must be tested on minimal
pairs whose document texts and graph budget are controlled, and on naturally
retrieved wrong paths. Ordinary random corruption is not sufficient evidence.

The immediate task is Phase 0 diagnostics. Do not train a verifier yet.

## Confirmed execution decisions

- The current NVIDIA A100-SXM4-80GB host is the primary experiment machine.
  Do not wait for or search for an RTX 4090.
- Reuse the working CUDA-enabled PyTorch installation through a dedicated
  Python 3.11 virtual environment with `--system-site-packages` if needed.
- Required packages are `transformers>=4.51,<5`,
  `FlagEmbedding>=1.3.5,<2`, `accelerate`, `sentencepiece`, `ijson`, `httpx`,
  and `pytest`.
- The project is new; there is no predecessor repository or experiment state
  to inherit.
- HotpotQA is rooted at `/file_system/datasets/hotpotqa`; concrete source files
  must be discovered without dumping their contents.
- `/file_system/models/bge-m3` was not present at bootstrap. The discovered
  model directory is `/file_system/models/embedding_models/bge-m3`.
- 2WikiMultiHopQA is the second dataset, deferred until HotpotQA passes the kill
  test.

## Phase 0 deliverables

1. Record the virtual environment's Python path, `pip freeze`, PyTorch and CUDA
   versions, and GPU name.
2. Stream-count each HotpotQA source file and report counts and field shapes
   without logging example bodies.
3. Select only `type == bridge` questions from official train and official
   dev-distractor. Sample 1,000 train and 500 dev items with seed `20260711`,
   stratified by `level`.
4. Save selected IDs to the two required JSON files and report SHA256 for each
   source file and selected-ID file.
5. Add deterministic title normalization, minimal graph construction, and
   invariant tests.
6. Encode exactly 20 non-sensitive texts with bge-m3 on the A100. Report vector
   shape, elapsed time, and peak allocated GPU memory.
7. Report gold title-mention coverage in directed and direction-ignored form,
   and natural-negative path counts per question.

Recommended natural-candidate summaries include total candidates, questions
with zero candidates, mean, median, minimum, maximum, and useful quantiles. Keep
the underlying question text and document text out of reports.

## Data invariants

Official train and dev must never be mixed. The frozen files are:

```text
data/splits/hotpot_train_bridge_1000_ids.json
data/splits/hotpot_dev_bridge_500_ids.json
```

Each file must have the requested number of unique IDs, contain bridge examples
only, come exclusively from its named official split, and be reproducible for
the fixed seed. Stratification must operate on the observed `level` values and
use a deterministic allocation rule whose rounded totals equal the requested
sample size.

Source and output SHA256 values belong in the Phase 0 report. Source records,
questions, answers, contexts, supporting sentences, and credentials do not.

## Graph and coverage contract

For each question, create `q -> d_i` retrieval edges using bge-m3 and
`d_i -> d_j` edges when document `d_i` mentions the normalized title of
document `d_j`. Candidate paths are `q -> d_i -> d_j`.

Title normalization must be deterministic. The baseline contract is consistent
Unicode normalization, underscore-to-space conversion, whitespace collapse,
and case-insensitive matching while preserving meaningful punctuation and
parenthetical qualifiers. The implementation and its boundary behavior must be
unit-tested.

A gold path uses the two supporting titles and a real title-mention edge. A
natural error path is retrieved naturally, contains a real mention edge, does
not cover both supporting titles, and has the same number of edges as a gold
path. Match natural errors as closely as possible by retrieval score and token
length; do not modify edges manually.

The coverage report must state:

- eligible bridge-question denominator;
- how supporting-title order is determined for the directed metric;
- directed covered count and rate;
- direction-ignored covered count and rate;
- exclusions and their reasons;
- natural-negative count distribution.

The directed coverage rate is a hard decision gate. If it is below `0.60`, stop
before verifier training, publish the diagnostics, and redesign the graph. The
undirected rate does not override this condition.

## Later topology controls

Synthetic pairs must hold constant the question, document-node/text multiset,
path length, edge count, relation-type multiset, and token length within 5%.
The planned perturbations are asymmetric direction reversal (T1), bridge-role
swap within the same question and nodes (T2), and degree-preserving 2-switch
rewiring (T3).

Use leave-one-perturbation-out evaluation. In each run, two perturbation types
may be used for training and the third is held out completely. Natural wrong
paths stay out of synthetic training data and receive a separate evaluation.

## Evaluation contract

Serialize a path with document order, directed arrows, relation markers,
titles, and necessary sentences intact. Compare against BM25 and bge-m3 cosine
semantic baselines.

For each positive/negative pair:

```text
1.0  if score_positive > score_negative
0.5  if score_positive == score_negative
0.0  otherwise
```

Average pairs within a question and then macro-average over questions. Also
report mean and median score margin, AUROC, MRR, Recall@K, official HotpotQA
Answer EM/F1, Supporting Fact EM/F1, and mean input-token count.

The later kill test passes only with an 8-10 point matched-pair ranking gain and
either a 1.5-2 point EM/F1 gain on both datasets or a 25%-30% context reduction
without accuracy loss. Stop if gains are synthetic-only, absent on natural
errors, lose more than half across datasets, or reflect perturbation templates.

## Closed-source API hold

No API request is authorized until the user confirms key rotation. The client
must read only these environment variables:

```text
OPENAI_BASE_URL
OPENAI_API_KEY
OPENAI_MODEL
```

After confirmation, strip a trailing slash from the base URL and append
`/chat/completions`. Send one request with single concurrency, a 60-second
timeout, `temperature=0`, `max_tokens=8`, and `Return exactly OK.` Validate
`choices[0].message.content` in memory.

Persist only status code, latency, usage, response field shape, and an
exact-match boolean. Never print or store the API key, authorization header,
request headers, base URL, or raw request/response body.

## Verification commands

```bash
cd ~/topocf-rag
source .venv/bin/activate
python --version
python scripts/capture_environment.py
python scripts/prepare_hotpot_splits.py > reports/hotpot_split_summary.json
python -m pytest -q
python -m pip check
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_bge_m3.py \
  --model /file_system/models/embedding_models/bge-m3 --device cuda:0
CUDA_VISIBLE_DEVICES=0 python scripts/analyze_title_graph.py \
  --model /file_system/models/embedding_models/bge-m3 \
  --device cuda:0 --batch-size 16 --retrieval-top-k 10
sha256sum /file_system/datasets/hotpotqa/hotpot_train_v1.1.json \
  /file_system/datasets/hotpotqa/hotpot_dev_distractor_v1.json \
  data/splits/hotpot_train_bridge_1000_ids.json \
  data/splits/hotpot_dev_bridge_500_ids.json
```

The final Phase 0 report must include command outcomes, hashes, bge-m3 smoke
metrics, graph coverage, natural-path distribution, the 60% gate decision, and
a recommendation for the next phase. It must contain no dataset content or
credentials.

The completed report is [`PHASE0_REPORT.md`](PHASE0_REPORT.md).
