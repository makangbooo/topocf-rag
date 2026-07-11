# TopoCF-RAG

Topology-Counterfactual Evidence Path Verification for Multi-Hop GraphRAG.

This project tests whether a path verifier can distinguish a valid multi-hop
chain from a topology-only counterfactual: the question, document texts,
entities, path size, relation-type multiset, and token budget are matched, but
the direction or bridge topology no longer supports the question. The initial
target is a short HotpotQA kill test, not a ground-up GraphRAG rewrite.

## Phase 0 objective

Phase 0 establishes a reproducible environment and determines whether the
minimal title-mention graph has enough gold-path coverage to justify training.
It performs the following work only:

1. Create an isolated Python 3.11 environment and record its package and GPU
   state.
2. Stream-inspect the official HotpotQA train and dev-distractor files without
   emitting example text.
3. Freeze deterministic, level-stratified bridge-only ID sets: 1,000 official
   train items and 500 official dev-distractor items, seed `20260711`.
4. Test title normalization, query-document graph construction, and split/data
   invariants.
5. Encode 20 non-sensitive texts with local bge-m3 on the A100 and record output
   shape, elapsed time, and peak GPU memory.
6. Measure directed and direction-ignored gold title-mention coverage and the
   natural-negative candidate count distribution.

No verifier is trained in Phase 0. Directed coverage below 60% is a hard stop:
report the result and redesign the graph before any path training.

## Resources

- HotpotQA: `/file_system/datasets/hotpotqa`
- bge-m3 requested location: `/file_system/models/bge-m3`
- bge-m3 resolved location:
  `/file_system/models/embedding_models/bge-m3`
- Project: `~/topocf-rag`
- GPU: available NVIDIA A100-SXM4-80GB

The requested bge-m3 location was absent during bootstrap; the resolved local
directory above is the path Phase 0 must use. Dataset scripts should discover
the concrete filenames beneath the HotpotQA root without recursively printing
file content.

## Data products

Phase 0 writes the selected IDs, not source examples, to:

```text
data/splits/hotpot_train_bridge_1000_ids.json
data/splits/hotpot_dev_bridge_500_ids.json
```

The Phase 0 report must include SHA256 digests of the two source JSON files and
the two ID files. Train and dev remain in their official splits. Public answers
and supporting facts may be used; the project does not add human labels.

## Minimal graph

The graph contains a question node `q`, context-document nodes `d_i`, bge-m3
retrieval edges `q -> d_i`, and title-mention edges `d_i -> d_j`. A mention edge
exists only when the source document body contains the deterministically
normalized target title. Candidate paths have exactly two edges:
`q -> d_i -> d_j`.

Positive paths cover both gold supporting titles and contain a real mention
connection. Natural negatives must be real retrieved paths with a real mention
edge, must not cover both gold titles, and must match positive paths in length
while being as close as possible in retrieval score and token count. Natural
edges are never manually corrupted.

## Reproduce Phase 0

```bash
cd ~/topocf-rag
python3.11 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

# Environment and dependency record
python scripts/capture_environment.py

# Streaming inspection, deterministic split freeze, and validation
python scripts/prepare_hotpot_splits.py > reports/hotpot_split_summary.json
python -m pytest -q
python -m pip check

# GPU smoke test and graph diagnostics
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_bge_m3.py \
  --model /file_system/models/embedding_models/bge-m3 --device cuda:0
CUDA_VISIBLE_DEVICES=0 python scripts/analyze_title_graph.py \
  --model /file_system/models/embedding_models/bge-m3 \
  --device cuda:0 --batch-size 16 --retrieval-top-k 10
```

Do not run the closed-source API smoke test until key rotation has been
confirmed. Its endpoint, model, and credential must come exclusively from
`OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_API_KEY`; see
[`docs/HANDOFF.md`](docs/HANDOFF.md) for the exact safety contract.

## Kill-test evaluation

After the 60% graph-coverage gate passes, later phases must evaluate matched
topology counterfactuals, a held-out perturbation type, and natural retrieval
errors separately. The semantic baselines are BM25 and bge-m3 cosine over an
order- and direction-preserving path serialization. A result that improves only
on synthetic perturbations is a stop signal, not evidence of the thesis claim.

The second dataset is 2WikiMultiHopQA and is intentionally deferred until the
HotpotQA kill test passes.

The completed bootstrap results and limitations are in
[`docs/PHASE0_REPORT.md`](docs/PHASE0_REPORT.md). The machine-readable graph
statistics, including full natural-candidate histograms, are in
`reports/title_graph/analysis.json`.

## Phase 1 design gate

Phase 1a audited the requested T1/T2/T3 generators before freezing formal
pairs. It found that T1 and T2 have the same canonical typed collider topology,
while a valid directed T3 two-switch requires four documents and cannot be
represented by the current two-document `CandidatePath`. Formal pair
generation, Phase 1 bge-m3 scoring, and verifier training are therefore paused
until a common input representation is selected.

The implemented support includes validated `EvidenceTopology` objects,
canonical signatures, strict matched-pair checks, question-local BM25, and
question-macro ranking metrics with analytic tie handling. Reproduce the
aggregate audit without emitting dataset text:

```bash
cd ~/topocf-rag
.venv/bin/python scripts/audit_phase1_design.py
.venv/bin/python -m pytest -q
```

The decision and continuation options are documented in
[`docs/PHASE1_DESIGN_GATE.md`](docs/PHASE1_DESIGN_GATE.md). Aggregate counts
and input hashes are in `reports/phase1/design_audit.json`.
