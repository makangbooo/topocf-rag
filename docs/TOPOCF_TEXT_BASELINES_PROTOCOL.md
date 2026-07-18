# TopoCF text-baseline protocol

Frozen: 2026-07-18

## Purpose

The binding-and-repair claim is not evaluated against a deliberately weak
semantic control. Two learned baselines share the same official Qwen
checkpoint, LoRA capacity, fit examples, pairwise loss, optimizer candidates,
and exact S4 validation protocol:

1. `flat_cross_encoder` jointly reads the question and the complete neutral
   candidate-topology serialization in one single-tower reranker call.
2. `independent_edge` receives four separate question-plus-local-edge inputs
   and averages their logits. A local input contains only that typed edge and
   its endpoint document payloads; it never receives the other candidate
   edges or a one-to-one assignment constraint.

Their only intended difference is the scoring factorization. Hidden observed
flags, provenance, generator names, labels, pair IDs, and question IDs are not
model inputs.

## Backbone and score

Bounded method development uses the official
`Qwen/Qwen3-Reranker-0.6B` checkpoint. The 4B checkpoint is reserved for a
scale confirmation only after the 0.6B method kill test passes. The model
score is the final-token `yes` logit minus the `no` logit under the official
Qwen reranker prefix and suffix. The complete prompt must fit in 1,024 tokens;
truncation is a hard error.

Both baselines use the same LoRA targets and rank, pairwise margin, gradient
accumulation, weight decay, early-stopping rule, and two frozen learning-rate
candidates. Epoch zero records the zero-shot checkpoint. Selection is by:

1. inner-validation question-macro matched-pair accuracy;
2. mean of per-question mean score margins;
3. the earlier epoch.

The exact prompt hash, data hashes, hyperparameters, seeds, and checkpoint
rule are frozen in
`configs/certificate_v1/topocf_text_baselines_v1.json`.

## Data isolation and permutation control

Only 129 official-train fit questions and their 294
`synthetic_common/t3` pairs contribute gradients. Each epoch uses the one
deterministic pair-synchronized S4 relabeling produced by
`method_data.py`. The 32-question/74-pair train-only inner validation set is
scored over all 24 synchronized relabelings; the 24 candidate logits are
averaged before comparison.

Official HotpotQA dev, all-observed confirmation pairs, natural errors, T1,
T2, fork, and double-collider pairs are inaccessible to this training script.
Official dev is evaluated once only after learning rate, epoch, and checkpoint
are frozen.

## Staged execution

First run aggregate-only input audits for both factorizations:

```bash
python scripts/train_method_baseline.py \
  --baseline flat_cross_encoder \
  --audit-only

python scripts/train_method_baseline.py \
  --baseline independent_edge \
  --audit-only
```

The audit must report 129/294 fit and 32/74 validation counts, zero official
dev use, and a maximum prompt length no larger than 1,024. It writes no dataset
text or IDs. Formal training remains code-locked until the audited combined
model fingerprint is copied into the frozen configuration in a reviewed
commit.

The 2026-07-18 audits passed with the same combined model fingerprint
`01f807839563e5e18293e9498f59e5e025ecd134fcf8a1cd2076e840faf8b4fb`
for both factorizations. Flat inputs peaked at 677 tokens and independent-edge
inputs at 519 tokens, so the 1,024-token hard limit preserves every audited
input. The fingerprint and every constituent file digest are now frozen in
the configuration.

Then run one bounded smoke for each baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_method_baseline.py \
  --baseline flat_cross_encoder \
  --learning-rate 0.0001 \
  --smoke

CUDA_VISIBLE_DEVICES=0 python scripts/train_method_baseline.py \
  --baseline independent_edge \
  --learning-rate 0.0001 \
  --smoke
```

Smoke runs use eight fit pairs, two validation pairs, one epoch, and a reduced
bootstrap. They test execution only and are never paper results. Formal runs
use both frozen learning-rate candidates for seed `20260718`; after choosing
one learning rate per baseline on inner validation, repeat that configuration
for seeds `20260719` and `20260720`.

The seed-`20260718` selection is now frozen. The flat cross-encoder selected
learning rate `1e-4` and epoch 1 (pairwise accuracy `0.6962`). The independent
local-edge baseline selected learning rate `2e-4` and epoch 4 after its two
learning-rate candidates tied at pairwise accuracy `1.0` and the frozen
secondary margin rule preferred `2e-4`. The aggregate-only selection evidence,
report hashes, run fingerprints, and checkpoint hashes are recorded in
`reports/phase1/text_baseline_selection.json`.

The remaining seeds are replications, not additional checkpoint-selection
runs. The script rejects a non-selection seed unless `--replicate-selected` is
present, the selected learning rate is supplied, and the frozen epoch is used:

```bash
for seed in 20260719 20260720; do
  CUDA_VISIBLE_DEVICES=0 python scripts/train_method_baseline.py \
    --baseline flat_cross_encoder \
    --learning-rate 0.0001 \
    --seed "$seed" \
    --replicate-selected

  CUDA_VISIBLE_DEVICES=0 python scripts/train_method_baseline.py \
    --baseline independent_edge \
    --learning-rate 0.0002 \
    --seed "$seed" \
    --replicate-selected
done
```

During replication, inner-validation metrics are still reported for stability,
but they cannot select a different epoch. Only the frozen final epoch is saved.

Both fixed independent-edge replications reproduced pairwise accuracy `1.0`:

| Seed | Frozen epoch | Pairwise accuracy | AUROC | Mean margin |
|---:|---:|---:|---:|---:|
| 20260718 | 4 | 1.0000 | 0.9985 | 2.6112 |
| 20260719 | 4 | 1.0000 | 0.9994 | 2.0960 |
| 20260720 | 4 | 1.0000 | 1.0000 | 2.2165 |

The corresponding flat scores were `0.6962`, `0.5145`, and `0.5934`. Thus the
ceiling is specific to decomposed local-edge grounding rather than a universal
property of the reranker. The aggregate-only hashes and metrics are frozen in
`reports/phase1/text_baseline_selection.json`. This confirms the train-only
stop signal and pauses global binding-model training.

### One-shot official-dev confirmation

Before redesigning T3, all three independent-edge adapters receive exactly one
external evaluation on the untouched 65-question/150-pair official-dev T3
stratum. `scripts/evaluate_method_baseline_dev.py` derives the checkpoint path
from the frozen baseline, learning rate, seed, and epoch; it accepts no free
checkpoint path and verifies the adapter tree hash before loading official dev.
It performs no training, ensembling, thresholding, or checkpoint selection.

The decision rule is frozen before evaluation:

- stop the current method if every seed reaches at least `0.90` question-macro
  pairwise accuracy and their mean is at least `0.95`;
- treat a three-seed mean below `0.80` as an external contradiction;
- treat every other outcome as ambiguous and diagnose it without training the
  global binding model.

Run all three seeds consecutively; do not inspect one seed and conditionally
omit the others:

```bash
for seed in 20260718 20260719 20260720; do
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_method_baseline_dev.py \
    --baseline independent_edge \
    --seed "$seed"
done
```

Each output path is write-once below
`/home/mkb524/topocf-rag-runs/text-baseline-dev-v1`. Existing output causes a
hard failure. Flat official-dev scoring is deferred because it is unnecessary
for the local-decomposability stop decision.

Checkpoints and reports are written below
`/home/mkb524/topocf-rag-runs/text-baselines-v1` with mode-restricted parent
directories. They must not be committed. Reports contain aggregate metrics,
runtime metadata, paths, and hashes only.

## Stop conditions

Do not evaluate official dev if either baseline cannot complete exact S4
validation, silently truncates input, or changes the frozen data/prompt hash.
Do not claim a constrained-binding contribution unless the final method beats
both learned baselines and yields a positive structured-repair result. A win
only over BM25/bge-m3 is insufficient.

The three-seed independent-edge ceiling confirms the train-only stop signal.
Do not train the global binding model while the one-shot official-dev result is
pending. If the frozen external gate also stops, redesign the counterfactual so
every local edge is individually plausible and only their global composition
distinguishes the positive from the negative before method development resumes.
