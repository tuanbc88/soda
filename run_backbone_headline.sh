#!/bin/bash
# Backbone-headline leg — Qwen2.5-7B at the EXACT headline config, all 3 gold benchmarks.
#
# WHY: T3 (the T4 size-sweep) showed Qwen2.5-7B beating the Qwen3-8B headline by ~+0.05 strict
# on webnlg Mode-3 -- but at 4-bit + minilm, so it is not headline-comparable. This leg re-runs
# Qwen2.5-7B at bf16 + bge-m3 so the ONLY variable vs the headline is the backbone, which is what
# licenses a paper claim (the tab:edc "SODA (Qwen2.5-7B backbone, no-refine)" row).
#
# NOTE: OIE_MODEL cascades to SD/SC/EE (run_soda.sh L92-94/102) -- that is
# intended here: we are swapping the whole shared backbone (see DECISIONS 2026-07-15).
#
# Config vs the headline (Qwen3-8B): IDENTICAL except the backbone.
#   Mode 3 / item / bf16 / bge-m3 / greedy / EVAL_EMBEDDER fixed (minilm) / DATE_TAG=0627open
#
# Output dirs (distinct from the headline via MODEL_TAG, so nothing is clobbered):
#   output/{webnlg,rebel,wiki-nre}_selfcanon2_mode3_item_qwen2.5-7b_bgem3_A100_0627open/
#   (headline for comparison:  ..._mode3_item_qwen3-8b_bgem3_A100_0627open/)
#
# Usage
#   bash run_backbone_headline.sh                          # all 3 datasets, sequential
#   DATASETS=rebel bash run_backbone_headline.sh           # one dataset
#   DATASETS=rebel RESUME_FROM=sd bash run_backbone_headline.sh   # resume a died run
#   MODEL_TAG=sailor2-8b OIE_MODEL=sail/Sailor2-8B-Chat bash run_backbone_headline.sh  # other backbone
#
# Runtime (A100, bf16, ~7B): webnlg ~15h + rebel ~16h + wiki-nre ~11h = ~42h total.
# VRAM: 7B bf16 ~15GB + bge-m3 ~2GB -> needs >=24GB (A100-40GB fine; 20GB card is too tight).
# Eval is AUTOMATIC (the runner calls run_eval.sh per mode) -- no separate eval step.

MODEL="${OIE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TAG="${MODEL_TAG:-qwen2.5-7b}"
DATASETS="${DATASETS:-webnlg rebel wiki-nre}"
GPU_TAG="${GPU_TAG:-A100}"
DATE_TAG="${DATE_TAG:-0627open}"
RESUME_FROM="${RESUME_FROM:-none}"

echo "=============================================================="
echo " backbone-headline leg"
echo "   backbone : $MODEL   (tag=$TAG)"
echo "   config   : Mode-3 / item / bf16 / bge-m3 / greedy   (headline-matched)"
echo "   datasets : $DATASETS"
echo "   date tag : $DATE_TAG        resume: $RESUME_FROM"
echo "   -> output/<ds>_selfcanon2_mode3_item_${TAG}_bgem3_${GPU_TAG}_${DATE_TAG}/"
echo "=============================================================="

for DS in $DATASETS; do
  echo ""
  echo "===== $(date '+%F %H:%M') backbone-headline: $TAG on $DS ====="
  GPU_TAG="$GPU_TAG" \
  MODEL_TAG="$TAG" \
  OIE_MODEL="$MODEL" \
  EDC_LOAD_IN_4BIT=0 \
  DATASET="$DS" \
  EMB_TAG=bgem3 \
  SC_EMBEDDER=BAAI/bge-m3 \
  DATE_TAG="$DATE_TAG" \
  RUN_MODE=3 \
  USE_CLUSTER=false \
  RESUME_FROM="$RESUME_FROM" \
      bash "$(dirname "$0")/run_soda.sh" \
    && echo ">>> DONE $DS" \
    || echo ">>> FAILED $DS  (fix, then: DATASETS=$DS RESUME_FROM=<stage> bash run_backbone_headline.sh)"
done

echo ""
echo "BACKBONE_HEADLINE_ALL_DONE"
echo "Compare against the headline with:"
echo "  python scripts/compare_backbone.py"
