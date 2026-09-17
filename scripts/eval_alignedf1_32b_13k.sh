#!/bin/bash
# Aligned-F1 for the 32B scale row: the 13k cell, plus a 6k CROSS-CHECK.
#
# Only this metric was missing after the 12/08 slice-back: relation B-cubed, #rel, entity #ent and
# entity B-cubed all compute locally, but mode1_metric imports sentence-transformers, which the
# Windows box does not have. So it runs on p247-80 while that booking is still ours and idle.
#
# CPU-ONLY on purpose: no GPU is needed for an embedding alignment over a few hundred relation names,
# and the box may be handed to another project at any time.
#
# The 6k arm is not redundant. tab:scale already prints 0.318 for phi=1 at the 6k cut, so if this
# reproduces that number the 13k figures come from a chain that is known to agree with the paper.
# If it does NOT reproduce it, the 13k numbers must not be reported.
set -euo pipefail

BASE=${BASE:-/home/jovyan}
export PATH="$BASE/miniconda3/envs/edc/bin:$PATH"
export HF_HOME=${HF_HOME:-$BASE/.cache/huggingface}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""

cd "$BASE/edc"
date -u +%Y-%m-%dT%H:%M:%SZ

# --embedder and --rel_threshold are left at their defaults deliberately: MiniLM is the metric's
# fixed ruler (run_eval.sh pins EVAL_EMBEDDER to it) and 0.5 is the operating point every
# reported aligned-F1 uses. Passing anything else here would silently change the metric.

echo "=== CROSS-CHECK: 6k cut (tab:scale prints 0.318 for phi=1) ==="
nice -n 19 python soda/evaluate/mode1_metric.py \
  --gt_kg soda/evaluate/references/webnlg_full_6k.txt \
  --pred_dir output/webnlg_full_6k_selfcanon2_mode1_item_qwen3-32b_bgem3_H100_20260627open/iter0 \
  --output_dir /tmp/xcheck_6k_mode1

echo
echo "=== TARGET: 13k cut (the empty column of tab:scale) ==="
nice -n 19 python soda/evaluate/mode1_metric.py \
  --gt_kg soda/evaluate/references/webnlg_full_full.txt \
  --pred_dir output/webnlg_full_full_selfcanon2_mode1_item_qwen3-32b_bgem3_H100_20260627open/iter0 \
  --output_dir output/webnlg_full_full_selfcanon2_mode1_item_qwen3-32b_bgem3_H100_20260627open/iter0/eval_mode1

echo "ALIGNEDF1_DONE"
date -u +%Y-%m-%dT%H:%M:%SZ
