#!/bin/bash
# Re-run the tau_a sweep after the 2026-08-10 fixes to soda/evaluate/tau_sweep_aligned_f1.py
# (ruler pinned to MiniLM, range widened to bracket the reported tau_a=0.5).
#
# The first run of this sweep used bge-m3 and swept 0.70-0.90; both were wrong, so its output
# under output/_tau_sweep/ answers a different question and is superseded by _tau_sweep_v2/.
#
# CPU-only and throttled, because it runs BESIDE the Q1 MiniLM SC job on the same box.
set -euo pipefail

BASE=${BASE:-/home/jovyan}
export PATH="$BASE/miniconda3/envs/edc/bin:$PATH"
export HF_HOME=${HF_HOME:-$BASE/.cache/huggingface}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""

cd "$BASE/edc"
date -u +%Y-%m-%dT%H:%M:%SZ

for ds in webnlg rebel wiki-nre; do
  echo "--- tau sweep v2: $ds ---"
  nice -n 19 python soda/evaluate/tau_sweep_aligned_f1.py \
    --dataset "$ds" \
    --pred_dir "output/${ds}_selfcanon2_mode1_item_qwen3-8b_bgem3_A100_0627open/iter0" \
    --out_dir  "output/_tau_sweep_v2/${ds}_mode1"
done

echo "TAU2_DONE"
date -u +%Y-%m-%dT%H:%M:%SZ
