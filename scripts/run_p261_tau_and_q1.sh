#!/bin/bash
# The two ND01 leftovers that still needed a server, on box p261-20 (10-13/08 spare).
#
#   bash scripts/run_p261_tau_and_q1.sh
#
# JOB 1 - tau_a sensitivity sweep (review L2, minor #7). CPU ONLY.
#   Answers whether the Mode-1 aligned-F1 signal RANKING is stable in the metric's free
#   parameter. A ranking that flips with tau_a would not be a finding about signals, so this
#   is a claim-integrity check, not a number for a table. Reads canon_kg_*.txt, re-runs no
#   pipeline. It is open only because mode1_metric imports sentence-transformers, which the
#   Windows box does not have.
#
# JOB 2 - Q1 Mode-1 MiniLM leg. GPU, SC only, ~3 h.
#   The manuscript answers Q1 (is the canon flip retriever-invariant?) on the SEEDED side
#   only, and says so. The Mode-1 side is unchecked because the sole webnlg Mode-1 MiniLM run
#   is `_20260516`, the v1 EIGHT-case run -- not comparable to the v2 nine-case grid.
#   This adds the missing v2 MiniLM cell by reusing the bge-m3 run's OIE+SD and re-running
#   only SC, which is what makes it 3 h instead of 30.
#
# Both write to NEW directories; neither can overwrite a reported run.
set -euo pipefail

BASE=${BASE:-/home/jovyan}
ENVBIN=$BASE/miniconda3/envs/edc/bin
export PATH="$ENVBIN:$PATH"
export HF_HOME=${HF_HOME:-$BASE/.cache/huggingface}
# Throttled per the standing rule: an unthrottled CPU stage once took a co-resident GPU job
# to 0 items/60 s. Nothing else should be on this box, but the cost of the flags is zero.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

cd "$BASE/edc"
date -u +%Y-%m-%dT%H:%M:%SZ

############################ JOB 1 : tau_a sweep (CPU) ############################
echo "=== JOB 1/2 : tau_a sweep, 3 datasets, CPU-only ==="
# CUDA_VISIBLE_DEVICES="" so a CPU job cannot silently take VRAM that job 2 needs.
for ds in webnlg rebel wiki-nre; do
  echo "--- tau sweep: $ds ---"
  CUDA_VISIBLE_DEVICES="" nice -n 19 python soda/evaluate/tau_sweep_aligned_f1.py \
    --dataset "$ds" \
    --pred_dir "output/${ds}_selfcanon2_mode1_item_qwen3-8b_bgem3_A100_0627open/iter0" \
    --out_dir  "output/_tau_sweep/${ds}_mode1"
done
echo "JOB1_DONE"
date -u +%Y-%m-%dT%H:%M:%SZ

############################ JOB 2 : Q1 Mode-1 MiniLM (GPU) ############################
echo "=== JOB 2/2 : Q1 Mode-1 MiniLM leg, SC-only ==="
# REUSE_STAGE_DIR points at the BASE dir (NOT iter0) of the bge-m3 Mode-1 run, so OIE+SD are
# read from there and only SC re-runs. EMB_TAG must change with SC_EMBEDDER or the output
# directory would collide with the bge-m3 run it is being compared against.
# bf16 deliberately: the comparison is only controlled if precision matches the bge-m3 cell.
# On a 20GB slice 8B bf16 is TIGHT (~19 GB) -- if this OOMs, do NOT switch to 4-bit, that
# would confound the retriever question with a quantization change. Move it to a 40GB box.
set +e
REUSE_STAGE_DIR=./output/webnlg_selfcanon2_mode1_item_qwen3-8b_bgem3_A100_0627open \
RESUME_FROM=sc \
MODEL_TAG=qwen3-8b OIE_MODEL=Qwen/Qwen3-8B \
EMB_TAG=minilm SC_EMBEDDER=sentence-transformers/all-MiniLM-L6-v2 \
GPU_TAG=H100 DATE_TAG=0627open \
DATASET=webnlg RUN_MODE=1 USE_CLUSTER=false \
EDC_LOAD_IN_4BIT=0 \
bash run_soda.sh
RC=$?
set -e

echo "EXIT=$RC"
date -u +%Y-%m-%dT%H:%M:%SZ
exit $RC
