#!/bin/bash
# The 32B scale cell (webnlg_full 13,211) on box p247-80, 10-13/08 window.
# This is the script the tab:scale / tab:scale_ent 32B numbers must trace back to.
#
#   bash scripts/run_scale32b_p247.sh
#
# It resumes SD from item 6,400 rather than restarting. Two flags do that together and
# BOTH are required:
#   RESUME_FROM=sd        -> skip the completed OIE stage (13,211 items already saved)
#   EDC_RESUME_INPLACE=1  -> continue SD from the checkpoint instead of item 0.
# Without the second one, `*_total.json` is treated as a crash checkpoint rather than
# resume state and SD redoes 6,400 items -- about 32 h burned for nothing.
#
# The checkpoint is found BY PATH, so every tag below that feeds the output directory
# name is load-bearing. The name this must produce, byte for byte, is
#   output/webnlg_full_full_selfcanon2_mode1_item_qwen3-32b_bgem3_H100_20260627open/iter0/
# Do not "tidy" a tag: one wrong character restarts the run silently.
#
# SD is greedy here (SD_TEMPERATURE deliberately unset). The 5b scale curve is
# same-precision and same-decoding across 4B/8B/32B by design; a sampled SD would also
# add an _sdt tag to the directory name and miss the checkpoint.
set -euo pipefail

BASE=${BASE:-/home/jovyan}
ENVBIN=$BASE/miniconda3/envs/edc/bin

# 'tmux new -d' starts a NON-interactive shell that never sources the conda profile, so
# `python` would resolve to /usr/bin/python and runa.py would die instantly on
# ModuleNotFoundError: sentence_transformers. This killed three rebel seeds on 28/07 and
# left an H100 idle for 40 minutes. Hence PATH here, inside what tmux actually runs.
export PATH="$ENVBIN:$PATH"
# HF_HOME must be set at RUN time, not only while downloading, or the job ignores the
# 66 GB cache and re-pulls 62 GB into ephemeral storage.
export HF_HOME=${HF_HOME:-$BASE/.cache/huggingface}
# A CPU stage at 139 threads once dropped a co-resident GPU job to 0 items/60s.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

cd "$BASE/edc"

echo "=== p247-80 :: Qwen3-32B scale cell ==="
date -u +%Y-%m-%dT%H:%M:%SZ
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# set -e must NOT swallow the exit marker: a failing run has to REPORT that it failed.
# The monitor greps for `EXIT=`, and a process count is the thing it replaced -- pgrep -fc
# prints 0 and exits 1, so a watcher keyed on it never fires and a dead job reads as a
# running one (that cost ~2 h of idle H100 once).
set +e
MODEL_TAG=qwen3-32b \
OIE_MODEL=Qwen/Qwen3-32B SD_MODEL=Qwen/Qwen3-32B \
SC_MODEL=Qwen/Qwen3-32B EE_MODEL=Qwen/Qwen3-32B \
EMB_TAG=bgem3 SC_EMBEDDER=BAAI/bge-m3 \
GPU_TAG=H100 DATE_TAG=20260627open \
EDC_LOAD_IN_4BIT=0 \
DATASET=webnlg_full_full RUN_MODE=1 USE_CLUSTER=false \
RESUME_FROM=sd EDC_RESUME_INPLACE=1 \
bash run_soda.sh
RC=$?
set -e

echo "EXIT=$RC"
date -u +%Y-%m-%dT%H:%M:%SZ
exit $RC
