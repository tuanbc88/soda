#!/bin/bash
# Launch one batch of wiki-nre seed legs on the 40GB MIG box.
#   usage: bash launch_wikinre.sh <MODE> <SEED_A> [SEED_B]
# Two concurrent seeds max: 2 x 19 GB = 38 GB against a 39.4 GiB slice. A third OOMs and takes
# the others with it (RUN_02AUG_40GB_WIKINRE.md, capacity table).
set -euo pipefail

MODE=${1:?mode}
SEEDS=("${@:2}")
[ ${#SEEDS[@]} -le 2 ] || { echo "refusing: at most 2 concurrent seeds on a 40GB card"; exit 1; }

ENVBIN=/home/jovyan/miniconda3/envs/edc/bin
REPO=/home/jovyan/edc
OIE_SRC=output/wiki-nre_selfcanon2_mode1_item_qwen3-8b_bgem3_A100_0627open

# EDC_RESUME_INPLACE=1 is what makes RESUME_FROM=sd continue from the checkpoint. Without it the
# stage restarts at item 0 and the uploaded 100/50/50 are wasted.
# OMP/MKL are pinned because an unthrottled CPU stage beside a GPU job has starved it to 0 items/60s.
COMMON="MODEL_TAG=qwen3-8b OIE_MODEL=Qwen/Qwen3-8B SD_MODEL=Qwen/Qwen3-8B SC_MODEL=Qwen/Qwen3-8B \
EE_MODEL=Qwen/Qwen3-8B EMB_TAG=bgem3 SC_EMBEDDER=BAAI/bge-m3 GPU_TAG=H100 DATE_TAG=0627open \
EDC_LOAD_IN_4BIT=0 SD_TEMPERATURE=0.3 EDC_RESUME_INPLACE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
TOKENIZERS_PARALLELISM=false"

for s in "${SEEDS[@]}"; do
  sess="wikim${MODE}s${s}"
  log="/tmp/wiki_m${MODE}_seed${s}.log"
  tmux kill-session -t "$sess" 2>/dev/null || true
  # PATH and HF_HOME go INSIDE the tmux command: `tmux new -d` runs a non-interactive shell that
  # never sources the conda profile, so `python` would be /usr/bin/python and the job would die
  # instantly on ModuleNotFoundError. This killed three seeds on 28/07.
  tmux new -d -s "$sess" "export PATH=$ENVBIN:\$PATH HF_HOME=/home/jovyan/.cache/huggingface; \
cd $REPO && env $COMMON DATASET=wiki-nre RUN_MODE=$MODE USE_CLUSTER=false SEED=$s \
RESUME_FROM=sd REUSE_STAGE_DIR=$OIE_SRC \
bash run_soda.sh 2>&1 | tee $log; echo EXIT=\$? >> $log"
  echo "launched $sess -> $log"
done

sleep 25
echo
echo "=== tmux (a MISSING session did not fail to start, it started and DIED) ==="
tmux ls || echo "NO SESSIONS - check the logs"
echo
for s in "${SEEDS[@]}"; do
  echo "--- seed $s ---"
  tail -3 "/tmp/wiki_m${MODE}_seed${s}.log" 2>/dev/null || echo "(no log yet)"
done
