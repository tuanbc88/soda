#!/bin/bash
# Bring a bare GPU pod to a runnable SODA environment. Proven end to end on the 40GB MIG box on
# 2026-08-04 (setup to import-gate in ~5 minutes, smoke EXIT 0).
#
#   usage:  BASE=/home/jovyan MODELS="Qwen/Qwen3-8B BAAI/bge-m3" bash pod_setup_soda.sh
#
# BASE must be on a NON-EPHEMERAL disk. On these pods /root and /workspace are the container's
# overlay rootfs: a restart takes everything on them, which cost a sibling project a full day and
# ~3.5 h of finished compute on 2026-08-01. Check with `df -h` / `findmnt` before choosing, and read
# H100_POD_PLAYBOOK.md section 1b. The repo is expected at $BASE/edc (tar-pipe it in separately).
set -euo pipefail

BASE=${BASE:-/home/jovyan}
MODELS=${MODELS:-"Qwen/Qwen3-8B BAAI/bge-m3"}
MC=$BASE/miniconda3
ENVBIN=$MC/envs/edc/bin
export HF_HOME=$BASE/.cache/huggingface

step() { echo; echo "=== $* ==="; date -u +%H:%M:%SZ; }

step "1/6 miniconda -> $MC"
# An existing miniconda here may belong to another project on a shared box. Reusing it is fine and
# saves the download; our env lives under envs/edc and removes cleanly with `conda env remove -n edc`.
if [ ! -x "$MC/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$MC"
else
  echo "reusing the miniconda already installed here"
fi
"$MC/bin/conda" --version

step "2/6 conda env py3.9"
# Plain `conda create` stops on a Terms-of-Service prompt. Never accept legal terms on the owner's
# behalf: -c conda-forge --override-channels sidesteps it and is the standard scientific channel.
[ -x "$ENVBIN/python" ] || "$MC/bin/conda" create -y -n edc python=3.9 -c conda-forge --override-channels
"$ENVBIN/python" --version

step "3/6 pip stack"
# This is the stack a working box runs, NOT the repo's environment.yml, which is stale (pre-Qwen3)
# and produces an env that cannot load the models.
P="$ENVBIN/pip"
$P install -q --no-input --root-user-action=ignore torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
$P install -q --no-input --root-user-action=ignore \
    transformers==4.57.6 tokenizers==0.22.2 accelerate==1.10.1 \
    bitsandbytes==0.48.2 sentence-transformers==3.0.1
# The third line is the one every earlier doc omitted, and none of it is optional:
#   datasets  - sentence_transformers.model_card imports it, and the repo's own datasets/ directory
#               shadows it otherwise ("cannot import name 'Dataset' from 'datasets' (unknown
#               location)"; "(unknown location)" is the namespace-package tell)
#   openai    - soda/utils/llm_utils.py imports it at module scope even for pure-HF runs
#   evaluate  - soda/utils/e5_mistral_utils.py, same
$P install -q --no-input --root-user-action=ignore \
    pandas openpyxl scikit-learn nervaluate tqdm huggingface_hub \
    datasets openai evaluate peft nltk unidecode ujson beautifulsoup4 matplotlib

step "4/6 NLTK punkt_tab"
# NLTK >= 3.9 renamed punkt -> punkt_tab and pip does not ship it. Without it the closed-schema eval
# scores EVERY item 0 and raises nothing: strict/exact/partial all 0.000 while B-cubed looks normal.
# A smoke test on a gold-free dataset does not exercise this path, so it will not catch it either.
"$ENVBIN/python" - <<'PY'
import nltk
for r in ("punkt_tab", "punkt"):
    nltk.download(r, quiet=True)
nltk.data.find("tokenizers/punkt_tab")
print("[ok] punkt_tab present")
PY

step "5/6 CRLF fix + import gate"
if [ -d "$BASE/edc" ]; then
  cd "$BASE/edc"
  # A Windows checkout carries \r\n through the tar-pipe and bash dies on line 1 with
  # "syntax error near unexpected token $'in\r'". Only .sh: Python tolerates CRLF and data files
  # must not be rewritten blindly.
  find . -name "*.sh" -print0 | xargs -0 sed -i 's/\r$//'
  echo "CRLF stripped from $(find . -name '*.sh' | wc -l) shell scripts"
  PATH="$ENVBIN:$PATH" "$ENVBIN/python" -c "from soda.edc_framework import EDC; print('[ok] SODA imports')"
else
  echo "!! $BASE/edc not found - tar-pipe the repo in, then re-run this step"
  exit 1
fi

step "6/6 models: $MODELS"
# Niced because this is pure download/disk and may run beside someone else's GPU job.
HF_HOME=$HF_HOME nice -n 19 "$ENVBIN/python" - <<PY
from huggingface_hub import snapshot_download
for m in "$MODELS".split():
    snapshot_download(m, max_workers=8); print("[ok]", m)
PY

step "versions"
"$ENVBIN/python" - <<'PY'
import torch, transformers, tokenizers, accelerate, sentence_transformers as st
print("torch       ", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("tokenizers  ", tokenizers.__version__)
print("accelerate  ", accelerate.__version__)
print("sent-transf ", st.__version__)
PY

echo
echo "POD_SETUP_DONE   env=$ENVBIN   HF_HOME=$HF_HOME"
echo "Remember at RUN time: export PATH=$ENVBIN:\$PATH and HF_HOME=$HF_HOME"
echo "INSIDE the tmux command -- 'tmux new -d' does not source the conda profile."
date -u +%H:%M:%SZ
