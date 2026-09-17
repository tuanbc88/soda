#!/bin/bash
# =============================================================================
# SMOKE TEST — small model, 4-bit, fits a 16GB card. Validates the 9-case pipeline
# end-to-end AND the new token-metering (stage_timing.json must carry a token
# block). NOT a results run — example2 has no gold KG, so no accuracy is judged.
#
# Goal checklist this verifies:
#   1. pipeline runs clean on a tiny corpus (example2, 6 items), Mode-1 self-canon,
#      all 9 SC cases + entity-canon, greedy/deterministic;
#   2. iter0/stage_timing.json exists and contains: oie_tokens / sd_tokens /
#      sc_tokens / ec_tokens / tokens_total / tokens_per_item with total_tokens>0.
#
# Run:   bash run_smoke.sh
# Smaller/bigger model:  MODEL=Qwen/Qwen3-1.7B bash run_smoke.sh
# Needs (4-bit): pip install -U bitsandbytes
# =============================================================================
set -e

# ---- small model, 4-bit -------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-4B}"          # override to Qwen3-1.7B for an even faster check
EMBEDDER="${EMBEDDER:-sentence-transformers/all-MiniLM-L6-v2}"   # light, English (example2 is EN)
DATASET="${DATASET:-example2}"
PROMPT_LANG="${PROMPT_LANG:-eng}"
DATE_TAG="${DATE_TAG:-smoke}"
OUT="./output/${DATASET}_smoke_${DATE_TAG}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EDC_LOAD_IN_4BIT="${EDC_LOAD_IN_4BIT:-1}"   # 4-bit so it fits a 16GB card with room to spare

echo "======================================"
echo ">>> SMOKE TEST (${MODEL} / 4bit=${EDC_LOAD_IN_4BIT})"
echo "Dataset: ${DATASET} (${PROMPT_LANG}) | Mode-1 self-canon | 9 cases + entity-canon | greedy"
echo "Out    : ${OUT}"
echo "======================================"

# ---- construction only (Mode-1, item retrieval, entity-canon on, greedy) ------
python runa.py \
    --dataset_name "${DATASET}" \
    --prompt_lang "${PROMPT_LANG}" \
    --oie_llm "${MODEL}" \
    --sd_llm "${MODEL}" \
    --sc_llm "${MODEL}" \
    --sc_embedder "${EMBEDDER}" \
    --eval_embedder "${EMBEDDER}" \
    --no_schema \
    --enable_entity_canon \
    --sd_temperature 0.0 \
    --resume_from none \
    --refinement_iterations 0 \
    --output_dir "${OUT}"

# ---- verify the token block landed in stage_timing.json -----------------------
echo "======================================"
echo ">>> VERIFY token-metering in stage_timing.json"
echo "======================================"
python - "$OUT/iter0/stage_timing.json" <<'PY'
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p, encoding="utf-8"))
except Exception as e:
    print(f"[FAIL] cannot read {p}: {e}"); sys.exit(1)

need_stage = ["oie_tokens", "sd_tokens", "sc_tokens"]   # ec_tokens only if entity-canon ran
need_agg   = ["tokens_total", "tokens_per_item"]
missing = [k for k in need_stage + need_agg if k not in d]
if missing:
    print(f"[FAIL] stage_timing.json missing keys: {missing}")
    print("present keys:", sorted(d.keys())); sys.exit(1)

tot = d["tokens_total"]
print("tokens_total   :", tot)
print("tokens_per_item:", d["tokens_per_item"])
for k in ("oie_tokens","sd_tokens","sc_tokens","ec_tokens"):
    if k in d: print(f"  {k:11s}:", d[k])
rt = d.get("runtime", {})
print("runtime        : gpu=%s peak_alloc=%sGB ram_used=%sGB dtype=%s" % (
    rt.get("gpu"), rt.get("gpu_peak_alloc_gb"), rt.get("ram_used_gb"), rt.get("dtype")))
print("timing (s)     : oie=%s sd=%s sc=%s ec=%s" % (
    d.get("oie_sec"), d.get("sd_sec"), d.get("sc_sec"), d.get("ec_sec")))

if tot.get("total_tokens", 0) <= 0:
    print("[FAIL] total_tokens is 0 — metering did not record."); sys.exit(1)
print("\n[PASS] token-metering works; pipeline produced a token-annotated stage_timing.json.")
PY

echo ">>> Also inspect (optional): $OUT/iter0/canon_kg_case7_detail_typed.txt and pipeline_log.json"
