#!/bin/bash
set -e

##################################
# CONFIG  (Qwen3-8B bf16 on a single NVIDIA A100 40GB)
# Sibling of run_selfcanon_iter2.sh (T4/Qwen2.5-7B-4bit); same toggles,
# A100-appropriate model + dtype. ~16GB weights -> fits even sharing the GPU.
##################################
# DATASET + PROMPT_LANG are env-overridable so wrappers (e.g. run_edu_chunk.sh) can
# target another dataset/language without editing this file. Defaults = webnlg/eng.
DATASET="${DATASET:-webnlg}"
PROMPT_LANG="${PROMPT_LANG:-eng}"

# Tags used in the output folder name (so the 18 SC variants don't clobber each
# other across runs). METHOD_STR / BASE_OUT are assembled AFTER the toggles below.
# MODEL_TAG / DATE_TAG / OIE_MODEL / EDC_LOAD_IN_4BIT are env-overridable so the OIE
# size-ablation (qwen2.5-3b / qwen3-8b / qwen3-32b) reuses THIS one script — see
# RUN_GUIDE.md and run_oie_ablation_*.sh. Defaults below = the Qwen3-8B baseline.
MODEL_TAG="${MODEL_TAG:-qwen3-8b}"
GPU_TAG="${GPU_TAG:-A100}"   # env-overridable so multi-server runs (A100/H100/…) don't share a tag
DATE_TAG="${DATE_TAG:-20260516}"
# Embedder tag (MUST match SC_EMBEDDER below) — goes in the output folder name so
# embedder sweeps (minilm / bgem3 / e5large) don't clobber each other.
EMB_TAG="${EMB_TAG:-bgem3}"        # env-overridable; must match SC_EMBEDDER

##################################
# >>> EXPERIMENT TOGGLES <<<   (edit these, nothing else)
##################################
# RUN_MODE:
#   1 = empty schema, self-generate (--no_schema)
#   2 = seeded schema + enrich       (--enrich_schema)
#   3 = seeded schema, NO enrich     (frozen schema)
# (env-overridable so wrappers can set them without editing this file)
RUN_MODE="${RUN_MODE:-1}"

# Cluster-augmented retrieval (item -> item+cluster).
# Run once false then once true to cover all 9x2 = 18 SC variants.
USE_CLUSTER="${USE_CLUSTER:-true}"
CLUSTER_SIM_THRESHOLD=0.85
CLUSTER_TOP_M=2
CLUSTER_EXTRA_K=3

# Entity-type canonicalization pass (needs schema with `parent` + ec_* templates
# in the chosen prompt_lang dir — eng AND vni now both have ec_template_option1/2/3)
ENABLE_ENTITY_CANON="${ENABLE_ENTITY_CANON:-true}"

# Coreference REWRITE before OIE. OFF by default. GraphRAG datasets ONLY — it changes
# entity surface forms so it breaks strict/exact/partial on gold-triple sets. §8.4.
#   COREF_METHOD=llm      → Qwen3 rewrite (EN+VI, reuses OIE model)
#   COREF_METHOD=maverick → Maverick (ACL'24, ENGLISH-only; pip install maverick-coref)
ENABLE_COREF="${ENABLE_COREF:-false}"
COREF_METHOD="${COREF_METHOD:-llm}"

# Resume: reuse cached stage outputs in the iter dir instead of recomputing.
#   none = full run | sd = skip OIE | sc = skip OIE+SD
# Env-overridable (RESUME_FROM=sd bash ...) so a crashed run can resume from its OWN
# already-saved oie_total.json (same --output_dir) without needing REUSE_STAGE_DIR.
RESUME_FROM="${RESUME_FROM:-none}"

# Reuse OIE+SD from a PREVIOUS run (they don't depend on mode/retrieval/case),
# then only re-run SC. Workflow for the 16 SC variants:
#   1) run once full (RESUME_FROM=none, REUSE_STAGE_DIR empty) -> base run (18 SC variants total)
#   2) for every other (mode, retrieval): set RESUME_FROM=sc and
#      REUSE_STAGE_DIR=<base run output dir>  (the ./output/<...> folder, NOT iter0)
# Leave empty for a normal full run. Env-overridable, same pattern as RESUME_FROM.
REUSE_STAGE_DIR="${REUSE_STAGE_DIR:-}"

##################################
# A100 RUNTIME ENV
##################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# GPU pick — env-overridable for SHARED servers (check `nvidia-smi` for a free GPU first,
# then e.g. CUDA_VISIBLE_DEVICES=3 bash run_track3_scale_32b.sh). Default 0 = unchanged.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# >>> QUANTIZATION TOGGLE <<<
# A100 has plenty of VRAM for 8B -> keep 4-bit OFF (code auto-picks bf16 on A100).
# Flip EDC_LOAD_IN_4BIT=1 only if you move to a much bigger model (e.g. 32B) or
# the GPU is heavily shared.
export EDC_LOAD_IN_4BIT="${EDC_LOAD_IN_4BIT:-0}"   # env-overridable (set 1 for 32B)
# export EDC_LOAD_IN_8BIT=1
# Requires (only when a quant flag is on): pip install -U bitsandbytes

##################################
# MODELS
##################################
# Per-stage LLM. By default all stages share ONE model (per-role quant mixing not
# implemented -> one model in VRAM, cached by model string): set OIE_MODEL to change all.
# For EXPLICIT per-stage control set SD_MODEL / SC_MODEL / EE_MODEL (each defaults to
# OIE_MODEL, which defaults to Qwen3-8B). NOTE: MODEL_TAG (the output-folder label) is a
# SINGLE tag -> if you mix models across stages, set MODEL_TAG to something descriptive.
OIE_LLM="${OIE_MODEL:-Qwen/Qwen3-8B}"
SD_LLM="${SD_MODEL:-${OIE_MODEL:-Qwen/Qwen3-8B}}"
SC_LLM="${SC_MODEL:-${OIE_MODEL:-Qwen/Qwen3-8B}}"

# SC embedder = the METHOD variable for the embedder sweep. Change this + EMB_TAG together.
#   minilm  -> sentence-transformers/all-MiniLM-L6-v2
#   bgem3   -> BAAI/bge-m3
#   e5large -> intfloat/e5-large-v2
SC_EMBEDDER="${SC_EMBEDDER:-BAAI/bge-m3}"   # env-overridable; pair with EMB_TAG

EE_LLM="${EE_MODEL:-${OIE_MODEL:-Qwen/Qwen3-8B}}"
SR_EMBEDDER=sentence-transformers/all-MiniLM-L6-v2
# EVAL_EMBEDDER = the metric's RULER. KEEP IT FIXED across embedder sweeps so runs stay
# comparable (do NOT set it to SC_EMBEDDER). clustering_metric uses no embedder.
EVAL_EMBEDDER=sentence-transformers/all-MiniLM-L6-v2

##################################
# REFINEMENT  (off by default)  — see RUN_GUIDE.md §8 "OIE refinement"
##################################
# END-TO-END refine (EDC+R style): re-run the whole pipeline REFINE_ITERS extra times,
# each re-extracting OIE with hints from the previous iteration. (env-overridable)
DO_REFINEMENT="${DO_REFINEMENT:-false}"
REFINE_ITERS="${REFINE_ITERS:-1}"
FREEZE_ITER="${FREEZE_ITER:-2}"

# PER-ITEM IMMEDIATE refine: re-extract each item right after its OIE pass-1 with a
# local EE-merged entity hint. Independent of DO_REFINEMENT. Reuses EE_LLM (=OIE model,
# cached -> no extra VRAM) + SR_EMBEDDER (MiniLM). Set true for the per-item ablation.
REFINE_PER_ITEM="${REFINE_PER_ITEM:-false}"

##################################
# BUILD EXTRA ARGS FROM TOGGLES
##################################
case "$RUN_MODE" in
    1) SCHEMA_ARGS="--no_schema" ;;
    2) SCHEMA_ARGS="--enrich_schema" ;;
    3) SCHEMA_ARGS="" ;;
    *) echo "Invalid RUN_MODE=$RUN_MODE (use 1/2/3)"; exit 1 ;;
esac

CLUSTER_ARGS=""
if [ "$USE_CLUSTER" = true ]; then
    CLUSTER_ARGS="--retrieval_mode item+cluster \
        --cluster_sim_threshold ${CLUSTER_SIM_THRESHOLD} \
        --cluster_top_m ${CLUSTER_TOP_M} \
        --cluster_extra_k ${CLUSTER_EXTRA_K}"
fi

EC_ARGS=""
if [ "$ENABLE_ENTITY_CANON" = true ]; then
    EC_ARGS="--enable_entity_canon"
fi

COREF_ARGS=""
COREF_TAG=""
if [ "$ENABLE_COREF" = true ]; then
    COREF_ARGS="--enable_coref --coref_method ${COREF_METHOD}"
    if [ "$COREF_METHOD" = maverick ]; then COREF_TAG="_corefmav"; else COREF_TAG="_coref"; fi
fi

REUSE_ARGS=""
if [ -n "$REUSE_STAGE_DIR" ]; then
    REUSE_ARGS="--reuse_stage_dir ${REUSE_STAGE_DIR}"
fi

# SD sampling temperature (env-overridable). 0 = greedy (default, deterministic). >0 SAMPLES
# the SD definition/typing stage → the variance source for the ≥3-seed study. OIE + SC stay greedy.
SD_TEMPERATURE="${SD_TEMPERATURE:-0.0}"
SDT_ARG="--sd_temperature ${SD_TEMPERATURE}"
SDT_TAG=""
case "$SD_TEMPERATURE" in 0|0.0|0.00|"") SDT_TAG="" ;; *) SDT_TAG="_sdt${SD_TEMPERATURE}" ;; esac

# Variance seed (env-overridable). Empty = unseeded. Only meaningful when SD samples
# (SD_TEMPERATURE>0); then run the matrix 3x with SEED=1,2,3 for mean±std.
SEED="${SEED:-}"
SEED_ARG=""
SEED_TAG=""
if [ -n "$SEED" ]; then SEED_ARG="--seed ${SEED}"; SEED_TAG="_seed${SEED}"; fi

# SC per-case LLM batching (item-13 speed-up, env-overridable). Empty/1 = OFF (default,
# byte-identical to the sequential path). >1 batches up to N LLM-cases of a triplet into
# one generate(). Tagged so a batched run lands in its OWN output dir for the
# batched-vs-off validation diff (greedy must reproduce). See DECISIONS 2026-06-27c.
SC_BATCH_SIZE="${SC_BATCH_SIZE:-}"
SCBATCH_ARG=""
SCBATCH_TAG=""
if [ -n "$SC_BATCH_SIZE" ] && [ "$SC_BATCH_SIZE" != "1" ]; then
    SCBATCH_ARG="--sc_batch_size ${SC_BATCH_SIZE}"
    SCBATCH_TAG="_scb${SC_BATCH_SIZE}"
fi

EXTRA_ARGS="${SCHEMA_ARGS} ${CLUSTER_ARGS} ${EC_ARGS} ${COREF_ARGS} ${SDT_ARG} ${SEED_ARG} ${SCBATCH_ARG} --resume_from ${RESUME_FROM} ${REUSE_ARGS}"

# Per-item refine: needs the EE model (=OIE model, cached) + SR embedder (MiniLM). Must
# override the runa defaults (Mistral-7B / e5-mistral) or it loads 2 big extra models.
REFINE_PI_ARGS=""
REFINE_TAG=""
if [ "$REFINE_PER_ITEM" = true ]; then
    REFINE_PI_ARGS="--refine_per_item --ee_llm ${EE_LLM} --sr_embedder ${SR_EMBEDDER}"
    REFINE_TAG="_refpi"
elif [ "$DO_REFINEMENT" = true ]; then
    REFINE_TAG="_refe2e${REFINE_ITERS}"
fi

##################################
# OUTPUT NAMING (encode mode + retrieval + refine so runs don't overwrite)
##################################
if [ "$USE_CLUSTER" = true ]; then RETR_TAG=itemcluster; else RETR_TAG=item; fi

METHOD_STR=selfcanon2_mode${RUN_MODE}_${RETR_TAG}_${MODEL_TAG}_${EMB_TAG}_${GPU_TAG}_${DATE_TAG}${SDT_TAG}${SEED_TAG}${REFINE_TAG}${COREF_TAG}${SCBATCH_TAG}
BASE_OUT="./output/${DATASET}_${METHOD_STR}"

##################################
# RUN
##################################
echo "======================================"
echo ">>> RUN EDC  (${GPU_TAG} / tag=${MODEL_TAG} / 4BIT=${EDC_LOAD_IN_4BIT})"
echo "Models  : OIE=${OIE_LLM} | SD=${SD_LLM} | SC=${SC_LLM} | EE=${EE_LLM}"
echo "Dataset : ${DATASET} | Prompt: ${PROMPT_LANG}"
echo "RUN_MODE: ${RUN_MODE} | CLUSTER=${USE_CLUSTER} | ENTITY_CANON=${ENABLE_ENTITY_CANON}"
echo "EXTRA   : ${EXTRA_ARGS}"
echo "Out     : ${BASE_OUT}"
echo "======================================"

if [ "$DO_REFINEMENT" = false ]; then

    python runa.py \
        --dataset_name ${DATASET} \
        --prompt_lang ${PROMPT_LANG} \
        \
        --oie_llm ${OIE_LLM} \
        --sd_llm ${SD_LLM} \
        --sc_llm ${SC_LLM} \
        --sc_embedder ${SC_EMBEDDER} \
        \
        --eval_embedder ${EVAL_EMBEDDER} \
        \
        ${EXTRA_ARGS} \
        ${REFINE_PI_ARGS} \
        --refinement_iterations 0 \
        \
        --output_dir "${BASE_OUT}"

else

    python runa.py \
        --dataset_name ${DATASET} \
        --prompt_lang ${PROMPT_LANG} \
        \
        --oie_llm ${OIE_LLM} \
        --sd_llm ${SD_LLM} \
        --sc_llm ${SC_LLM} \
        --sc_embedder ${SC_EMBEDDER} \
        \
        --ee_llm ${EE_LLM} \
        --sr_embedder ${SR_EMBEDDER} \
        \
        --eval_embedder ${EVAL_EMBEDDER} \
        \
        ${EXTRA_ARGS} \
        \
        --refinement_iterations ${REFINE_ITERS} \
        --freeze_iter ${FREEZE_ITER} \
        \
        --output_dir "${BASE_OUT}" \
        --logging_verbose

fi

##################################
# EVALUATION (separate step, after KGC)
##################################
# Only datasets with a gold KG can be evaluated intrinsically. GraphRAG-only
# datasets (eduhcmut/hotpot/trivia) have no gold KG -> evaluated via QA elsewhere.
# OIE eval runs once (single oie_total.json). Canon eval covers all 9 SC cases:
#   ALL modes           -> run_full_evaluation.py (strict/exact/partial; on Mode-1
#                          strict is name-drift-penalized -> read with aligned-F1)
#   Mode 1 (self-canon) -> + mode1_metric.py (name-drift robust) + redundancy
#   Mode 2/3 (seeded)   -> + retrieval-recall + error-attribution
# End-to-end refine writes the FINAL refined KG to iter${REFINE_ITERS}; eval that one.
# Per-item refine stays in iter0 (it modifies OIE in place, no extra iteration).
# Evaluate EVERY iteration the run produced, not just the last one: a refine run
# now gets iter0 (no-refine) AND iter1.. (refined), so the no-refine vs refined
# comparison is available without a separate eval pass. The metric suite lives in
# run_eval.sh (single source of truth; mirrors the per-mode branch logic).
if [ "$DO_REFINEMENT" = true ]; then EVAL_ITERS=$(seq 0 ${REFINE_ITERS}); else EVAL_ITERS=0; fi
for IT in ${EVAL_ITERS}; do
    DATASET="${DATASET}" METHOD="${METHOD_STR}" ITER="iter${IT}" \
        RUN_MODE="${RUN_MODE}" EVAL_EMBEDDER="${EVAL_EMBEDDER}" \
        bash run_eval.sh
done
