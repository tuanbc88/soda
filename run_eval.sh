#!/bin/bash
# =============================================================================
# Evaluate ONE iteration of a finished run. Single source of truth for the eval
# suite (the main run scripts call this per-iteration). Mirrors the eval logic
# that used to be inlined in run_soda.sh.
#
# Usage (env-parameterized):
#   DATASET=webnlg \
#   METHOD=selfcanon2_mode3_item_qwen3-8b_bgem3_A100_20260622_refe2e1 \
#   ITER=iter0 RUN_MODE=3 bash run_eval.sh
#
# METHOD = the folder name AFTER "<dataset>_" (i.e. output/<DATASET>_<METHOD>/<ITER>).
# RUN_MODE: 1 = self-canon (full + mode1 + redundancy); 2/3 = seeded (full + recall + error-attr).
# run_full_evaluation (Table-1) runs for ALL modes since 2026-07-02 (M1 strict is
# name-drift-penalized but needed for the Mode3-Mode1 gap; fair M1 = aligned-F1/B3).
# Only webnlg/rebel/wiki-nre have a gold KG; other datasets get intrinsic-only.
# =============================================================================
set -u

DATASET="${DATASET:?set DATASET=webnlg|rebel|wiki-nre|...}"
METHOD="${METHOD:?set METHOD=<folder name after '<dataset>_'>}"
ITER="${ITER:-iter0}"
RUN_MODE="${RUN_MODE:-3}"
# EVAL_EMBEDDER = the metric's fixed RULER (keep MiniLM across embedder sweeps so runs compare).
EVAL_EMBEDDER="${EVAL_EMBEDDER:-sentence-transformers/all-MiniLM-L6-v2}"

echo "======================================"
echo ">>> EVAL  ${DATASET} / ${METHOD} / ${ITER} / mode ${RUN_MODE}"
echo "======================================"

# Intrinsic structural metrics (no gold KG needed) -> run for ANY dataset.
python soda/evaluate/intrinsic_metrics.py \
    --dataset "${DATASET}" --method "${METHOD}" --iter "${ITER}" \
    || echo "[eval] intrinsic_metrics failed"

case "$DATASET" in
    webnlg|rebel|wiki-nre)
        # OIE stage (single oie_total.json)
        python soda/evaluate/oie_metric.py \
            --dataset "${DATASET}" --method "${METHOD}" --iter "${ITER}" \
            --embedder "${EVAL_EMBEDDER}" || echo "[eval] oie_metric failed"

        # Canonicalization-as-clustering (B-cubed/pairwise/macro/micro) -> all modes
        python soda/evaluate/clustering_metric.py \
            --dataset "${DATASET}" --method "${METHOD}" --iter "${ITER}" \
            || echo "[eval] clustering_metric failed"

        # Table-1 strict/exact/partial (+ literal-normalized + gap) -> ALL modes.
        # On Mode-1 the discovered relation names drift from gold, so strict is
        # name-drift-penalized (read alongside mode1_metric aligned-F1 / B-cubed);
        # it is still needed for the Mode3-Mode1 strict gap (added 2026-07-02 —
        # M1 dirs used to miss eval/table1_* and needed a separate manual pass).
        python soda/evaluate/run_full_evaluation.py \
            --dataset "${DATASET}" --method "${METHOD}" --iter "${ITER}" \
            || echo "[eval] run_full_evaluation failed"

        if [ "$RUN_MODE" = 1 ]; then
            python soda/evaluate/mode1_metric.py \
                --dataset "${DATASET}" --method "${METHOD}" --iter "${ITER}" \
                --embedder "${EVAL_EMBEDDER}" || echo "[eval] mode1_metric failed"
            python soda/evaluate/redundancy_metric.py \
                --dataset "${DATASET}" --method "${METHOD}" --iter "${ITER}" \
                --embedder "${EVAL_EMBEDDER}" || echo "[eval] redundancy_metric failed"
        else
            python soda/evaluate/retrieval_recall_metric.py \
                --dataset "${DATASET}" --method "${METHOD}" --iter "${ITER}" \
                || echo "[eval] retrieval_recall_metric failed"
            python soda/evaluate/error_attribution_metric.py \
                --dataset "${DATASET}" --method "${METHOD}" --iter "${ITER}" \
                || echo "[eval] error_attribution_metric failed"
        fi
        ;;
    *)
        echo ">>> No gold KG for ${DATASET} -> intrinsic only (GraphRAG/QA evaluated elsewhere)."
        ;;
esac
echo ">>> EVAL done: ${DATASET}/${METHOD}/${ITER}"
