"""Sensitivity of Mode-1 aligned F1 to the alignment threshold tau_a (review L2, minor #7).

Aligned F1 is the headline Mode-1 metric: a discovered relation is mapped to its nearest gold
relation when cosine >= tau_a, and only then can a triple match. The reviewer's point is fair --
the metric therefore has a free parameter, and a signal RANKING that flips with it would not be a
finding about signals. This sweeps tau_a and reports (a) whether the ranking of the nine signals is
stable and (b) how far the absolute numbers move.

Reads canon_kg_*.txt only, so it needs no re-run -- but it DOES need the embedder, which means
sentence-transformers and torch. Per the project rule, run it on a box, not on the Windows laptop.

Server command (CPU is fine; throttle so it cannot starve a GPU job sharing the machine):

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nice -n 19 \
    python soda/evaluate/tau_sweep_aligned_f1.py \
        --dataset webnlg \
        --pred_dir output/webnlg_selfcanon2_mode1_item_qwen3-8b_bgem3_A100_0627open/iter0 \
        --out_dir output/_tau_sweep/webnlg_mode1

Repeat with --dataset rebel / wiki-nre against their Mode-1 dirs.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from soda.evaluate.retrieval_recall_metric import GT_CONFIG, load_kg_txt

# mode1_metric imports sentence-transformers at module level, so it is pulled in inside main():
# that keeps the ranking helper below importable (and unit-testable) on a machine without torch.

# ★ The swept range MUST bracket the value the paper actually reports. `mode1_metric.py` takes
# --rel_threshold default 0.5 and no run script overrides it, so every reported aligned-F1 is at
# tau_a=0.5. The original range started at 0.70 and so excluded the operating point entirely,
# which is the one value a sensitivity analysis cannot omit.
DEFAULT_TAUS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
OPERATING_TAU = 0.50
# ★ MUST be the metric's fixed RULER, not the pipeline's SC embedder.
# `run_eval.sh` pins EVAL_EMBEDDER=MiniLM precisely so runs stay comparable across the
# embedder sweep, and `mode1_metric.py` / `threshold_sweep.py` both use it. This file defaulted
# to bge-m3 until 2026-08-10, which made the sweep vary the ruler AND tau_a at once: measured at
# the same tau_a=0.70 it moved alignment coverage by +0.11..+0.28 and wiki-nre F1 by up to +0.087
# against the reported numbers. A sensitivity analysis of a metric has to hold the metric fixed.
DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def kendall_tau(order_a, order_b):
    """Kendall's tau-b between two rankings given as lists of items, best first."""
    items = [x for x in order_a if x in order_b]
    rank_a = {x: i for i, x in enumerate(order_a)}
    rank_b = {x: i for i, x in enumerate(order_b)}
    conc = disc = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            x, y = items[i], items[j]
            s = (rank_a[x] - rank_a[y]) * (rank_b[x] - rank_b[y])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    n = conc + disc
    return round((conc - disc) / n, 4) if n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="webnlg|rebel|wiki-nre")
    ap.add_argument("--pred_dir", required=True, help="Mode-1 iter dir holding canon_kg_*.txt")
    ap.add_argument("--gt_kg", help="explicit gold KG .txt (overrides --dataset)")
    ap.add_argument("--out_dir", default="output/_tau_sweep")
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    ap.add_argument("--taus", default=",".join(str(t) for t in DEFAULT_TAUS))
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer   # imported late: heavy, and absent locally
    from soda.evaluate.mode1_metric import CASES, evaluate_case

    taus = [float(t) for t in args.taus.split(",")]
    gt_kg = args.gt_kg or GT_CONFIG[args.dataset]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"dataset  : {args.dataset}\npred_dir : {args.pred_dir}\ngold     : {gt_kg}")
    print(f"taus     : {taus}\nembedder : {args.embedder}")
    gold_data = load_kg_txt(gt_kg)
    model = SentenceTransformer(args.embedder)

    # case -> tau -> metrics. The embedder is loaded once and reused across thresholds, so the sweep
    # costs barely more than a single evaluation.
    results = {}
    for case in CASES:
        kg_path = os.path.join(args.pred_dir, f"canon_kg_{case}.txt")
        if not os.path.exists(kg_path):
            print(f"  skip {case} (missing)")
            continue
        pred_data = load_kg_txt(kg_path)
        results[case] = {}
        for tau in taus:
            m = evaluate_case(pred_data, gold_data, model, tau)
            results[case][tau] = m
            print(f"  {case:26s} tau={tau:.2f}  F1={m['aligned_triple_f1']:.4f} "
                  f"cov={m['alignment_coverage']:.2f}")

    rows = []
    for case, per_tau in results.items():
        for tau, m in per_tau.items():
            rows.append({"case": case, "tau_a": tau,
                         "aligned_f1": m["aligned_triple_f1"],
                         "aligned_precision": m["aligned_triple_precision"],
                         "aligned_recall": m["aligned_triple_recall"],
                         "alignment_coverage": m["alignment_coverage"],
                         "n_pred_relations": m["n_pred_relations"]})
    csv_path = os.path.join(args.out_dir, f"tau_sweep_{args.dataset}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Stability: rank the signals by aligned F1 at each tau, then compare every tau to the default.
    orders = {tau: [c for c, _ in sorted(((c, results[c][tau]["aligned_triple_f1"])
                                          for c in results), key=lambda kv: -kv[1])]
              for tau in taus}
    # Compare every tau against the value actually reported, not against the middle of the range:
    # "how far does the ranking move from what we published" is the question minor #7 asks.
    ref = OPERATING_TAU if OPERATING_TAU in taus else taus[len(taus) // 2]
    stability = {"reference_tau": ref, "winner_per_tau": {t: orders[t][0] for t in taus},
                 "kendall_tau_vs_reference": {t: kendall_tau(orders[ref], orders[t]) for t in taus},
                 "f1_range_per_case": {c: round(max(results[c][t]["aligned_triple_f1"] for t in taus)
                                                - min(results[c][t]["aligned_triple_f1"] for t in taus), 4)
                                       for c in results}}
    json_path = os.path.join(args.out_dir, f"tau_sweep_{args.dataset}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "stability": stability}, f, indent=2)

    print("\n--- stability ---")
    print(f"winner per tau        : {stability['winner_per_tau']}")
    print(f"Kendall tau vs {ref}  : {stability['kendall_tau_vs_reference']}")
    print(f"max F1 swing per case : {max(stability['f1_range_per_case'].values()):.4f}")
    print(f"\nSaved -> {csv_path}\n         {json_path}")


if __name__ == "__main__":
    main()
