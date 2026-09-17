"""
Error attribution funnel — where does a gold triple get lost? (RQ1b/RQ2; Mode 2/3)

WHY (user question, 2026-06-18): in Mode 3 (frozen seed schema) a wrong/missing
final triple can fail at THREE different stages, and the remedy differs:
  - OIE stage      -> the extractor never produced the right (head, tail)
                      => fix = a stronger OIE LLM (bigger Qwen), better prompt.
  - RETRIEVAL stage-> OIE got (head, tail) but the schema retriever did not put the
                      gold relation in the top-k candidates => fix = better embedder / larger k.
  - CANON stage    -> gold relation WAS in the candidates but the LLM MCQ picked
                      another (or "new") => fix = better signal (SC case) / verify.

This decomposes every GOLD triple into exactly one bucket, so the percentages sum to
100% and you can read off "is the bottleneck OIE or retrieval?".

  For each gold triple (h_g, r_g, t_g) in record i:
    1. OIE recovered?  exists an OIE triple in record i with (norm h == norm h_g and
       norm t == norm t_g)?            no -> OIE_MISS
    2. retrieval hit?  norm(r_g) in that triple's top-k candidate names?
                                        no -> RETRIEVAL_MISS
    3. canon correct?  the final chosen relation == r_g (norm)?
                                        no -> CANON_MISS    yes -> CORRECT

All POST-HOC from existing outputs (gold KG + sc_mcp_logs) — no model, no pipeline
change, no SC re-run. The SC trace already stores the final canonical relation
(`pred_relation` in the JSONL; the `[FINAL]`-marked candidate in the old dir logs).

FAIR FOR Mode 2 / Mode 3 (candidate pool = seed schema, so gold relation names are
comparable). Mode 1 self-invents names -> use clustering_metric.py / mode1_metric.py.

CAVEAT — entity match is string-normalised (exact). Surface-form differences
("Karlín" vs "Prague-Karlín") get charged to OIE_MISS, so OIE_MISS is an UPPER bound
on true OIE loss; we also report a relaxed (token-overlap) OIE_MISS as a lower-ish
bound. The true (soft) OIE recall is in eval_oie/ (oie_metric.py).

Run (quick mode):
    python soda/evaluate/error_attribution_metric.py --dataset rebel \
        --method <method> --iter iter0
"""

import os
import csv
import json
import argparse

try:   # package import (when run as a module)
    from soda.evaluate.retrieval_recall_metric import (
        CASES, GT_CONFIG, norm_entity, norm_rel, load_kg_txt, load_case_triplets,
    )
except ImportError:   # script run: `python soda/evaluate/error_attribution_metric.py`
    from retrieval_recall_metric import (
        CASES, GT_CONFIG, norm_entity, norm_rel, load_kg_txt, load_case_triplets,
    )


def _tokens(norm_str):
    return set(t for t in norm_str.split("_") if t)


def build_gold_triples(gold_data):
    """record_idx -> list of (norm_h, norm_t, norm_r)."""
    out = []
    for sample in gold_data:
        rows = []
        for t in sample:
            if len(t) >= 3:
                rows.append((norm_entity(t[0]), norm_entity(t[2]), norm_rel(t[1])))
        out.append(rows)
    return out


def build_oie_index(pred_dir, case):
    """record_idx -> dict (norm_h, norm_t) -> {"cands": set(norm_rel), "finals": set(norm_rel)}.
    Aggregates over all OIE triples sharing a (h,t) in the record."""
    rows, src = load_case_triplets(pred_dir, case)
    by_idx = {}
    for idx, trip, topk, final_pred in rows:
        if idx is None:
            continue
        key = (norm_entity(trip[0]), norm_entity(trip[2]))
        d = by_idx.setdefault(idx, {})
        e = d.setdefault(key, {"cands": set(), "finals": set()})
        e["cands"].update(norm_rel(c) for c in topk)
        if final_pred:
            e["finals"].add(norm_rel(final_pred))
    return by_idx, src


def _relaxed_recovered(gold_key, rec_index):
    """True if some OIE (h,t) in the record token-overlaps the gold (h,t) (Jaccard>=0.5)."""
    gh, gt = _tokens(gold_key[0]), _tokens(gold_key[1])
    for (oh, ot) in rec_index.keys():
        oh_t, ot_t = _tokens(oh), _tokens(ot)
        def jac(a, b):
            return len(a & b) / len(a | b) if (a or b) else 0.0
        if jac(gh, oh_t) >= 0.5 and jac(gt, ot_t) >= 0.5:
            return True
    return False


def evaluate_case(pred_dir, case, gold_triples):
    oie_index, src = build_oie_index(pred_dir, case)
    n_gold = 0
    oie_miss = retr_miss = canon_miss = correct = 0
    oie_miss_relaxed = 0

    for idx, golds in enumerate(gold_triples):
        rec_index = oie_index.get(idx, {})
        for (gh, gt, gr) in golds:
            n_gold += 1
            key = (gh, gt)
            entry = rec_index.get(key)
            if entry is None:
                oie_miss += 1
                if not _relaxed_recovered(key, rec_index):
                    oie_miss_relaxed += 1
                continue
            if gr not in entry["cands"]:
                retr_miss += 1
                continue
            if gr in entry["finals"]:
                correct += 1
            else:
                canon_miss += 1

    recovered = n_gold - oie_miss
    retrieved = recovered - retr_miss
    pct = lambda x: round(100.0 * x / n_gold, 2) if n_gold else None
    cond = lambda x, d: round(100.0 * x / d, 2) if d else None
    return {
        "case": case,
        "source": src,
        "n_gold": n_gold,
        # absolute buckets (sum == n_gold)
        "OIE_MISS": oie_miss,
        "RETRIEVAL_MISS": retr_miss,
        "CANON_MISS": canon_miss,
        "CORRECT": correct,
        # as % of all gold triples
        "OIE_MISS_pct": pct(oie_miss),
        "RETRIEVAL_MISS_pct": pct(retr_miss),
        "CANON_MISS_pct": pct(canon_miss),
        "CORRECT_pct": pct(correct),
        # conditional rates (the attribution read)
        "recovered_pct": pct(recovered),
        "retr_given_recovered_pct": cond(retrieved, recovered),
        "canon_given_retrieved_pct": cond(correct, retrieved),
        # relaxed entity-match lower bound on OIE loss
        "OIE_MISS_relaxed_pct": pct(oie_miss_relaxed),
    }


def run(gt_kg, pred_dir, output_dir, cases):
    os.makedirs(output_dir, exist_ok=True)
    gold_triples = build_gold_triples(load_kg_txt(gt_kg))
    n_gold_total = sum(len(g) for g in gold_triples)
    print(f"GT KG     : {gt_kg}  ({len(gold_triples)} records, {n_gold_total} gold triples)")
    print(f"PRED DIR  : {pred_dir}\n")
    print(f"{'case':28s} {'src':5s} {'OIE_miss':>9s} {'RETR_miss':>10s} "
          f"{'CANON_miss':>11s} {'CORRECT':>8s}   (% of gold)")

    rows = []
    for case in cases:
        r = evaluate_case(pred_dir, case, gold_triples)
        rows.append(r)
        if r["n_gold"]:
            print(f"{case:28s} {r['source']:5s} {r['OIE_MISS_pct']:>8}% {r['RETRIEVAL_MISS_pct']:>9}% "
                  f"{r['CANON_MISS_pct']:>10}% {r['CORRECT_pct']:>7}%   "
                  f"[recov={r['recovered_pct']}% | retr|recov={r['retr_given_recovered_pct']}% "
                  f"| canon|retr={r['canon_given_retrieved_pct']}%]")
        else:
            print(f"{case:28s} {r['source']:5s}  no SC trace / empty")

    with open(os.path.join(output_dir, "error_attribution.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    keys = ["case", "source", "n_gold", "OIE_MISS", "RETRIEVAL_MISS", "CANON_MISS", "CORRECT",
            "OIE_MISS_pct", "RETRIEVAL_MISS_pct", "CANON_MISS_pct", "CORRECT_pct",
            "recovered_pct", "retr_given_recovered_pct", "canon_given_retrieved_pct",
            "OIE_MISS_relaxed_pct"]
    with open(os.path.join(output_dir, "error_attribution.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"\nSaved -> {output_dir}/error_attribution.csv (+ .json)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="webnlg|rebel|wiki-nre (quick-mode gold path)")
    ap.add_argument("--method", help="quick mode: builds pred_dir ./output/{dataset}_{method}/{iter}")
    ap.add_argument("--iter", help="quick mode iter dir name, e.g. iter0")
    ap.add_argument("--gt_kg", help="explicit gold KG .txt (overrides --dataset)")
    ap.add_argument("--pred_dir", help="iter dir holding sc_mcp_logs(.jsonl); or use --method/--iter")
    ap.add_argument("--output_dir")
    ap.add_argument("--cases", help="comma list of cases (default: all 8)")
    args = ap.parse_args()

    gt_kg = args.gt_kg or (GT_CONFIG.get(args.dataset)
                           or (f"./soda/evaluate/references/{args.dataset}.txt" if args.dataset else None))
    if not gt_kg or not os.path.isfile(gt_kg):
        raise SystemExit(f"[error_attribution] gold KG not found: {gt_kg} "
                         f"(pass --gt_kg or a valid --dataset)")
    pred_dir = args.pred_dir
    if not pred_dir and args.dataset and args.method and args.iter:
        pred_dir = f"./output/{args.dataset}_{args.method}/{args.iter}"
    if not pred_dir:
        raise SystemExit("[error_attribution] need --pred_dir (or --dataset/--method/--iter)")
    output_dir = args.output_dir or os.path.join(pred_dir, "eval_error_attribution")
    cases = [c.strip() for c in args.cases.split(",")] if args.cases else CASES

    run(gt_kg, pred_dir, output_dir, cases)


if __name__ == "__main__":
    main()
