"""
OIE-stage evaluation (RQ1a) — did extraction pull out enough (recall) and the
right (precision) content from the raw text, BEFORE canonicalization?

Reads the OIE dump `oie_total.json` (per-sample list of raw [s, r, o] triplets,
written by EDC.oie) and compares to the gold KG. Because OIE relation names are
raw (not yet canonicalised), we do NOT score the relation string strictly; we
measure CONTENT extraction with three views:

  1. Entity P/R/F1  — gold entities (subjects+objects) vs predicted entities,
     matched by normalised exact OR embedding cosine >= --ent_threshold.
  2. Triplet count parity — per record, |pred| vs |gold| (are we extracting the
     right NUMBER of triplets?). Reports mean ratio and mean min/max coverage.
     (This is the author's "rho"/count-parity recall idea.)
  3. Triple soft-match P/R/F1 — embed each triple as the string "h r t", greedily
     match pred<->gold within a sample when cosine >= --triple_threshold; counts
     a triple as recovered if its content matches, regardless of exact wording.

Run (quick mode):
    python soda/evaluate/oie_metric.py --dataset webnlg --method <m> --iter iter0
Run (manual):
    python soda/evaluate/oie_metric.py \
        --gt_kg ./soda/evaluate/references/webnlg.txt \
        --oie_json ./output/<...>/iter0/oie_total.json \
        --output_dir ./output/<...>/iter0/eval_oie
"""

import os
import re
import ast
import json
import argparse

import numpy as np
from sentence_transformers import SentenceTransformer, util

GT_CONFIG = {
    "webnlg":   "./soda/evaluate/references/webnlg.txt",
    "rebel":    "./soda/evaluate/references/rebel.txt",
    "wiki-nre": "./soda/evaluate/references/wiki-nre.txt",
}


def norm(x):
    if not isinstance(x, str):
        x = str(x)
    x = x.strip().strip('"').strip("'")
    x = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", x)
    x = x.replace("_", " ").replace("-", " ")
    x = re.sub(r"\s+", " ", x).strip().lower()
    return x


def load_kg_txt(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            try:
                data.append(ast.literal_eval(line) if line else [])
            except Exception:
                data.append([])
    return data


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def greedy_match(pred_texts, gold_texts, model, threshold):
    """Greedy 1-1 matching by cosine >= threshold. Returns #matches."""
    if not pred_texts or not gold_texts:
        return 0
    pe = model.encode(pred_texts, convert_to_tensor=True, normalize_embeddings=True)
    ge = model.encode(gold_texts, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(pe, ge).cpu().numpy()   # [P, G]

    matches = 0
    used_g = set()
    # sort all (i, j) pairs by similarity descending, take greedily
    pairs = [(sims[i, j], i, j)
             for i in range(sims.shape[0]) for j in range(sims.shape[1])]
    pairs.sort(reverse=True)
    used_p = set()
    for s, i, j in pairs:
        if s < threshold:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches += 1
    return matches


def entities_of(sample):
    ents = []
    for t in sample:
        if len(t) >= 3:
            ents.append(t[0])
            ents.append(t[2])
    return ents


def triple_str(t):
    return f"{norm(t[0])} {norm(t[1])} {norm(t[2])}"


def run(gt_kg, oie_json, output_dir, ent_threshold, triple_threshold, embedder):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 50)
    print(">>> OIE-stage EVALUATION")
    print(f"GT KG     : {gt_kg}")
    print(f"OIE JSON  : {oie_json}")
    print(f"ent_thr={ent_threshold}  triple_thr={triple_threshold}")
    print("=" * 50)

    gold_data = load_kg_txt(gt_kg)
    pred_data = load_json(oie_json)
    print(f"Loading embedder: {embedder}")
    model = SentenceTransformer(embedder)

    # ----- entity-level (micro) -----
    e_tp = e_fp = e_fn = 0
    # ----- triple soft-match (micro) -----
    t_tp = t_fp = t_fn = 0
    # ----- count parity -----
    ratios = []
    coverages = []
    # ----- per-item entity coverage (is the entity-miss concentrated?) -----
    ent_recall_per_item = []

    n = min(len(gold_data), len(pred_data))
    for i in range(n):
        gold_sample = gold_data[i] or []
        pred_sample = pred_data[i] or []

        # entities (dedup by normalised form within a sample)
        g_ents = list({norm(e) for e in entities_of(gold_sample)})
        p_ents = list({norm(e) for e in entities_of(pred_sample)})
        em = greedy_match(p_ents, g_ents, model, ent_threshold)
        e_tp += em
        e_fp += max(len(p_ents) - em, 0)
        e_fn += max(len(g_ents) - em, 0)
        if len(g_ents) > 0:
            ent_recall_per_item.append(em / len(g_ents))

        # triples
        g_tr = [triple_str(t) for t in gold_sample if len(t) >= 3]
        p_tr = [triple_str(t) for t in pred_sample if len(t) >= 3]
        tm = greedy_match(p_tr, g_tr, model, triple_threshold)
        t_tp += tm
        t_fp += max(len(p_tr) - tm, 0)
        t_fn += max(len(g_tr) - tm, 0)

        # count parity
        gp, pp = len(g_tr), len(p_tr)
        if gp > 0:
            ratios.append(pp / gp)
            coverages.append(min(pp, gp) / max(pp, gp) if max(pp, gp) else 1.0)

    ep, er, ef = prf(e_tp, e_fp, e_fn)
    tp_, tr_, tf_ = prf(t_tp, t_fp, t_fn)

    result = {
        "n_samples": n,
        "entity_precision": round(ep, 4),
        "entity_recall": round(er, 4),
        "entity_f1": round(ef, 4),
        # raw entity-localization counts (how many gold entities does OIE miss?)
        "entities_gold_total": int(e_tp + e_fn),
        "entities_pred_total": int(e_tp + e_fp),
        "entities_matched": int(e_tp),
        "entities_missed": int(e_fn),          # gold entities not found by OIE
        "entities_spurious": int(e_fp),        # predicted entities not in gold
        # per-item entity coverage (is the miss concentrated in some items?)
        "entity_recall_per_item_median": round(float(np.median(ent_recall_per_item)), 4) if ent_recall_per_item else 0.0,
        "frac_items_entity_recall_below_0.5": round(float(np.mean([1.0 if r < 0.5 else 0.0 for r in ent_recall_per_item])), 4) if ent_recall_per_item else 0.0,
        "frac_items_entity_recall_1.0": round(float(np.mean([1.0 if r >= 0.999 else 0.0 for r in ent_recall_per_item])), 4) if ent_recall_per_item else 0.0,
        "triple_soft_precision": round(tp_, 4),
        "triple_soft_recall": round(tr_, 4),
        "triple_soft_f1": round(tf_, 4),
        "count_ratio_mean(pred/gold)": round(float(np.mean(ratios)), 4) if ratios else 0.0,
        "count_coverage_mean(min/max)": round(float(np.mean(coverages)), 4) if coverages else 0.0,
        "thresholds": {"entity": ent_threshold, "triple": triple_threshold},
    }

    out = os.path.join(output_dir, "oie_metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--method")
    ap.add_argument("--iter")
    ap.add_argument("--gt_kg")
    ap.add_argument("--oie_json")
    ap.add_argument("--output_dir")
    ap.add_argument("--ent_threshold", type=float, default=0.7)
    ap.add_argument("--triple_threshold", type=float, default=0.6)
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    if args.dataset and args.method and args.iter:
        # convention fallback: any dataset with references/{name}.txt works (scale study)
        gt_kg = args.gt_kg or GT_CONFIG.get(args.dataset) \
            or f"./soda/evaluate/references/{args.dataset}.txt"
        pred_dir = f"./output/{args.dataset}_{args.method}/{args.iter}"
        oie_json = args.oie_json or os.path.join(pred_dir, "oie_total.json")
        output_dir = args.output_dir or os.path.join(pred_dir, "eval_oie")
    else:
        gt_kg, oie_json, output_dir = args.gt_kg, args.oie_json, args.output_dir

    run(gt_kg, oie_json, output_dir, args.ent_threshold, args.triple_threshold, args.embedder)
