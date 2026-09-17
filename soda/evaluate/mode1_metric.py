"""
Mode-1 (self-canonicalization) evaluation — name-drift robust.

WHY (see CLAUDE.md Research Design / RQ2):
In Mode 1 the schema is empty, so the model INVENTS relation names. The strict/
exact/partial metric (run_full_evaluation.py) then unfairly penalises a relation
that is semantically correct but lexically different from gold
(e.g. predicted `place_of_birth` vs gold `birthPlace`). This module fixes that
with *alignment-before-scoring*:

  1. Collect every distinct predicted relation.
  2. Embed predicted-relation and gold-relation NAMES (camelCase / underscores
     split into words first) with SBERT; map each predicted relation to its
     nearest gold relation if cosine >= --rel_threshold, else leave unaligned.
  3. Re-score triples as SETS: a predicted (h, r_pred, t) matches a gold
     (h, r_gold, t) iff normalised h==h, t==t, and aligned(r_pred)==r_gold.
     Report micro precision / recall / F1.

Auxiliary (compactness, NOT correctness — gameable, report alongside):
  - n_pred_relations, alignment_coverage (% distinct pred rels mapped to gold),
  - relation redundancy (reuses compute_relation_redundancy_semantic if available).

This is the OpenIE-canonicalization-style evaluation (judge the clustering, not
the surface string). Use it for Mode 1; keep strict/exact/partial for Mode 2/3.

Run (quick mode):
    python soda/evaluate/mode1_metric.py --dataset webnlg \
        --method <method> --iter iter0
Run (manual):
    python soda/evaluate/mode1_metric.py \
        --gt_kg ./soda/evaluate/references/webnlg.txt \
        --pred_dir ./output/<...>/iter0 --output_dir ./output/<...>/iter0/eval_mode1
"""

import os
import re
import ast
import csv
import json
import argparse

import numpy as np
from sentence_transformers import SentenceTransformer, util

CASES = [
    "case1_embed_threshold",
    "case2_name_only",
    "case3_name_gendef_edc",
    "case4_name_gendef_abstract",
    "case5_name_detail",
    "case6_name_detail_headtail",
    "case7_detail_typed",
    "case8_concat",
    "case9_weighted",
]

GT_CONFIG = {
    "webnlg":   ("./soda/evaluate/references/webnlg.txt",   "./schemas/webnlg_schema.csv"),
    "rebel":    ("./soda/evaluate/references/rebel.txt",    "./schemas/rebel_schema.csv"),
    "wiki-nre": ("./soda/evaluate/references/wiki-nre.txt", "./schemas/wiki-nre_schema.csv"),
}


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def norm_entity(x):
    """Normalise an entity surface form for set comparison."""
    if not isinstance(x, str):
        x = str(x)
    x = x.strip().strip('"').strip("'")
    x = x.replace(",_", "_").replace(",", "_")
    x = re.sub(r"\s+", "_", x)
    x = re.sub(r"[^a-zA-Z0-9_]", "", x)
    x = re.sub(r"_+", "_", x).strip("_").lower()
    return x


def rel_to_words(rel):
    """Turn a relation name into natural words so SBERT can match across forms.
    `place_of_birth` / `birthPlace` -> 'place of birth' / 'birth place'."""
    s = str(rel)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)   # camelCase
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_kg_txt(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                data.append([])
                continue
            try:
                data.append(ast.literal_eval(line))
            except Exception:
                data.append([])
    return data


def gold_relation_set(gold_data):
    rels = set()
    for sample in gold_data:
        for t in sample:
            if len(t) >= 3:
                rels.add(t[1])
    return sorted(rels)


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------
def align_relations(pred_rels, gold_rels, model, threshold):
    """Map each predicted relation -> nearest gold relation (cosine >= threshold).
    Many pred rels may map to the same gold rel (that is the point)."""
    if not pred_rels or not gold_rels:
        return {}, {}

    pred_emb = model.encode([rel_to_words(r) for r in pred_rels],
                            convert_to_tensor=True, normalize_embeddings=True)
    gold_emb = model.encode([rel_to_words(r) for r in gold_rels],
                            convert_to_tensor=True, normalize_embeddings=True)

    sims = util.cos_sim(pred_emb, gold_emb).cpu().numpy()   # [P, G]

    mapping, scores = {}, {}
    for i, pr in enumerate(pred_rels):
        j = int(np.argmax(sims[i]))
        best = float(sims[i][j])
        scores[pr] = best
        mapping[pr] = gold_rels[j] if best >= threshold else None
    return mapping, scores


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def triple_set(sample, rel_map=None):
    out = set()
    for t in sample:
        if len(t) < 3:
            continue
        h, r, tail = t[0], t[1], t[2]
        if rel_map is not None:
            r = rel_map.get(r, None)
            if r is None:
                continue   # unaligned predicted relation -> cannot match gold
        out.add((norm_entity(h), r, norm_entity(tail)))
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate_case(pred_data, gold_data, model, threshold):
    # distinct predicted relations across the whole case
    pred_rels = sorted({t[1] for s in pred_data for t in s if len(t) >= 3})
    gold_rels = gold_relation_set(gold_data)

    rel_map, _ = align_relations(pred_rels, gold_rels, model, threshold)

    tp = fp = fn = 0
    for i, gold_sample in enumerate(gold_data):
        pred_sample = pred_data[i] if i < len(pred_data) else []
        pset = triple_set(pred_sample, rel_map=rel_map)
        gset = triple_set(gold_sample, rel_map=None)
        tp += len(pset & gset)
        fp += len(pset - gset)
        fn += len(gset - pset)

    p, r, f = prf(tp, fp, fn)

    n_pred = len(pred_rels)
    aligned = sum(1 for v in rel_map.values() if v is not None)
    coverage = aligned / n_pred if n_pred else 0.0

    return {
        "aligned_triple_precision": round(p, 4),
        "aligned_triple_recall": round(r, 4),
        "aligned_triple_f1": round(f, 4),
        "n_pred_relations": n_pred,
        "n_gold_relations": len(gold_rels),
        "alignment_coverage": round(coverage, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def run(gt_kg, pred_dir, output_dir, rel_threshold, embedder):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 50)
    print(">>> MODE-1 (name-drift robust) EVALUATION")
    print(f"GT KG     : {gt_kg}")
    print(f"PRED DIR  : {pred_dir}")
    print(f"threshold : {rel_threshold}")
    print("=" * 50)

    gold_data = load_kg_txt(gt_kg)
    print(f"Loading embedder: {embedder}")
    model = SentenceTransformer(embedder)

    rows = []
    for case in CASES:
        kg_path = os.path.join(pred_dir, f"canon_kg_{case}.txt")
        if not os.path.exists(kg_path):
            print(f"  skip {case} (missing {kg_path})")
            continue
        pred_data = load_kg_txt(kg_path)
        m = evaluate_case(pred_data, gold_data, model, rel_threshold)
        m["case"] = case
        rows.append(m)
        print(f"  {case:26s} F1={m['aligned_triple_f1']:.4f} "
              f"cov={m['alignment_coverage']:.2f} npred={m['n_pred_relations']}")

    out_json = os.path.join(output_dir, "mode1_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # csv
    if rows:
        keys = ["case", "aligned_triple_precision", "aligned_triple_recall",
                "aligned_triple_f1", "alignment_coverage", "n_pred_relations",
                "n_gold_relations", "tp", "fp", "fn"]
        with open(os.path.join(output_dir, "mode1_metrics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})

    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--method")
    ap.add_argument("--iter")
    ap.add_argument("--gt_kg")
    ap.add_argument("--pred_dir")
    ap.add_argument("--output_dir")
    ap.add_argument("--rel_threshold", type=float, default=0.5)
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    if args.dataset and args.method and args.iter:
        # convention fallback: any dataset with references/{name}.txt works (scale study)
        _cfg = GT_CONFIG.get(args.dataset)
        gt_kg = args.gt_kg or (_cfg[0] if _cfg else f"./soda/evaluate/references/{args.dataset}.txt")
        pred_dir = args.pred_dir or f"./output/{args.dataset}_{args.method}/{args.iter}"
        output_dir = args.output_dir or os.path.join(pred_dir, "eval_mode1")
    else:
        gt_kg, pred_dir, output_dir = args.gt_kg, args.pred_dir, args.output_dir

    run(gt_kg, pred_dir, output_dir, args.rel_threshold, args.embedder)
