"""
Threshold sensitivity sweep (integrity guard, RQ2).

WHY (CLAUDE.md integrity guards): the soft-match thresholds in the eval suite are
heuristic. Before any headline number we must show the CONCLUSIONS are not artifacts
of a particular threshold. This sweeps the eval-only thresholds and reports the metric
vs threshold curve (plot it / eyeball the plateau):

  - OIE entity-F1   vs --ent_range     (oie_metric `ent_threshold`)
  - OIE triple-F1   vs --triple_range  (oie_metric `triple_threshold`)
  - Mode-1 aligned-triple-F1 + alignment coverage vs --rel_range (mode1_metric
    `rel_threshold`), per SC case  [meaningful for Mode-1 self-canon runs]

Embeds ONCE (batched) then re-applies each threshold, so the whole sweep costs ~one
metric run. Needs an embedder -> run on the server (one-off, not auto-wired).

NOT covered: `case0_sim_threshold` changes the PIPELINE decision (which relations get
merged), so it cannot be swept post-hoc — it needs a re-run per value
(`--case0_sim_threshold` on runa.py). Note that separately.

Run (quick mode):
    python soda/evaluate/threshold_sweep.py --dataset webnlg \
        --pred_dir ./output/<run>/iter0
"""

import os
import csv
import json
import argparse

import numpy as np
from sentence_transformers import SentenceTransformer, util

# reuse the exact helpers the real metrics use (run as script -> sibling import)
try:
    from soda.evaluate.oie_metric import (
        norm, load_kg_txt, load_json, entities_of, triple_str, prf,
    )
    from soda.evaluate.mode1_metric import (
        norm_entity, rel_to_words, gold_relation_set, triple_set,
    )
    from soda.evaluate.retrieval_recall_metric import CASES, GT_CONFIG
except ImportError:
    from oie_metric import norm, load_kg_txt, load_json, entities_of, triple_str, prf
    from mode1_metric import norm_entity, rel_to_words, gold_relation_set, triple_set
    from retrieval_recall_metric import CASES, GT_CONFIG


def frange(a, b, step):
    out, x = [], a
    while x <= b + 1e-9:
        out.append(round(x, 4))
        x += step
    return out


# ---------------------------------------------------------------------------
# encode-once helpers
# ---------------------------------------------------------------------------
def _encode_grouped(groups, model):
    """groups = list[list[str]]. Encode all texts in ONE batch, return list of
    per-group embedding matrices (np), preserving order."""
    flat, offsets = [], []
    for g in groups:
        offsets.append((len(flat), len(flat) + len(g)))
        flat.extend(g)
    if not flat:
        return [np.zeros((0, 1)) for _ in groups]
    emb = model.encode(flat, convert_to_numpy=True, normalize_embeddings=True,
                        batch_size=256, show_progress_bar=False)
    return [emb[a:b] for (a, b) in offsets]


def _greedy_count(sims, threshold):
    """#greedy 1-1 matches with cosine >= threshold (same logic as oie_metric)."""
    if sims.size == 0:
        return 0
    pairs = [(sims[i, j], i, j) for i in range(sims.shape[0]) for j in range(sims.shape[1])]
    pairs.sort(reverse=True)
    up, ug, m = set(), set(), 0
    for s, i, j in pairs:
        if s < threshold:
            break
        if i in up or j in ug:
            continue
        up.add(i); ug.add(j); m += 1
    return m


# ---------------------------------------------------------------------------
# OIE sweep (entity-F1 vs ent_thr ; triple-F1 vs triple_thr) — shared, one
# ---------------------------------------------------------------------------
def sweep_oie(gold_data, pred_data, model, ent_range, triple_range):
    n = min(len(gold_data), len(pred_data))
    g_ent_groups, p_ent_groups, g_tri_groups, p_tri_groups = [], [], [], []
    for i in range(n):
        gs, ps = gold_data[i] or [], pred_data[i] or []
        g_ent_groups.append(list({norm(e) for e in entities_of(gs)}))
        p_ent_groups.append(list({norm(e) for e in entities_of(ps)}))
        g_tri_groups.append([triple_str(t) for t in gs if len(t) >= 3])
        p_tri_groups.append([triple_str(t) for t in ps if len(t) >= 3])

    ge = _encode_grouped(g_ent_groups, model); pe = _encode_grouped(p_ent_groups, model)
    gt = _encode_grouped(g_tri_groups, model); pt = _encode_grouped(p_tri_groups, model)

    def sims(p, g):
        return util.cos_sim(p, g).cpu().numpy() if (len(p) and len(g)) else np.zeros((len(p), len(g)))

    ent_sims = [sims(pe[i], ge[i]) for i in range(n)]
    tri_sims = [sims(pt[i], gt[i]) for i in range(n)]
    ent_np = sum(len(x) for x in p_ent_groups); ent_ng = sum(len(x) for x in g_ent_groups)
    tri_np = sum(len(x) for x in p_tri_groups); tri_ng = sum(len(x) for x in g_tri_groups)

    def sweep(sims_list, n_pred, n_gold, rng):
        rows = []
        for thr in rng:
            tp = sum(_greedy_count(s, thr) for s in sims_list)
            p, r, f = prf(tp, n_pred - tp, n_gold - tp)
            rows.append({"threshold": thr, "precision": round(p, 4),
                         "recall": round(r, 4), "f1": round(f, 4),
                         "tp": tp, "n_pred": n_pred, "n_gold": n_gold})
        return rows

    return (sweep(ent_sims, ent_np, ent_ng, ent_range),
            sweep(tri_sims, tri_np, tri_ng, triple_range))


# ---------------------------------------------------------------------------
# Mode-1 rel_threshold sweep (per case) — alignment sensitivity
# ---------------------------------------------------------------------------
def sweep_mode1(gold_data, pred_dir, cases, model, rel_range):
    gold_rels = gold_relation_set(gold_data)
    if not gold_rels:
        return []
    g_emb = model.encode([rel_to_words(r) for r in gold_rels],
                          convert_to_numpy=True, normalize_embeddings=True)
    rows = []
    for case in cases:
        kg_path = os.path.join(pred_dir, f"canon_kg_{case}.txt")
        if not os.path.isfile(kg_path):
            continue
        pred_data = load_kg_txt(kg_path)
        pred_rels = sorted({t[1] for s in pred_data for t in s if len(t) >= 3})
        if not pred_rels:
            continue
        p_emb = model.encode([rel_to_words(r) for r in pred_rels],
                             convert_to_numpy=True, normalize_embeddings=True)
        sims = util.cos_sim(p_emb, g_emb).cpu().numpy()   # [P, G]
        best_j = sims.argmax(axis=1)
        best_s = sims.max(axis=1)
        for thr in rel_range:
            rel_map = {pr: (gold_rels[best_j[k]] if best_s[k] >= thr else None)
                       for k, pr in enumerate(pred_rels)}
            tp = fp = fn = 0
            for i, gs in enumerate(gold_data):
                ps = pred_data[i] if i < len(pred_data) else []
                pset = triple_set(ps, rel_map=rel_map)
                gset = triple_set(gs, rel_map=None)
                tp += len(pset & gset); fp += len(pset - gset); fn += len(gset - pset)
            p, r, f = prf(tp, fp, fn)
            aligned = sum(1 for v in rel_map.values() if v is not None)
            rows.append({"case": case, "rel_threshold": thr,
                         "aligned_triple_f1": round(f, 4),
                         "precision": round(p, 4), "recall": round(r, 4),
                         "n_pred_relations": len(pred_rels), "n_aligned": aligned,
                         "alignment_coverage": round(aligned / len(pred_rels), 4)})
    return rows


def _write(rows, path, keys):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run(gt_kg, pred_dir, output_dir, embedder, ent_range, triple_range, rel_range, cases):
    os.makedirs(output_dir, exist_ok=True)
    gold_data = load_kg_txt(gt_kg)
    print(f"GT KG     : {gt_kg}  ({len(gold_data)} records)")
    print(f"PRED DIR  : {pred_dir}")
    print(f"Embedder  : {embedder}\n")
    model = SentenceTransformer(embedder)

    oie_json = os.path.join(pred_dir, "oie_total.json")
    if os.path.isfile(oie_json):
        ent_rows, tri_rows = sweep_oie(gold_data, load_json(oie_json), model,
                                       ent_range, triple_range)
        _write(ent_rows, os.path.join(output_dir, "sweep_oie_entity.csv"),
               ["threshold", "precision", "recall", "f1", "tp", "n_pred", "n_gold"])
        _write(tri_rows, os.path.join(output_dir, "sweep_oie_triple.csv"),
               ["threshold", "precision", "recall", "f1", "tp", "n_pred", "n_gold"])
        print("OIE entity-F1 vs ent_threshold:")
        for r in ent_rows:
            print(f"   thr={r['threshold']:<5} F1={r['f1']:<7} P={r['precision']:<7} R={r['recall']}")
        print("OIE triple-soft-F1 vs triple_threshold:")
        for r in tri_rows:
            print(f"   thr={r['threshold']:<5} F1={r['f1']:<7} P={r['precision']:<7} R={r['recall']}")
    else:
        print(f"[sweep] no oie_total.json in {pred_dir} -> skip OIE sweep")

    m1 = sweep_mode1(gold_data, pred_dir, cases, model, rel_range)
    if m1:
        _write(m1, os.path.join(output_dir, "sweep_mode1_rel.csv"),
               ["case", "rel_threshold", "aligned_triple_f1", "precision", "recall",
                "n_pred_relations", "n_aligned", "alignment_coverage"])
        print("\nMode-1 aligned-triple-F1 vs rel_threshold (per case) -> sweep_mode1_rel.csv")
        # compact: show one case's curve as a sanity print
        seen = set()
        for r in m1:
            if r["case"] not in seen and len(seen) < 2:
                seen.add(r["case"])
        for r in m1:
            if r["case"] in seen:
                print(f"   {r['case']:24s} thr={r['rel_threshold']:<5} F1={r['aligned_triple_f1']:<7} "
                      f"cov={r['alignment_coverage']:<6} #aligned={r['n_aligned']}")
    else:
        print("[sweep] no canon_kg_*.txt -> skip Mode-1 rel sweep")

    print(f"\nSaved sweeps -> {output_dir}/sweep_*.csv")
    print("NOTE: case0_sim_threshold is a PIPELINE decision (not eval-only); sweep it by "
          "re-running runa.py with --case0_sim_threshold <v> (cannot be done post-hoc).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--method")
    ap.add_argument("--iter")
    ap.add_argument("--gt_kg")
    ap.add_argument("--pred_dir")
    ap.add_argument("--output_dir")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--ent_range", default="0.5,0.9,0.05", help="start,stop,step")
    ap.add_argument("--triple_range", default="0.4,0.8,0.05")
    ap.add_argument("--rel_range", default="0.3,0.7,0.05")
    ap.add_argument("--cases")
    args = ap.parse_args()

    gt_kg = args.gt_kg or (GT_CONFIG.get(args.dataset) if args.dataset else None)
    if not gt_kg or not os.path.isfile(gt_kg):
        raise SystemExit(f"[sweep] gold KG not found: {gt_kg} (pass --gt_kg or a valid --dataset)")
    pred_dir = args.pred_dir
    if not pred_dir and args.dataset and args.method and args.iter:
        pred_dir = f"./output/{args.dataset}_{args.method}/{args.iter}"
    if not pred_dir:
        raise SystemExit("[sweep] need --pred_dir (or --dataset/--method/--iter)")
    output_dir = args.output_dir or os.path.join(pred_dir, "eval_sweep")
    rng = lambda s: frange(*[float(x) for x in s.split(",")])
    cases = [c.strip() for c in args.cases.split(",")] if args.cases else CASES

    run(gt_kg, pred_dir, output_dir, args.embedder,
        rng(args.ent_range), rng(args.triple_range), rng(args.rel_range), cases)


if __name__ == "__main__":
    main()
