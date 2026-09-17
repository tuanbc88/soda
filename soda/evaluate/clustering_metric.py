"""
Canonicalization-as-clustering metrics (RQ1b) — isolates the MERGE decision.

RQ1b is fundamentally "did mentions that the gold groups together get grouped
together by the system, regardless of the surface relation name?". The OpenIE-
canonicalization line (Galárraga et al. 2014; Vashishth et al. 2018 / CESI)
scores this with macro / micro / pairwise and B-cubed P/R/F1 over the predicted
relation→canonical grouping vs the gold grouping. This complements the triple
metrics (which also penalise name drift) by scoring ONLY the clustering.

How the mention set is built (relation-only clustering):
- Align predicted triples to gold triples per sample by the normalised (head,
  tail) pair (greedy one-to-one). Each matched pair is a "mention".
- gold cluster label  = gold relation of that mention
- pred cluster label  = predicted (canonical) relation of that mention
Then compute clustering agreement between the two labelings over the mentions.
(Only entity-pair-matched triples are scored, so the relation grouping is what
varies — OIE entity errors are not charged here.)

Reported per case: pairwise / B-cubed / macro / micro P-R-F1, plus #mentions,
#pred_clusters (= distinct predicted relations used), #gold_clusters.

Run (quick): python soda/evaluate/clustering_metric.py --dataset webnlg --method <m> --iter iter0
Run (manual): ... --gt_kg <gold.txt> --pred_dir <iterdir> --output_dir <dir>

No embeddings needed (pure string clustering) — runs without a GPU/model.
"""
import os, re, ast, json, csv, argparse, itertools
from collections import defaultdict

# Optional: information-theoretic / chance-adjusted partition metrics (sklearn).
# These COMPLEMENT B-cubed: ARI/AMI are adjusted-for-chance, V-measure is entropy-based.
# Kept optional so the script still runs (pure-string) without sklearn.
try:
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,
        v_measure_score, homogeneity_score, completeness_score, fowlkes_mallows_score,
    )
    _HAS_SK = True
except Exception:
    _HAS_SK = False

import glob

CASES = ["case1_embed_threshold","case2_name_only","case3_name_gendef_edc","case4_name_gendef_abstract",
         "case5_name_detail","case6_name_detail_headtail","case7_detail_typed","case8_concat","case9_weighted"]
GT = {"webnlg":"./soda/evaluate/references/webnlg.txt",
      "rebel":"./soda/evaluate/references/rebel.txt",
      "wiki-nre":"./soda/evaluate/references/wiki-nre.txt"}
GT_SCHEMA = {"webnlg":"./schemas/webnlg_schema.json",
             "rebel":"./schemas/rebel_schema.json",
             "wiki-nre":"./schemas/wiki-nre_schema.json"}
# KB-grounded gold entity-types (schemas/build_gold_entity_types.py): per-slot status confirms the
# schema low-level type against Wikidata/DBpedia domain/range. Used to also score the entity-canon over
# only the KB-confirmed slots (so the gold is authoritative, not the circular Claude-authored typing).
GOLD_GROUND = {"webnlg":"./schemas/gold_entity_types_webnlg.json",
               "rebel":"./schemas/gold_entity_types_rebel.json",
               "wiki-nre":"./schemas/gold_entity_types_wiki-nre.json"}
CONSISTENT = {"confirmed", "narrower", "broader", "accepted"}  # KB backs the type, or author-adjudicated OK
# Per-ENTITY instance-of (P31) gold (schemas/build_gold_entity_p31.py): each gold SURFACE -> its KB
# instance-of type, mapped to the schema taxonomy. Sharper than the slot-derived gold (a surface's type
# comes from its identity, not the relation role it happens to play). Used for a third entity-canon
# variant (clustering_entity_p31.csv). Optional -> absent file = variant skipped.
GOLD_P31 = {"webnlg":"./schemas/gold_entity_p31_webnlg.json",
            "rebel":"./schemas/gold_entity_p31_rebel.json",
            "wiki-nre":"./schemas/gold_entity_p31_wiki-nre.json"}


def norm(x):
    if not isinstance(x, str): x = str(x)
    x = x.strip().strip('"').strip("'")
    x = x.replace(",_", "_").replace(",", "_")
    x = re.sub(r"\s+", "_", x)
    x = re.sub(r"[^a-zA-Z0-9_]", "", x)
    return re.sub(r"_+", "_", x).strip("_").lower()


def load_kg_txt(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            try: out.append(ast.literal_eval(line) if line else [])
            except Exception: out.append([])
    return out


def build_mentions(pred_data, gold_data):
    """Return list of (pred_rel, gold_rel) for entity-pair-matched triples."""
    mentions = []
    n = min(len(pred_data), len(gold_data))
    for i in range(n):
        gold_by_pair = defaultdict(list)
        for t in (gold_data[i] or []):
            if len(t) >= 3:
                gold_by_pair[(norm(t[0]), norm(t[2]))].append(t[1])
        for t in (pred_data[i] or []):
            if len(t) < 3: continue
            key = (norm(t[0]), norm(t[2]))
            if gold_by_pair.get(key):
                g = gold_by_pair[key].pop(0)
                mentions.append((t[1], g))
    return mentions


def _clusters(labels):
    c = defaultdict(set)
    for idx, lab in enumerate(labels):
        c[lab].add(idx)
    return c


def _f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def cluster_scores(mentions):
    pred = [m[0] for m in mentions]
    gold = [m[1] for m in mentions]
    N = len(mentions)
    P = _clusters(pred); G = _clusters(gold)
    res = {"n_mentions": N, "n_pred_clusters": len(P), "n_gold_clusters": len(G)}
    if N == 0:
        return res

    # ---- pairwise ----
    def pairs(cs): return sum(len(s) * (len(s) - 1) // 2 for s in cs.values())
    same_pred = pairs(P); same_gold = pairs(G)
    correct = 0
    for cp in P.values():
        for cg in G.values():
            o = len(cp & cg)
            if o > 1: correct += o * (o - 1) // 2
    pp = correct / same_pred if same_pred else 0.0
    pr = correct / same_gold if same_gold else 0.0
    res["pairwise"] = {"p": round(pp,4), "r": round(pr,4), "f1": round(_f1(pp,pr),4)}

    # ---- B-cubed ----
    gold_of = gold; pred_of = pred
    Gmap = {lab:s for lab,s in G.items()}; Pmap = {lab:s for lab,s in P.items()}
    bp = br = 0.0
    for i in range(N):
        cp = Pmap[pred_of[i]]; cg = Gmap[gold_of[i]]
        o = len(cp & cg)
        bp += o/len(cp); br += o/len(cg)
    bp /= N; br /= N
    res["bcubed"] = {"p": round(bp,4), "r": round(br,4), "f1": round(_f1(bp,br),4)}

    # ---- macro (avg over clusters of best-overlap purity) ----
    macro_p = sum(max(len(cp & cg) for cg in G.values())/len(cp) for cp in P.values())/len(P)
    macro_r = sum(max(len(cp & cg) for cp in P.values())/len(cg) for cg in G.values())/len(G)
    res["macro"] = {"p": round(macro_p,4), "r": round(macro_r,4), "f1": round(_f1(macro_p,macro_r),4)}

    # ---- micro (overlap mass / N) ----
    micro_p = sum(max(len(cp & cg) for cg in G.values()) for cp in P.values())/N
    micro_r = sum(max(len(cp & cg) for cp in P.values()) for cg in G.values())/N
    res["micro"] = {"p": round(micro_p,4), "r": round(micro_r,4), "f1": round(_f1(micro_p,micro_r),4)}

    # ---- chance-adjusted / information-theoretic partition metrics (sklearn) ----
    # Complement B-cubed: ARI/AMI subtract the agreement expected by chance (so a system
    # that just splits everything no longer scores high), V-measure = harmonic mean of
    # homogeneity (no mixed clusters) + completeness (no split gold classes), FMI = geo-mean
    # of pairwise P/R. All in [0,1] except ARI/AMI which can go slightly negative.
    if _HAS_SK:
        res["ari"] = round(adjusted_rand_score(gold, pred), 4)            # adjusted Rand
        res["nmi"] = round(normalized_mutual_info_score(gold, pred), 4)   # normalized MI
        res["ami"] = round(adjusted_mutual_info_score(gold, pred), 4)     # adjusted MI
        res["v_measure"] = round(v_measure_score(gold, pred), 4)
        res["homogeneity"] = round(homogeneity_score(gold, pred), 4)
        res["completeness"] = round(completeness_score(gold, pred), 4)
        res["fmi"] = round(fowlkes_mallows_score(gold, pred), 4)          # Fowlkes-Mallows
    return res


# ===========================================================================
# ENTITY-TYPE canonicalization clustering (the entity-types half of the schema)
# ===========================================================================
# SODA canonicalizes BOTH relation types and ENTITY types. The entity-canon stage
# assigns every entity a canonical TYPE (`pred_entity_type`, e.g. City->Settlement),
# logged per record in entity_canon_ec{1,2,3}_*.json. We score that assignment the
# SAME clustering way as relations:
#   mention   = one entity occurrence in a gold triple whose surface the entity-canon saw
#   gold label = the entity type the REFERENCE schema implies for that slot
#               (relation's head_type / tail_type)
#   pred label = the canonical pred_entity_type the entity-canon assigned
# Clustering metrics are label-INVARIANT (they compare GROUPINGS, not type names) so a
# self-generated taxonomy (Mode 1) is fine: we only ask "did entities the schema groups
# as one type get one pred type?". One row per entity-canon case (ec1/ec2/ec3).
def load_schema_types(schema_path):
    s = json.load(open(schema_path, encoding="utf-8"))
    rt = s.get("relation_types", s)
    out = {}
    for rel, info in rt.items():
        if isinstance(info, dict):
            out[norm(rel)] = (info.get("head_type"), info.get("tail_type"))
    return out


def load_grounded_status(path):
    """KB-grounded gold (schemas/gold_entity_types_{ds}.json) -> {norm(rel): {head,tail: status}}."""
    if not (path and os.path.exists(path)):
        return None
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for rel, v in d.items():
        if rel.startswith("_") or not isinstance(v, dict):
            continue
        out[norm(rel)] = {"head": v.get("head_status"), "tail": v.get("tail_status")}
    return out


def load_p31_gold(path):
    """Per-entity P31 gold (schemas/gold_entity_p31_{ds}.json) -> {norm(surface): schema_type} for LINKED
    entities only (status 'linked'). Absent file / no linked entities -> None (variant skipped)."""
    if not (path and os.path.exists(path)):
        return None
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for k, v in d.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        if v.get("status") == "linked" and v.get("schema_type"):
            out[k] = v["schema_type"]        # key is already norm()'d by the builder (same norm())
    return out or None


def build_entity_mentions_p31(gold_data, ec_data, p31_gold):
    """Like build_entity_mentions but the gold type of each entity occurrence is its per-entity P31 type
    (from identity), not the relation-slot type. Only occurrences whose surface is P31-linked are scored."""
    mentions = []
    n = min(len(gold_data), len(ec_data))
    for i in range(n):
        surf2type = {}
        for ent in (ec_data[i].get("entities") or []):
            pt = ent.get("pred_entity_type")
            for msurf in (ent.get("members") or []):
                if pt:
                    surf2type[norm(msurf)] = pt
        for t in (gold_data[i] or []):
            if len(t) < 3:
                continue
            for surf in (norm(t[0]), norm(t[2])):
                gtype = p31_gold.get(surf)
                pt = surf2type.get(surf)
                if gtype and pt:
                    mentions.append((pt, gtype))
    return mentions


def build_entity_mentions(gold_data, ec_data, schema_types, ground=None):
    """Return [(pred_type, gold_type, slot_status)] over entity occurrences in gold triples whose
    surface the entity-canon (record-aligned) actually typed. slot_status = the KB-grounding status of
    that relation slot ('na' if no grounded gold) so callers can restrict to KB-confirmed slots."""
    mentions = []
    n = min(len(gold_data), len(ec_data))
    for i in range(n):
        surf2type = {}
        for ent in (ec_data[i].get("entities") or []):
            pt = ent.get("pred_entity_type")
            for msurf in (ent.get("members") or []):
                if pt:
                    surf2type[norm(msurf)] = pt
        for t in (gold_data[i] or []):
            if len(t) < 3:
                continue
            rel = norm(t[1])
            types = schema_types.get(rel)
            if not types:
                continue
            gstat = (ground or {}).get(rel, {})
            for surf, gtype, slot in ((norm(t[0]), types[0], "head"), (norm(t[2]), types[1], "tail")):
                pt = surf2type.get(surf)
                if gtype and pt:
                    mentions.append((pt, gtype, gstat.get(slot, "na")))
    return mentions


def _write_entity_csv(path, rows, coverage=None):
    cols = ["ec_case", "bcubed_p", "bcubed_r", "bcubed_f1", "pairwise_f1", "macro_f1", "micro_f1",
            "ari", "nmi", "ami", "v_measure", "homogeneity", "completeness", "fmi",
            "n_mentions", "n_pred_clusters", "n_gold_clusters"] + (["grounded_coverage"] if coverage is not None else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for c, s in rows:
            if "pairwise" not in s:
                continue
            row = [c, s["bcubed"]["p"], s["bcubed"]["r"], s["bcubed"]["f1"], s["pairwise"]["f1"],
                   s["macro"]["f1"], s["micro"]["f1"], s.get("ari"), s.get("nmi"), s.get("ami"),
                   s.get("v_measure"), s.get("homogeneity"), s.get("completeness"), s.get("fmi"),
                   s["n_mentions"], s["n_pred_clusters"], s["n_gold_clusters"]]
            if coverage is not None:
                row.append(coverage.get(c))
            w.writerow(row)


def run_entity(gt_kg, gt_schema, pred_dir, output_dir, ground_path=None, p31_path=None):
    os.makedirs(output_dir, exist_ok=True)
    if not (gt_schema and os.path.exists(gt_schema)):
        print(f"  [entity] reference schema not found ({gt_schema}) -> skip entity-type clustering")
        return
    gold = load_kg_txt(gt_kg)
    schema_types = load_schema_types(gt_schema)
    ground = load_grounded_status(ground_path)
    if ground_path and not ground:
        print(f"  [entity] grounded gold not found ({ground_path}) -> all-slots metric only")
    p31_gold = load_p31_gold(p31_path)
    if p31_path and not p31_gold:
        print(f"  [entity] P31 gold not found/empty ({p31_path}) -> P31 variant skipped")
    ec_files = sorted(glob.glob(os.path.join(pred_dir, "entity_canon_ec*.json")))
    if not ec_files:
        print("  [entity] no entity_canon_ec*.json -> skip (entity-canon was off)")
        return
    rows, rows_g, cov, rows_p31, cov_p31 = [], [], {}, [], {}
    for ec_path in ec_files:
        ec_name = os.path.basename(ec_path)[len("entity_canon_"):-len(".json")]
        try:
            ec_data = json.load(open(ec_path, encoding="utf-8"))
        except Exception:
            continue
        mentions = build_entity_mentions(gold, ec_data, schema_types, ground)
        s = cluster_scores([(p, g) for p, g, _ in mentions])
        rows.append((ec_name, s))
        if "pairwise" in s:
            print(f"  [entity] {ec_name:26s} bcubedF1={s['bcubed']['f1']:.3f} "
                  f"ari={s.get('ari')} v={s.get('v_measure')} #predType={s['n_pred_clusters']} "
                  f"#goldType={s['n_gold_clusters']} N={s['n_mentions']}")
        if ground:
            gm = [(p, g) for p, g, st in mentions if st in CONSISTENT]
            sg = cluster_scores(gm)
            rows_g.append((ec_name, sg))
            cov[ec_name] = round(len(gm) / len(mentions), 4) if mentions else 0.0
            if "pairwise" in sg:
                print(f"  [entity-grounded] {ec_name:20s} bcubedF1={sg['bcubed']['f1']:.3f} "
                      f"ari={sg.get('ari')} cov={cov[ec_name]} N={sg['n_mentions']}")
        if p31_gold:
            mp = build_entity_mentions_p31(gold, ec_data, p31_gold)
            sp = cluster_scores(mp)
            rows_p31.append((ec_name, sp))
            cov_p31[ec_name] = round(len(mp) / len(mentions), 4) if mentions else 0.0
            if "pairwise" in sp:
                print(f"  [entity-p31] {ec_name:24s} bcubedF1={sp['bcubed']['f1']:.3f} "
                      f"ari={sp.get('ari')} cov={cov_p31[ec_name]} N={sp['n_mentions']}")
    # all-slots outputs (unchanged columns -> the aggregator reads this)
    json.dump({c: s for c, s in rows},
              open(os.path.join(output_dir, "clustering_entity_metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    _write_entity_csv(os.path.join(output_dir, "clustering_entity_metrics.csv"), rows)
    print("  [entity] saved -> clustering_entity_metrics.csv")
    # KB-grounded subset: score only the slots whose schema type the KB backs (authoritative gold)
    if rows_g:
        json.dump({c: {**s, "grounded_coverage": cov.get(c)} for c, s in rows_g},
                  open(os.path.join(output_dir, "clustering_entity_grounded.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        _write_entity_csv(os.path.join(output_dir, "clustering_entity_grounded.csv"), rows_g, coverage=cov)
        print("  [entity] saved -> clustering_entity_grounded.csv (KB-confirmed slots only)")
    # per-entity P31 gold: type from entity identity (not slot). Sharper reference (ENTITY_TYPE_SCHEMA §8).
    if rows_p31:
        json.dump({c: {**s, "p31_coverage": cov_p31.get(c)} for c, s in rows_p31},
                  open(os.path.join(output_dir, "clustering_entity_p31.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        _write_entity_csv(os.path.join(output_dir, "clustering_entity_p31.csv"), rows_p31, coverage=cov_p31)
        print("  [entity] saved -> clustering_entity_p31.csv (per-entity instance-of gold)")


def run(gt_kg, pred_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    gold = load_kg_txt(gt_kg)
    rows = []
    for case in CASES:
        kg = os.path.join(pred_dir, f"canon_kg_{case}.txt")
        if not os.path.exists(kg):
            print("  skip", case, "(missing)"); continue
        m = build_mentions(load_kg_txt(kg), gold)
        s = cluster_scores(m)
        rows.append((case, s))
        if "pairwise" in s:
            print(f"  {case:28s} bcubedF1={s['bcubed']['f1']:.3f} pairF1={s['pairwise']['f1']:.3f} "
                  f"macroF1={s['macro']['f1']:.3f} #pred={s['n_pred_clusters']} #gold={s['n_gold_clusters']} N={s['n_mentions']}")
    json.dump({c: s for c, s in rows}, open(os.path.join(output_dir,"clustering_metrics.json"),"w",encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir,"clustering_metrics.csv"),"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case","bcubed_p","bcubed_r","bcubed_f1","pairwise_f1","macro_f1","micro_f1",
                    "ari","nmi","ami","v_measure","homogeneity","completeness","fmi",
                    "n_mentions","n_pred_clusters","n_gold_clusters"])
        for c, s in rows:
            if "pairwise" not in s: continue
            w.writerow([c, s["bcubed"]["p"], s["bcubed"]["r"], s["bcubed"]["f1"],
                        s["pairwise"]["f1"], s["macro"]["f1"], s["micro"]["f1"],
                        s.get("ari"), s.get("nmi"), s.get("ami"), s.get("v_measure"),
                        s.get("homogeneity"), s.get("completeness"), s.get("fmi"),
                        s["n_mentions"], s["n_pred_clusters"], s["n_gold_clusters"]])
    if not _HAS_SK:
        print("  [note] sklearn not available -> ARI/NMI/AMI/V-measure/FMI columns blank "
              "(pip install scikit-learn).")
    print("saved ->", output_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset"); ap.add_argument("--method"); ap.add_argument("--iter")
    ap.add_argument("--gt_kg"); ap.add_argument("--gt_schema"); ap.add_argument("--pred_dir")
    ap.add_argument("--output_dir")
    ap.add_argument("--level", choices=["relation", "entity", "both"], default="both",
                    help="relation-type clustering, entity-type clustering, or both")
    ap.add_argument("--gold_ground", help="KB-grounded gold entity-types json (default: by dataset)")
    ap.add_argument("--gold_p31", help="per-entity P31 gold json (default: by dataset; absent -> variant skipped)")
    a = ap.parse_args()
    if a.dataset and a.method and a.iter:
        # convention fallback: any dataset with references/{name}.txt works (scale study)
        gt = a.gt_kg or GT.get(a.dataset) or f"./soda/evaluate/references/{a.dataset}.txt"
        gt_schema = a.gt_schema or GT_SCHEMA.get(a.dataset) or f"./schemas/{a.dataset}_schema.json"
        pred = a.pred_dir or f"./output/{a.dataset}_{a.method}/{a.iter}"
        out = a.output_dir or os.path.join(pred, "eval_clustering")
    else:
        gt, pred, out = a.gt_kg, a.pred_dir, a.output_dir
        gt_schema = a.gt_schema or (GT_SCHEMA.get(a.dataset) if a.dataset else None)
    if a.level in ("relation", "both"):
        run(gt, pred, out)
    if a.level in ("entity", "both"):
        gp = a.gold_ground or (GOLD_GROUND.get(a.dataset) if a.dataset else None)
        p31 = a.gold_p31 or (GOLD_P31.get(a.dataset) if a.dataset else None)
        run_entity(gt, gt_schema, pred, out, ground_path=gp, p31_path=p31)
