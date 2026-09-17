"""
Intrinsic structural metrics for an automatically-built KG (Tier B / Nội dung 01).

Runs on the constructed KG alone (NO gold KG needed) → works for ANY dataset,
including eduhcmut. Reads per case:
  canon_kg_{case}.json   = {entities:[{id,type}], relations:[{head,relation,tail}]}
  canon_schema_{case}.json = {relation_types:{rel:{head_type,tail_type,...}}, entity_types:{...}}
  oie_total.json         = raw OIE triplets (for before/after-canon conciseness)

Covers 4 of the 6 Tier B dimensions (the structural, gold-free ones):
  3. CONNECTIVITY  — density, avg degree, #components, %largest comp, #isolated nodes
  4. CONSISTENCY   — schema type-conformance: % triples whose head/tail entity types
                     satisfy the relation's schema head_type/tail_type
  5. DIVERSITY     — #distinct relation/entity types + Shannon entropy + top-type share
  6. CONCISENESS   — #entities, #relation types after canon vs raw OIE (reduction);
                     (semantic relation redundancy needs an embedder → run separately)
Dimensions 1 (accuracy) & 2 (relative-recall) need a gold/annotated subset and are
covered elsewhere (run_full_evaluation / the eduhcmut annotated subset).

NOTE (research): connectivity / diversity / conciseness are DESCRIPTIVE and gameable
(over-merging shrinks them while hurting accuracy) — always report anchored by
accuracy + completeness, never alone.

Pure-python (no networkx, no embedder). Eval-only.

Run (quick): python soda/evaluate/intrinsic_metrics.py --dataset webnlg --method <m> --iter iter0
Run (manual): ... --pred_dir <iterdir> [--output_dir <dir>]
"""
import os, re, ast, json, csv, math, argparse
from collections import Counter, defaultdict

CASES = ["case1_embed_threshold","case2_name_only","case3_name_gendef_edc","case4_name_gendef_abstract",
         "case5_name_detail","case6_name_detail_headtail","case7_detail_typed","case8_concat","case9_weighted"]


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def entropy(counter):
    n = sum(counter.values())
    if n == 0:
        return 0.0
    return round(-sum((c/n) * math.log2(c/n) for c in counter.values() if c > 0), 4)


# ---------- union-find for connected components (undirected) ----------
class UF:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def connectivity(entities, relations):
    nodes = set(e["id"] for e in entities)
    for r in relations:
        nodes.add(r["head"]); nodes.add(r["tail"])
    deg = Counter()
    uf = UF()
    edge_pairs = set()
    for r in relations:
        h, t = r["head"], r["tail"]
        deg[h] += 1; deg[t] += 1
        uf.union(h, t)
        edge_pairs.add((h, t))
    N = len(nodes); E = len(edge_pairs)
    comp = defaultdict(int)
    for n in nodes: comp[uf.find(n)] += 1
    sizes = sorted(comp.values(), reverse=True) if comp else [0]
    isolated = sum(1 for n in nodes if deg[n] == 0)
    max_possible = N * (N - 1) / 2 if N > 1 else 1
    return {
        "n_nodes": N, "n_edges_unique_pairs": E,
        "avg_degree": round(2 * E / N, 3) if N else 0.0,
        "density": round(E / max_possible, 6) if N > 1 else 0.0,
        "n_components": len(sizes),
        "largest_component_frac": round(sizes[0] / N, 4) if N else 0.0,
        "n_isolated_nodes": isolated,
        "isolated_frac": round(isolated / N, 4) if N else 0.0,
    }


def consistency(entities, relations, schema):
    type_of = {e["id"]: e.get("type") for e in entities}
    rt = schema.get("relation_types", {})
    checked = conform = uncovered = 0
    for r in relations:
        rel = r["relation"]
        if rel not in rt:
            uncovered += 1; continue
        sht = rt[rel].get("head_type"); stt = rt[rel].get("tail_type")
        ht = type_of.get(r["head"]); tt = type_of.get(r["tail"])
        if ht is None or tt is None or not sht or not stt:
            continue
        checked += 1
        head_ok = (sht == "Entity") or (ht == sht)
        tail_ok = (stt == "Entity") or (tt == stt)
        if head_ok and tail_ok:
            conform += 1
    return {
        "type_conformance": round(conform / checked, 4) if checked else None,
        "n_checked": checked, "n_uncovered_relation": uncovered,
    }


def diversity(entities, relations):
    rel_types = Counter(r["relation"] for r in relations)
    ent_types = Counter(e.get("type") for e in entities if e.get("type"))
    def top_share(c):
        n = sum(c.values()); return round(max(c.values())/n, 4) if n else 0.0
    return {
        "n_distinct_relations": len(rel_types),
        "relation_type_entropy": entropy(rel_types),
        "relation_top_share": top_share(rel_types),
        "n_distinct_entity_types": len(ent_types),
        "entity_type_entropy": entropy(ent_types),
        "entity_top_type_share": top_share(ent_types),
    }


def _norm(x):
    return re.sub(r"\s+", "_", str(x).strip().lower())


def conciseness(entities, relations, oie_total):
    # after canon
    canon_rel = set(r["relation"] for r in relations)
    canon_ent = set(e["id"] for e in entities)
    # before canon (raw OIE)
    raw_rel, raw_ent = set(), set()
    for sample in (oie_total or []):
        for t in sample:
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                raw_rel.add(_norm(t[1])); raw_ent.add(_norm(t[0])); raw_ent.add(_norm(t[2]))
    out = {
        "n_relation_types_after_canon": len(canon_rel),
        "n_entities_after_canon": len(canon_ent),
        "n_raw_relation_surface_forms": len(raw_rel),
        "relation_reduction_ratio": round(1 - len(canon_rel)/len(raw_rel), 4) if raw_rel else None,
    }
    return out


def run_case(pred_dir, case):
    kg_p = os.path.join(pred_dir, f"canon_kg_{case}.json")
    sc_p = os.path.join(pred_dir, f"canon_schema_{case}.json")
    if not os.path.exists(kg_p):
        return None
    kg = load_json(kg_p)
    schema = load_json(sc_p) if os.path.exists(sc_p) else {"relation_types": {}}
    ents, rels = kg.get("entities", []), kg.get("relations", [])
    oie_p = os.path.join(pred_dir, "oie_total.json")
    oie = load_json(oie_p) if os.path.exists(oie_p) else []
    return {
        "connectivity": connectivity(ents, rels),
        "consistency": consistency(ents, rels, schema),
        "diversity": diversity(ents, rels),
        "conciseness": conciseness(ents, rels, oie),
    }


def run(pred_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    allres = {}
    for case in CASES:
        r = run_case(pred_dir, case)
        if r is None:
            print("  skip", case, "(missing)"); continue
        allres[case] = r
        c, cons, dv, cc = r["connectivity"], r["consistency"], r["diversity"], r["conciseness"]
        print(f"  {case:28s} comp={c['n_components']} largest={c['largest_component_frac']:.2f} "
              f"isol={c['isolated_frac']:.2f} | conform={cons['type_conformance']} | "
              f"#rel={dv['n_distinct_relations']} relEntropy={dv['relation_type_entropy']} | "
              f"relReduce={cc['relation_reduction_ratio']}")
    json.dump(allres, open(os.path.join(output_dir,"intrinsic_metrics.json"),"w",encoding="utf-8"),
              ensure_ascii=False, indent=2)
    # flat csv
    with open(os.path.join(output_dir,"intrinsic_metrics.csv"),"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case","n_nodes","avg_degree","density","n_components","largest_comp_frac",
                    "isolated_frac","type_conformance","n_distinct_relations","relation_entropy",
                    "entity_type_entropy","relation_reduction_ratio"])
        for case, r in allres.items():
            c, cons, dv, cc = r["connectivity"], r["consistency"], r["diversity"], r["conciseness"]
            w.writerow([case, c["n_nodes"], c["avg_degree"], c["density"], c["n_components"],
                        c["largest_component_frac"], c["isolated_frac"], cons["type_conformance"],
                        dv["n_distinct_relations"], dv["relation_type_entropy"],
                        dv["entity_type_entropy"], cc["relation_reduction_ratio"]])
    print("saved ->", output_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset"); ap.add_argument("--method"); ap.add_argument("--iter")
    ap.add_argument("--pred_dir"); ap.add_argument("--output_dir")
    a = ap.parse_args()
    if a.dataset and a.method and a.iter:
        pred = a.pred_dir or f"./output/{a.dataset}_{a.method}/{a.iter}"
    else:
        pred = a.pred_dir
    out = a.output_dir or os.path.join(pred, "eval_intrinsic")
    run(pred, out)
