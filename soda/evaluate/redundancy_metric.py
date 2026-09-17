"""
Relation redundancy (conciseness / Tier B dim 6) — for Mode 1 self-canon.

Average cosine similarity between each *grounded* relation and its nearest
neighbour in the schema (lower = more distinct = better canonicalization). This
is the "Red." column in the EDC paper (Table 2; their EDC=0.833 on WebNLG) and
the thing canonicalization is supposed to minimise.

Needs an embedder (SBERT) → run on the server. Reads per case:
  canon_schema_{case}.json  (relation_types: {rel: {definition, ...}})
  canon_kg_{case}.json      (to keep only relations that actually appear = grounded)

Run (quick): python soda/evaluate/redundancy_metric.py --dataset webnlg --method <m> --iter iter0
Run (manual): ... --pred_dir <iterdir> [--embedder ...] [--output_dir <dir>]

Mainly meaningful for Mode 1 (self-canon, system-built schema). For Mode 2/3 the
schema is the fixed seed, so redundancy reflects the seed, not the method.
"""
import os, json, csv, argparse

CASES = ["case1_embed_threshold","case2_name_only","case3_name_gendef_edc","case4_name_gendef_abstract",
         "case5_name_detail","case6_name_detail_headtail","case7_detail_typed","case8_concat","case9_weighted"]


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def redundancy(schema, grounded, model, util):
    rt = schema.get("relation_types", {})
    rels = [r for r in rt if (not grounded or r in grounded)]
    if len(rels) < 2:
        return {"redundancy_semantic": 0.0, "num_relations": len(rels)}
    texts = [f"{r}: {rt[r].get('definition','')}" for r in rels]
    emb = model.encode(texts, convert_to_tensor=True)
    sim = util.cos_sim(emb, emb)
    total = 0.0
    for i in range(len(rels)):
        best = max((sim[i][j].item() for j in range(len(rels)) if j != i), default=0.0)
        total += best
    return {"redundancy_semantic": round(total/len(rels), 4), "num_relations": len(rels)}


def run(pred_dir, output_dir, embedder):
    from sentence_transformers import SentenceTransformer, util
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading embedder: {embedder}")
    model = SentenceTransformer(embedder)
    rows = []
    for case in CASES:
        sc_p = os.path.join(pred_dir, f"canon_schema_{case}.json")
        kg_p = os.path.join(pred_dir, f"canon_kg_{case}.json")
        if not os.path.exists(sc_p):
            print("  skip", case, "(missing)"); continue
        schema = load(sc_p)
        grounded = None
        if os.path.exists(kg_p):
            grounded = set(r["relation"] for r in load(kg_p).get("relations", []))
        res = redundancy(schema, grounded, model, util)
        res["case"] = case
        rows.append(res)
        print(f"  {case:28s} redundancy={res['redundancy_semantic']}  #rel={res['num_relations']}")
    json.dump({r["case"]: r for r in rows},
              open(os.path.join(output_dir, "redundancy_metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "redundancy_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["case", "redundancy_semantic", "num_relations"])
        for r in rows:
            w.writerow([r["case"], r["redundancy_semantic"], r["num_relations"]])
    print("saved ->", output_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset"); ap.add_argument("--method"); ap.add_argument("--iter")
    ap.add_argument("--pred_dir"); ap.add_argument("--output_dir")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    a = ap.parse_args()
    if a.dataset and a.method and a.iter:
        pred = a.pred_dir or f"./output/{a.dataset}_{a.method}/{a.iter}"
    else:
        pred = a.pred_dir
    out = a.output_dir or os.path.join(pred, "eval_redundancy")
    run(pred, out, a.embedder)
