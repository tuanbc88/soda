"""
Candidate-retrieval recall@k for the SC stage (RQ1b diagnostic).

WHY (see CLAUDE.md Research Design + the embedder discussion 2026-06-18):
In our pipeline the schema embedder does ONE job — pick the top-k candidate
relations shown to the LLM in the canonicalization MCQ. The merge decision is the
LLM's; the embedder only controls candidate recall. The hypothesis (why a bigger
embedder / cluster-augmented retrieval barely moves canonicalization on webnlg) is
that on a small target schema **recall@k is already saturated** — the gold relation
is almost always inside the top-k regardless of embedder — so retrieval is NOT the
bottleneck. This script measures that directly, with NO model and NO SC re-run:

  For every OIE triplet that matches a gold triple by entity-pair (head, tail),
  was the gold relation among the top-k candidates the retriever surfaced?
  -> recall@1 / @2 / @3 / @5, candidate-pool hit-rate, and mean rank of the gold.

If recall@k ~ 0.95+ for BOTH MiniLM and bge-m3, that is the smoking gun: the
embedder cannot matter because the right answer is already shown to the LLM. To
turn "no effect" into a controlled finding, also sweep k (needs an SC re-run with a
smaller retrieval k — that part is a server run, not this script).

FAIR FOR **Mode 2 / Mode 3** only (candidate pool = the seed/gold schema, so the
gold relation NAME is in the pool). In Mode 1 the pool is self-invented names, so a
name-exact recall is meaningless — use clustering_metric.py there instead.

Reads the SC debug trace (gated by EDC_DEBUG_LOGS, default ON):
  - NEW runs:  <pred_dir>/sc_mcp_logs.jsonl   (one record per idx+case)
  - OLD runs:  <pred_dir>/sc_mcp_logs/idx_*.{case}.txt
Both are supported; JSONL is preferred if present.

Run (quick mode):
    python soda/evaluate/retrieval_recall_metric.py --dataset webnlg \
        --pred_dir ./output/<run>/iter0
Run (manual):
    python soda/evaluate/retrieval_recall_metric.py \
        --gt_kg ./soda/evaluate/references/webnlg.txt \
        --pred_dir ./output/<run>/iter0 \
        --output_dir ./output/<run>/iter0/eval_retrieval --ks 1,2,3,5
"""

import os
import re
import ast
import csv
import json
import glob
import argparse

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
    "webnlg":   "./soda/evaluate/references/webnlg.txt",
    "rebel":    "./soda/evaluate/references/rebel.txt",
    "wiki-nre": "./soda/evaluate/references/wiki-nre.txt",
}


# ---------------------------------------------------------------------------
# normalisation (shared style with mode1_metric.py)
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


def norm_rel(rel):
    """Canonicalise a relation name so candidate names match gold names across
    underscore / camelCase / spaced forms. `birthPlace` / `birth_place` ->
    'birth place'; 'located in the administrative territorial entity' unchanged."""
    s = str(rel)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)        # camelCase split
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


# ---------------------------------------------------------------------------
# gold KG
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


def build_gold_index(gold_data):
    """record_idx -> { (norm_h, norm_t): set(norm_rel) }."""
    idx = []
    for sample in gold_data:
        d = {}
        for t in sample:
            if len(t) >= 3:
                key = (norm_entity(t[0]), norm_entity(t[2]))
                d.setdefault(key, set()).add(norm_rel(t[1]))
        idx.append(d)
    return idx


# ---------------------------------------------------------------------------
# SC trace loaders -> yield (record_idx, input_triplet, top_k_names) per triplet
# ---------------------------------------------------------------------------
def iter_triplets_jsonl(pred_dir, case):
    path = os.path.join(pred_dir, "sc_mcp_logs.jsonl")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("case") != case:
                continue
            idx = rec.get("idx")
            for t in rec.get("triplets", []):
                trip = t.get("input_triplet")
                topk = t.get("top_k") or []
                # pred_relation IS the final canonical relation (edc_framework
                # stores final_rel here), or None for new/unmatched decisions.
                final_pred = t.get("pred_relation")
                if trip and len(trip) >= 3:
                    yield idx, trip, [str(c) for c in topk], final_pred


_TRIP_HDR = re.compile(r"=+\s*TRIPLET\s+\d+\s*=+")
_CAND = re.compile(r"^\s*(\d+)\.\s+(\S.*?)\s+\|\s+score=", re.M)
_FINAL = re.compile(r"^\s*\d+\.\s+(\S.*?)\s+\|\s+score=.*\[FINAL\]", re.M)


def iter_triplets_dir(pred_dir, case):
    logdir = os.path.join(pred_dir, "sc_mcp_logs")
    if not os.path.isdir(logdir):
        return
    files = sorted(glob.glob(os.path.join(logdir, f"idx_*.{case}.txt")))
    for fp in files:
        m = re.search(r"idx_(\d+)\.", os.path.basename(fp))
        idx = int(m.group(1)) if m else None
        try:
            txt = open(fp, "r", encoding="utf-8").read()
        except Exception:
            continue
        blocks = _TRIP_HDR.split(txt)[1:]   # drop the file header before TRIPLET 0
        for blk in blocks:
            # input triplet
            trip = None
            mi = re.search(r"\[INPUT TRIPLET\]\s*\n(.+)", blk)
            if mi:
                try:
                    trip = ast.literal_eval(mi.group(1).strip())
                except Exception:
                    trip = None
            if not trip or len(trip) < 3:
                continue
            # candidates (ranked) — only within the [RETRIEVAL CANDIDATES] section
            seg = blk
            ci = blk.find("[RETRIEVAL CANDIDATES]")
            if ci != -1:
                pj = blk.find("[PROMPT]", ci)
                seg = blk[ci: pj if pj != -1 else len(blk)]
            topk = [name.strip() for _, name in _CAND.findall(seg)]
            mf = _FINAL.search(seg)
            final_pred = mf.group(1).strip() if mf else None
            yield idx, trip, topk, final_pred


def load_case_triplets(pred_dir, case):
    """Prefer JSONL; fall back to the per-item dir. Returns a list."""
    rows = list(iter_triplets_jsonl(pred_dir, case))
    if rows:
        return rows, "jsonl"
    rows = list(iter_triplets_dir(pred_dir, case))
    return rows, "dir"


# ---------------------------------------------------------------------------
# per-case recall@k
# ---------------------------------------------------------------------------
def evaluate_case(pred_dir, case, gold_index, ks):
    rows, src = load_case_triplets(pred_dir, case)
    n_trip = len(rows)
    n_matched = 0           # OIE triplets whose (h,t) hits a gold pair
    hits = {k: 0 for k in ks}
    gold_in_pool = 0        # gold rel present anywhere in the logged top_k
    rank_sum = 0
    rank_n = 0
    max_cands = 0

    for idx, trip, topk, _final in rows:
        max_cands = max(max_cands, len(topk))
        if idx is None or idx >= len(gold_index):
            continue
        key = (norm_entity(trip[0]), norm_entity(trip[2]))
        gold_rels = gold_index[idx].get(key)
        if not gold_rels:
            continue
        n_matched += 1
        cand_norm = [norm_rel(c) for c in topk]
        # rank of first matching gold relation in the ranked candidate list
        rank = None
        for pos, c in enumerate(cand_norm, start=1):
            if c in gold_rels:
                rank = pos
                break
        if rank is not None:
            gold_in_pool += 1
            rank_sum += rank
            rank_n += 1
        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1

    out = {
        "case": case,
        "source": src,
        "n_triplets": n_trip,
        "n_matched_gold": n_matched,
        "coverage": round(n_matched / n_trip, 4) if n_trip else None,
        "max_candidates_logged": max_cands,
        "gold_in_pool_rate": round(gold_in_pool / n_matched, 4) if n_matched else None,
        "mean_gold_rank": round(rank_sum / rank_n, 3) if rank_n else None,
    }
    for k in ks:
        out[f"recall@{k}"] = round(hits[k] / n_matched, 4) if n_matched else None
    return out


def run(gt_kg, pred_dir, output_dir, ks, cases):
    os.makedirs(output_dir, exist_ok=True)
    gold_index = build_gold_index(load_kg_txt(gt_kg))

    print(f"GT KG     : {gt_kg}  ({len(gold_index)} records)")
    print(f"PRED DIR  : {pred_dir}")
    print(f"ks        : {ks}\n")

    rows = []
    for case in cases:
        r = evaluate_case(pred_dir, case, gold_index, ks)
        rows.append(r)
        if r["n_matched_gold"]:
            rk = "  ".join(f"R@{k}={r[f'recall@{k}']}" for k in ks)
            print(f"{case:28s} src={r['source']:5s} matched={r['n_matched_gold']:5d} "
                  f"cov={r['coverage']}  {rk}  pool={r['gold_in_pool_rate']} "
                  f"rank={r['mean_gold_rank']}")
        else:
            print(f"{case:28s} src={r['source']:5s}  NO matched triplets "
                  f"(empty/missing SC trace, or Mode 1 — see module docstring)")

    out_json = os.path.join(output_dir, "retrieval_recall.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    keys = ["case", "source", "n_triplets", "n_matched_gold", "coverage",
            "max_candidates_logged"] + [f"recall@{k}" for k in ks] + \
           ["gold_in_pool_rate", "mean_gold_rank"]
    with open(os.path.join(output_dir, "retrieval_recall.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    print(f"\nSaved -> {output_dir}/retrieval_recall.csv (+ .json)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="webnlg|rebel|wiki-nre (quick-mode gold path)")
    ap.add_argument("--method", help="quick mode: builds pred_dir ./output/{dataset}_{method}/{iter}")
    ap.add_argument("--iter", help="quick mode iter dir name, e.g. iter0")
    ap.add_argument("--gt_kg", help="explicit gold KG .txt (overrides --dataset)")
    ap.add_argument("--pred_dir", help="the iter dir holding sc_mcp_logs(.jsonl); or use --method/--iter")
    ap.add_argument("--output_dir")
    ap.add_argument("--ks", default="1,2,3,5", help="comma list, e.g. 1,2,3,5")
    ap.add_argument("--cases", help="comma list of cases (default: all 8)")
    args = ap.parse_args()

    gt_kg = args.gt_kg or (GT_CONFIG.get(args.dataset)
                           or (f"./soda/evaluate/references/{args.dataset}.txt" if args.dataset else None))
    if not gt_kg or not os.path.isfile(gt_kg):
        raise SystemExit(f"[retrieval_recall] gold KG not found: {gt_kg} "
                         f"(pass --gt_kg or a valid --dataset)")
    pred_dir = args.pred_dir
    if not pred_dir and args.dataset and args.method and args.iter:
        pred_dir = f"./output/{args.dataset}_{args.method}/{args.iter}"
    if not pred_dir:
        raise SystemExit("[retrieval_recall] need --pred_dir (or --dataset/--method/--iter)")
    output_dir = args.output_dir or os.path.join(pred_dir, "eval_retrieval")
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    cases = [c.strip() for c in args.cases.split(",")] if args.cases else CASES

    run(gt_kg, pred_dir, output_dir, ks, cases)


if __name__ == "__main__":
    main()
