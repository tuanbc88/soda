"""
OIE_MISS root-cause diagnosis (RQ1a) — WHY are ~46% of gold triples never extracted?

CONTEXT (DECISIONS 2026-06-24). `error_attribution_metric.py` told us OIE is the
dominant loss on webnlg: ~46% (strict) / ~30% (relaxed) of gold triples never get
their (head, tail) pair extracted. That is the *phenomenon*; this script attributes
the *cause*. It is the local twin of Wang et al. 2024 (LREC-COLING, arXiv 2404.09593),
who found the same signature (P 78 / R 46 on >=7-triple sentences) and blamed
(i) LLM bias toward simple sentences + (ii) decoding that repeats rather than
discovers — so "extract more" prompts don't help (matches our refine null result).

It classifies EVERY gold triple into one mutually-exclusive bucket (post-hoc, NO
model, NO pipeline re-run — reads oie_total.json + gold KG + source text, all
line-aligned by record index). OIE is case-independent, so this reads oie_total.json
directly and the verdict holds for all 9 SC cases.

  For each gold triple (h_g, r_g, t_g) in record i:
    CORRECT            -> (norm h_g, norm t_g) is an OIE (h,t) pair in record i (directed).
    else it is a MISS, bucketed (first match wins, so buckets partition the misses):
    B. SURFACE         -> not strict but token-overlap-recovered (Jaccard>=thr on BOTH
                          h and t vs some OIE pair). Entity content IS there, the
                          surface form / boundary differs ("Karlin" vs "Prague-Karlin").
                          Fix = normalization / entity-linking / eval, NOT a bigger LLM.
                          This is the strict-minus-relaxed gap (~16pp on webnlg).
    A. UNREALIZED      -> an entity surface is not in the source text AND no coref signal
                          -> the fact was never verbalized -> DATASET CEILING, not OIE's
                          fault. Reported separately so it can leave the denominator.
    E. COREF           -> entity not literally in text, but text has a pronoun/def-desc
                          AND the entity recurs in another gold triple of the record
                          (so it was likely pronominalized). Fix = coref-before-OIE.
    C. ENTITY_MISS     -> both entities ARE in the text but OIE never produced >=1 of
                          them anywhere in the record. Fix = better entity extraction.
    D. PAIRING_MISS    -> both entities WERE extracted somewhere in the record but never
                          as the gold (h,t) pair -> the RELATION between them was missed.
                          Fix = candidate-pair enumeration (Wang Stage 2) / coref. THE
                          interesting bucket: if it dominates, it explains why our
                          fresh-entity refine did nothing (the miss is a PAIR miss).

  Plus an IMPLICIT-RELATION tag (author request 2026-06-27): each gold relation is flagged
  *implicit* if no content token of its (camelCase-split) name appears in the source text
  (the predicate is not surfaced -> requires inference, e.g. "Fed chair Powell" -> worksFor).
  This is the extraction-vs-reasoning boundary: implicit relations are an inference task, not
  an extraction error. Reported in overall (relation_implicit_%, missed_%_of_miss) and written
  to oie_implicit_candidates.csv = MISSED implicit-relation triples to inspect by hand
  (HEURISTIC + English-only + noisy: schema relation names are not surface verbs).

  Plus CORRELATION DIAGNOSTICS (author request 2026-06-24): miss-rate + the A-E
  breakdown binned by (1) ITEM LENGTH (char + token, quantile bins) and (2) #GOLD
  TRIPLES per record (1/2/3/4-6/7+) -> does the miss rise with length / triple density
  (the Wang saturation effect)? + an auxiliary DIRECTION-SWAPPED count (gold (h,t)
  absent but (t,h) extracted).

Run (quick mode):
    python soda/evaluate/oie_miss_diagnosis.py --dataset webnlg \
        --method <method_str> --iter iter0
Run (manual):
    python soda/evaluate/oie_miss_diagnosis.py \
        --gt_kg ./soda/evaluate/references/webnlg.txt \
        --oie_json ./output/<run>/iter0/oie_total.json \
        --src_text ./datasets/webnlg.txt \
        --output_dir ./output/<run>/iter0/eval_oie_miss

NOTE: the realized-in-text + coref checks are English heuristics (the 3 gold datasets
are English). On Vietnamese/eduhcmut treat A/E as approximate.
"""

import os
import re
import csv
import json
import math
import argparse

try:   # package import
    from soda.evaluate.retrieval_recall_metric import (
        GT_CONFIG, norm_entity, norm_rel, load_kg_txt,
    )
    from soda.evaluate.literal_normalize import literal_realized_in_text
except ImportError:   # script run
    from retrieval_recall_metric import (
        GT_CONFIG, norm_entity, norm_rel, load_kg_txt,
    )
    from literal_normalize import literal_realized_in_text

# English pronouns / possessives + a few generic definite-description heads.
_PRONOUNS = {
    "he", "she", "it", "they", "him", "her", "them", "his", "hers", "its",
    "their", "theirs", "we", "us", "our", "ours", "i", "you", "your", "yours",
    "this", "that", "these", "those", "who", "which",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_src_text(path):
    """One record per line, aligned to the gold KG / oie_total.json by index."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _ent_tokens(norm_str):
    """Token set from a norm_entity() string ('swords_dublin' -> {swords, dublin})."""
    return set(t for t in norm_str.split("_") if t)


def _text_tokens(text):
    """Lowercased alnum token set of a raw source sentence."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def entity_in_text(norm_ent, text_tok, thr):
    """True if >= thr of the entity's tokens appear in the source-text token set."""
    et = _ent_tokens(norm_ent)
    if not et:
        return True   # empty/degenerate entity: don't charge it as unrealized
    hit = len(et & text_tok)
    return (hit / len(et)) >= thr


def text_has_pronoun(text):
    return bool(_PRONOUNS & _text_tokens(text))


# Relation-grounding heuristic (for the implicit/inference-required tag, 2026-06-27).
# A gold relation is "explicit" if at least one CONTENT token of its (camelCase-split)
# name appears in the source text; otherwise it is *implicit* — the predicate is not
# surfaced and must be inferred (e.g. text "Fed chair Powell" -> (Powell, worksFor, Fed),
# where "works"/"for" are absent). HEURISTIC + English-only + noisy (schema relation
# names are not surface verbs), so it flags CANDIDATES for human review, not ground truth.
_REL_STOP = {"in", "of", "is", "by", "the", "a", "an", "to", "at", "for", "on", "with",
             "as", "and", "or", "s", "was", "were", "be", "been", "has", "have", "had"}


def _rel_tokens(rel_raw):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(rel_raw))   # camelCase -> spaced
    toks = re.findall(r"[a-z0-9]+", s.lower())
    return set(t for t in toks if t not in _REL_STOP and len(t) > 1)


def relation_grounded(rel_raw, text_tok):
    """True if a content token of the relation name appears in the text (explicit);
    False -> implicit/inference-required candidate."""
    rt = _rel_tokens(rel_raw)
    if not rt:
        return True   # degenerate (all function words) -> don't flag as implicit
    return bool(rt & text_tok)


def _jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


# ---------------------------------------------------------------------------
# per-record OIE index
# ---------------------------------------------------------------------------
def build_oie_record(oie_sample):
    """Return (pairs, rev_pairs, entities, pair_token_list).
    pairs       : set of directed (norm_h, norm_t)
    rev_pairs   : set of (norm_t, norm_h)  (for the direction-swap stat)
    entities    : set of norm_entity over all subjects+objects
    pair_tokens : list of (set(h_tok), set(t_tok)) for relaxed matching
    """
    pairs, rev_pairs, entities, pair_tokens = set(), set(), set(), []
    for t in oie_sample or []:
        if not isinstance(t, (list, tuple)) or len(t) < 3:
            continue
        nh, nt = norm_entity(t[0]), norm_entity(t[2])
        pairs.add((nh, nt))
        rev_pairs.add((nt, nh))
        entities.add(nh)
        entities.add(nt)
        pair_tokens.append((_ent_tokens(nh), _ent_tokens(nt)))
    return pairs, rev_pairs, entities, pair_tokens


def relaxed_recovered(gh_tok, gt_tok, pair_tokens, jthr):
    for (oh, ot) in pair_tokens:
        if _jac(gh_tok, oh) >= jthr and _jac(gt_tok, ot) >= jthr:
            return True
    return False


CATS = ["A_unrealized", "B_surface", "F_literal_format",
        "C_entity_miss", "D_pairing_miss", "E_coref"]


def classify_record(gold_sample, oie_sample, text, present_thr, jthr):
    """Return per-record dict of category counts + correct + direction-swap + n_gold."""
    pairs, rev_pairs, entities, pair_tokens = build_oie_record(oie_sample)
    text_tok = _text_tokens(text)
    has_pron = text_has_pronoun(text)

    # entity -> how many gold triples in this record mention it (for the recurrence test)
    ent_freq = {}
    golds = []
    for t in gold_sample or []:
        if not isinstance(t, (list, tuple)) or len(t) < 3:
            continue
        gh, gt = norm_entity(t[0]), norm_entity(t[2])
        gr = norm_rel(t[1])
        golds.append((gh, gr, gt, t))
        ent_freq[gh] = ent_freq.get(gh, 0) + 1
        ent_freq[gt] = ent_freq.get(gt, 0) + 1

    counts = {c: 0 for c in CATS}
    correct = 0
    direction_swapped = 0
    rel_implicit_total = 0       # gold relations whose predicate is not surfaced in text
    rel_implicit_miss = 0        # ... of those, the ones that were MISSED (inference-required miss)
    per_triple = []   # (h, r, t, category, rel_implicit)

    for (gh, gr, gt, raw) in golds:
        rel_implicit = not relation_grounded(raw[1], text_tok)
        if rel_implicit:
            rel_implicit_total += 1
        cat = None
        if (gh, gt) in pairs:
            correct += 1
            cat = "CORRECT"
        else:
            gh_tok, gt_tok = _ent_tokens(gh), _ent_tokens(gt)
            if (gt, gh) in pairs:           # reversed pair exists (overlay stat only)
                direction_swapped += 1
            if relaxed_recovered(gh_tok, gt_tok, pair_tokens, jthr):
                cat = "B_surface"
            else:
                in_h = entity_in_text(gh, text_tok, present_thr)
                in_t = entity_in_text(gt, text_tok, present_thr)
                if not (in_h and in_t):
                    # at least one entity not realized by token overlap. Before charging
                    # it as unrealized, check whether a missing element is a date/number
                    # LITERAL that IS in the text under normalization -> F (format only).
                    missing = [(raw[0], in_h), (raw[2], in_t)]
                    if any((not ok) and literal_realized_in_text(val, text)
                           for val, ok in missing):
                        cat = "F_literal_format"
                    else:
                        miss_ents = [e for e, ok in ((gh, in_h), (gt, in_t)) if not ok]
                        recurs = any(ent_freq.get(e, 0) >= 2 for e in miss_ents)
                        cat = "E_coref" if (has_pron and recurs) else "A_unrealized"
                else:
                    # both entities realized in text -> OIE's fault
                    if (gh not in entities) or (gt not in entities):
                        cat = "C_entity_miss"
                    else:
                        cat = "D_pairing_miss"
            counts[cat] += 1
            if rel_implicit:
                rel_implicit_miss += 1
        per_triple.append((raw[0], raw[1], raw[2], cat, rel_implicit))

    n_gold = len(golds)
    n_miss = sum(counts.values())
    return {
        "n_gold": n_gold,
        "n_miss": n_miss,
        "correct": correct,
        "direction_swapped": direction_swapped,
        "rel_implicit_total": rel_implicit_total,
        "rel_implicit_miss": rel_implicit_miss,
        "len_char": len(text),
        "len_tok": len(text_tok),
        **counts,
    }, per_triple


# ---------------------------------------------------------------------------
# binning
# ---------------------------------------------------------------------------
def _quantile_edges(values, n_bins):
    """Inclusive upper edges for n_bins quantile bins of `values`."""
    if not values:
        return []
    s = sorted(values)
    edges = []
    for b in range(1, n_bins):
        q = b / n_bins
        edges.append(s[min(len(s) - 1, int(math.ceil(q * len(s))) - 1)])
    edges.append(s[-1])
    return edges


def _assign_len_bin(length, edges):
    for i, e in enumerate(edges):
        if length <= e:
            return i
    return len(edges) - 1


def _triple_bin(n_gold):
    if n_gold <= 1:
        return "1"
    if n_gold == 2:
        return "2"
    if n_gold == 3:
        return "3"
    if n_gold <= 6:
        return "4-6"
    return "7+"


def _agg(records, key_fn, order=None):
    """Aggregate per-record dicts into bins. Returns list of bin summaries."""
    bins = {}
    for r in records:
        k = key_fn(r)
        b = bins.setdefault(k, {"bin": k, "n_records": 0, "n_gold": 0, "n_miss": 0,
                                "correct": 0, "len_char_sum": 0, **{c: 0 for c in CATS}})
        b["n_records"] += 1
        b["n_gold"] += r["n_gold"]
        b["n_miss"] += r["n_miss"]
        b["correct"] += r["correct"]
        b["len_char_sum"] += r["len_char"]
        for c in CATS:
            b[c] += r[c]
    rows = []
    keys = order if order else sorted(bins.keys(), key=lambda x: str(x))
    for k in keys:
        if k not in bins:
            continue
        b = bins[k]
        ng = b["n_gold"] or 1
        rows.append({
            "bin": b["bin"],
            "n_records": b["n_records"],
            "mean_len_char": round(b["len_char_sum"] / b["n_records"], 1) if b["n_records"] else 0,
            "n_gold": b["n_gold"],
            "miss_rate_%": round(100.0 * b["n_miss"] / ng, 2),
            "A_unrealized_%": round(100.0 * b["A_unrealized"] / ng, 2),
            "B_surface_%": round(100.0 * b["B_surface"] / ng, 2),
            "F_literal_format_%": round(100.0 * b["F_literal_format"] / ng, 2),
            "C_entity_miss_%": round(100.0 * b["C_entity_miss"] / ng, 2),
            "D_pairing_miss_%": round(100.0 * b["D_pairing_miss"] / ng, 2),
            "E_coref_%": round(100.0 * b["E_coref"] / ng, 2),
        })
    return rows


def run(gt_kg, oie_json, src_text, output_dir, present_thr, jthr, n_len_bins):
    os.makedirs(output_dir, exist_ok=True)
    gold_data = load_kg_txt(gt_kg)
    oie_data = load_json(oie_json)
    src = load_src_text(src_text)

    n = min(len(gold_data), len(oie_data), len(src))
    if not (len(gold_data) == len(oie_data) == len(src)):
        print(f"[WARN] length mismatch gold={len(gold_data)} oie={len(oie_data)} "
              f"src={len(src)} -> using first {n} records.")

    records, all_triples = [], []
    for i in range(n):
        rec, per_triple = classify_record(gold_data[i], oie_data[i], src[i], present_thr, jthr)
        rec["idx"] = i
        records.append(rec)
        for (h, r, t, cat, rel_implicit) in per_triple:
            all_triples.append({"idx": i, "h": h, "r": r, "t": t, "category": cat,
                                "rel_implicit": int(rel_implicit)})

    # ---- overall ----
    n_gold = sum(r["n_gold"] for r in records)
    n_miss = sum(r["n_miss"] for r in records)
    correct = sum(r["correct"] for r in records)
    direction = sum(r["direction_swapped"] for r in records)
    rel_impl_total = sum(r["rel_implicit_total"] for r in records)
    rel_impl_miss = sum(r["rel_implicit_miss"] for r in records)
    cat_tot = {c: sum(r[c] for r in records) for c in CATS}
    pct = lambda x: round(100.0 * x / n_gold, 2) if n_gold else None
    # 3-way rollup of the miss: surface/format (recoverable w/o a better extractor),
    # genuine extraction failure, and ceiling/linking.
    surface_format = cat_tot["B_surface"] + cat_tot["F_literal_format"]
    genuine_extract = cat_tot["C_entity_miss"] + cat_tot["D_pairing_miss"]
    ceiling_linking = cat_tot["A_unrealized"] + cat_tot["E_coref"]
    overall = {
        "n_records": n, "n_gold": n_gold,
        "CORRECT": correct, "CORRECT_%": pct(correct),
        "MISS": n_miss, "MISS_%": pct(n_miss),
        **{c: cat_tot[c] for c in CATS},
        **{f"{c}_%": pct(cat_tot[c]) for c in CATS},
        "direction_swapped_among_miss": direction,
        "direction_swapped_%": pct(direction),
        # implicit/inference-required relations (heuristic: predicate not surfaced in text)
        "relation_implicit_total": rel_impl_total,
        "relation_implicit_%": pct(rel_impl_total),
        "relation_implicit_missed": rel_impl_miss,
        "relation_implicit_missed_%_of_gold": pct(rel_impl_miss),
        "relation_implicit_missed_%_of_miss": round(100.0 * rel_impl_miss / n_miss, 2) if n_miss else None,
        "relation_implicit_NOTE": ("UPPER BOUND on inference-required: 'implicit'=='relation predicate not "
                                   "lexically surfaced in text', which OVER-counts schema-name-vs-surface "
                                   "mismatch (e.g. relation 'country' vs text 'from Brazil' is explicit but "
                                   "not surfaced). True inference-required is a SUBSET; review "
                                   "oie_implicit_candidates.csv or use an LLM-judge for a clean "
                                   "explicit/coref/implicit split."),
        # the headline rollup
        "rollup_surface_or_format_%": pct(surface_format),
        "rollup_genuine_extraction_%": pct(genuine_extract),
        "rollup_ceiling_or_linking_%": pct(ceiling_linking),
        "params": {"present_thr": present_thr, "relaxed_jaccard": jthr},
    }

    # ---- bins ----
    len_vals = [r["len_char"] for r in records if r["n_gold"] > 0]
    edges = _quantile_edges(len_vals, n_len_bins)
    len_label = {i: f"Q{i+1}(<= {e}c)" for i, e in enumerate(edges)}
    len_bins = _agg(records, lambda r: _assign_len_bin(r["len_char"], edges),
                    order=list(range(len(edges))))
    for row in len_bins:
        row["bin"] = len_label.get(row["bin"], row["bin"])
    triple_bins = _agg(records, lambda r: _triple_bin(r["n_gold"]),
                       order=["1", "2", "3", "4-6", "7+"])

    # ---- write ----
    summary = {"overall": overall, "by_item_length": len_bins, "by_n_triples": triple_bins}
    with open(os.path.join(output_dir, "oie_miss_diagnosis.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    rec_keys = ["idx", "len_char", "len_tok", "n_gold", "n_miss", "correct",
                "direction_swapped", "rel_implicit_total", "rel_implicit_miss"] + CATS
    with open(os.path.join(output_dir, "oie_miss_per_record.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rec_keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in rec_keys})

    with open(os.path.join(output_dir, "oie_miss_per_triple.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "h", "r", "t", "category", "rel_implicit"])
        w.writeheader()
        w.writerows(all_triples)

    # shortlist: MISSED gold triples whose relation is implicit (inference-required) — the
    # items to inspect for the "extraction vs reasoning" boundary (heuristic; review by hand).
    implicit_missed = [t for t in all_triples
                       if t["rel_implicit"] and t["category"] != "CORRECT"]
    with open(os.path.join(output_dir, "oie_implicit_candidates.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "h", "r", "t", "category", "rel_implicit"])
        w.writeheader()
        w.writerows(implicit_missed)

    # ---- print ----
    print("=" * 64)
    print(">>> OIE_MISS DIAGNOSIS")
    print(f"GT KG    : {gt_kg}")
    print(f"OIE JSON : {oie_json}")
    print(f"SRC TEXT : {src_text}")
    print(f"records={n}  gold_triples={n_gold}")
    print("=" * 64)
    print(f"CORRECT        {overall['CORRECT_%']}%   ({correct})")
    print(f"MISS           {overall['MISS_%']}%   ({n_miss})")
    print(f"  A unrealized {overall['A_unrealized_%']}%   (residual: alias/morphology/true-miss)")
    print(f"  B surface    {overall['B_surface_%']}%   (entity boundary; linking/eval fix)")
    print(f"  F literal    {overall['F_literal_format_%']}%   (date/number format; normalization fix)")
    print(f"  C entity     {overall['C_entity_miss_%']}%   (entity never extracted; EE/NER)")
    print(f"  D pairing    {overall['D_pairing_miss_%']}%   (relation between known ents missed)")
    print(f"  E coref      {overall['E_coref_%']}%   (pronominalized entity)")
    print(f"  [aux] direction-swapped among miss: {overall['direction_swapped_%']}%")
    print(f"  [implicit~] relation NOT lexically surfaced: {overall['relation_implicit_%']}% of gold "
          f"| missed+not-surfaced {overall['relation_implicit_missed_%_of_miss']}% of miss "
          f"(UPPER BOUND: over-counts schema-name vs surface; true inference-required is a subset "
          f"-> review oie_implicit_candidates.csv / LLM-judge)")
    print(f"  ROLLUP -> surface/format(B+F) {overall['rollup_surface_or_format_%']}% | "
          f"genuine extraction(C+D) {overall['rollup_genuine_extraction_%']}% | "
          f"ceiling/linking(A+E) {overall['rollup_ceiling_or_linking_%']}%")
    hdr = (f"{'bin':14s} {'#rec':>5s} {'meanC':>6s} {'gold':>5s} {'miss%':>6s} "
           f"{'A%':>5s} {'B%':>5s} {'F%':>5s} {'C%':>5s} {'D%':>5s} {'E%':>5s}")
    rowfmt = lambda r: (f"{r['bin']:14s} {r['n_records']:>5d} {r['mean_len_char']:>6} {r['n_gold']:>5d} "
                        f"{r['miss_rate_%']:>6} {r['A_unrealized_%']:>5} {r['B_surface_%']:>5} "
                        f"{r['F_literal_format_%']:>5} {r['C_entity_miss_%']:>5} "
                        f"{r['D_pairing_miss_%']:>5} {r['E_coref_%']:>5}")
    print("\n-- by item length --")
    print(hdr)
    for r in len_bins:
        print(rowfmt(r))
    print("\n-- by #gold triples / record --")
    print(hdr)
    for r in triple_bins:
        print(rowfmt(r))
    print(f"\nSaved -> {output_dir}/oie_miss_diagnosis.json (+ per_record.csv, per_triple.csv)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="webnlg|rebel|wiki-nre (defaults for gold + src text)")
    ap.add_argument("--method", help="quick mode: pred_dir = ./output/{dataset}_{method}/{iter}")
    ap.add_argument("--iter", help="quick mode iter dir, e.g. iter0")
    ap.add_argument("--pred_dir", help="iter dir holding oie_total.json (overrides --method/--iter)")
    ap.add_argument("--gt_kg", help="explicit gold KG .txt (overrides --dataset)")
    ap.add_argument("--oie_json", help="explicit oie_total.json (overrides pred_dir)")
    ap.add_argument("--src_text", help="explicit source text (one record/line; overrides --dataset)")
    ap.add_argument("--output_dir")
    ap.add_argument("--present_thr", type=float, default=0.5,
                    help="fraction of an entity's tokens that must appear in the text to count as realized")
    ap.add_argument("--relaxed_jaccard", type=float, default=0.5,
                    help="token-overlap Jaccard for the relaxed (surface/boundary) recovery test")
    ap.add_argument("--len_bins", type=int, default=4, help="number of item-length quantile bins")
    args = ap.parse_args()

    gt_kg = args.gt_kg or (GT_CONFIG.get(args.dataset)
                           or (f"./soda/evaluate/references/{args.dataset}.txt" if args.dataset else None))
    if not gt_kg or not os.path.isfile(gt_kg):
        raise SystemExit(f"[oie_miss] gold KG not found: {gt_kg} (pass --gt_kg or a valid --dataset)")

    pred_dir = args.pred_dir
    if not pred_dir and args.dataset and args.method and args.iter:
        pred_dir = f"./output/{args.dataset}_{args.method}/{args.iter}"
    oie_json = args.oie_json or (os.path.join(pred_dir, "oie_total.json") if pred_dir else None)
    if not oie_json or not os.path.isfile(oie_json):
        raise SystemExit(f"[oie_miss] oie_total.json not found: {oie_json} "
                         f"(pass --oie_json or --pred_dir or --dataset/--method/--iter)")

    src_text = args.src_text or (f"./datasets/{args.dataset}.txt" if args.dataset else None)
    if not src_text or not os.path.isfile(src_text):
        raise SystemExit(f"[oie_miss] source text not found: {src_text} (pass --src_text or a valid --dataset)")

    output_dir = args.output_dir or (os.path.join(pred_dir, "eval_oie_miss") if pred_dir
                                     else os.path.join(os.path.dirname(oie_json), "eval_oie_miss"))
    run(gt_kg, oie_json, src_text, output_dir, args.present_thr, args.relaxed_jaccard, args.len_bins)


if __name__ == "__main__":
    main()
