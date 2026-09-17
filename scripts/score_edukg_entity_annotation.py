"""Score the Edu-KG entity annotation once the students return their sheets (review L2, Q2).

Answers the two questions the annotation was built for:

  1. **Do the annotators agree?** With two sheets, Cohen's kappa; with three or more, Fleiss' kappa plus
     every pairwise Cohen. All are reported **beside raw agreement**, because the label set is open: with
     free-text labels kappa is depressed by spelling variants, so a kappa that looks poor next to a high
     raw agreement means label drift, not disagreement about the entities. Normalising case, spacing and
     punctuation removes the cheapest variants; it cannot remove synonymy, and that residue is a real
     limit on the number.

  2. **Does the entity-signal inventory inversion on Edu-KG correspond to an accuracy inversion?** Entity
     B-cubed of each signal's grouping against the human grouping, reported PER STRATUM. The `disagree`
     stratum is where the signals differ and so carries the information; the `random` stratum is the
     unbiased estimate. Reporting one without the other misleads in opposite directions.

Adjudication, in order of preference:
  * `--adjudicated FILE`: a single reconciled sheet. Preferred whenever one exists.
  * **majority** across three or more sheets: a label carried by at least half the annotators who filled
    that row. Rows with no majority are counted and named in the output rather than dropped quietly, since
    silently discarding the hard rows is exactly how a subset like this flatters itself.
  * **unanimous** subset: rows where every annotator wrote the same normalised label. Reported alongside
    the majority view because the two bound the answer from either side.

Usage
-----
    python scripts/score_edukg_entity_annotation.py --sheets A.csv B.csv C.csv
    python scripts/score_edukg_entity_annotation.py --adjudicated final.csv
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations

KEY = "assets/review/peer_review_by_models_l2/edukg_entity_annotation/key_hidden.csv"
SIGNALS = ["ec1_type", "ec2_type", "ec3_type"]
SIGNAL_LABEL = {"ec1_type": "phi_e=1 name", "ec2_type": "phi_e=2 name+def",
                "ec3_type": "phi_e=3 def+parent"}


def norm(label):
    """Fold the cheap free-text variants: case, spacing, punctuation, underscores."""
    s = (label or "").strip().lower()
    s = re.sub(r"[\s_\-]+", " ", s)
    s = re.sub(r"[^\w ]+", "", s, flags=re.UNICODE)
    return s.strip()


def read_labels(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {r["id"].strip(): norm(r.get("gold_type")) for r in rows if norm(r.get("gold_type"))}


def read_key(path):
    with open(path, encoding="utf-8-sig") as f:
        return {r["id"].strip(): r for r in csv.DictReader(f)}


def cohens_kappa(a, b):
    ids = sorted(set(a) & set(b))
    if not ids:
        return None, 0, None
    n = len(ids)
    po = sum(1 for i in ids if a[i] == b[i]) / n
    ca, cb = Counter(a[i] for i in ids), Counter(b[i] for i in ids)
    pe = sum((ca[k] / n) * (cb.get(k, 0) / n) for k in ca)
    return ((po - pe) / (1 - pe) if pe < 1 else None), n, po


def fleiss_kappa(sheets):
    """Fleiss' kappa over the rows every annotator filled (fixed number of raters per row)."""
    ids = sorted(set.intersection(*[set(s) for s in sheets]))
    m = len(sheets)
    if m < 2 or not ids:
        return None, 0, None
    cats = sorted({s[i] for s in sheets for i in ids})
    idx = {c: k for k, c in enumerate(cats)}
    pi, pj = [], [0.0] * len(cats)
    for i in ids:
        row = [0] * len(cats)
        for s in sheets:
            row[idx[s[i]]] += 1
        for k, v in enumerate(row):
            pj[k] += v
        pi.append((sum(v * v for v in row) - m) / (m * (m - 1)))
    n = len(ids)
    pj = [v / (n * m) for v in pj]
    pbar = sum(pi) / n
    pebar = sum(v * v for v in pj)
    return ((pbar - pebar) / (1 - pebar) if pebar < 1 else None), n, pbar


def adjudicate(sheets):
    """Majority label per row, plus the rows that have none."""
    ids = sorted(set().union(*[set(s) for s in sheets]))
    gold, tied = {}, []
    for i in ids:
        votes = Counter(s[i] for s in sheets if i in s)
        if not votes:
            continue
        top, n_top = votes.most_common(1)[0]
        if n_top * 2 >= sum(votes.values()) and n_top > 1:
            gold[i] = top
        elif len(votes) == 1:
            gold[i] = top          # only one annotator reached this row
        else:
            tied.append(i)
    return gold, tied


def bcubed(pred, gold):
    items = sorted(set(pred) & set(gold))
    if not items:
        return None
    pc, gc = defaultdict(set), defaultdict(set)
    for i in items:
        pc[pred[i]].add(i)
        gc[gold[i]].add(i)
    p = sum(len(pc[pred[i]] & gc[gold[i]]) / len(pc[pred[i]]) for i in items) / len(items)
    r = sum(len(pc[pred[i]] & gc[gold[i]]) / len(gc[gold[i]]) for i in items) / len(items)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4), len(items)


def score(gold, key, title):
    print(f"\n=== entity B-cubed against {title} ===")
    strata = sorted({key[i]["stratum"] for i in gold if i in key})
    for stratum in strata + ["ALL"]:
        sub = {i: g for i, g in gold.items()
               if i in key and (stratum == "ALL" or key[i]["stratum"] == stratum)}
        if not sub:
            continue
        print(f"  stratum {stratum:9} n={len(sub)}")
        for sig in SIGNALS:
            res = bcubed({i: norm(key[i][sig]) for i in sub}, sub)
            if res:
                p, r, f, _ = res
                print(f"      {SIGNAL_LABEL[sig]:22} P {p:.3f}  R {r:.3f}  F1 {f:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", nargs="*", default=[], help="returned sheets, one per annotator")
    ap.add_argument("--adjudicated", help="single reconciled sheet; preferred when it exists")
    ap.add_argument("--key", default=KEY)
    args = ap.parse_args()

    if not args.sheets and not args.adjudicated:
        ap.error("give --sheets and/or --adjudicated")
    if not os.path.exists(args.key):
        ap.error(f"hidden key not found: {args.key}")
    key = read_key(args.key)
    print(f"key: {len(key)} rows  ({dict(Counter(r['stratum'] for r in key.values()))})")

    if args.adjudicated:
        gold = read_labels(args.adjudicated)
        print(f"\nadjudicated labels: {len(gold)} of {len(key)}")
        score(gold, key, "the adjudicated labels")
        if not args.sheets:
            return

    sheets = [read_labels(p) for p in args.sheets]
    for p, l in zip(args.sheets, sheets):
        print(f"  {os.path.basename(p):44} {len(l):4d} labelled, {len(set(l.values())):3d} distinct types")

    if len(sheets) >= 2:
        print("\n=== inter-annotator agreement ===")
        print("  NB an open label set depresses kappa through spelling variants; read each kappa beside its")
        print("     raw agreement, and treat a large gap between the two as label drift to reconcile.")
        for (i, a), (j, b) in combinations(list(enumerate(sheets)), 2):
            k, n, po = cohens_kappa(a, b)
            na, nb = os.path.basename(args.sheets[i]), os.path.basename(args.sheets[j])
            print(f"  Cohen  {na[:22]:24} vs {nb[:22]:24} n={n:4d}  raw={po:.3f}  "
                  f"kappa={'n/a' if k is None else round(k, 3)}")
        if len(sheets) >= 3:
            k, n, po = fleiss_kappa(sheets)
            print(f"  Fleiss over {len(sheets)} annotators              "
                  f"n={n:4d}  mean pairwise agreement={po:.3f}  "
                  f"kappa={'n/a' if k is None else round(k, 3)}")

        gold, tied = adjudicate(sheets)
        print(f"\n  majority label found for {len(gold)} rows; no majority for {len(tied)}"
              + (f" (ids {', '.join(tied[:12])}{' ...' if len(tied) > 12 else ''})" if tied else ""))
        if gold:
            score(gold, key, f"the majority label of {len(sheets)} annotators")

        common = set.intersection(*[set(s) for s in sheets])
        unan = {i: sheets[0][i] for i in common if len({s[i] for s in sheets}) == 1}
        print(f"\n  unanimous subset: {len(unan)} rows")
        if unan:
            score(unan, key, "the unanimous subset")

    for p, l in zip(args.sheets, sheets):
        score(l, key, os.path.basename(p))


if __name__ == "__main__":
    main()
