"""Propose - never decide - a synonym map for the Edu-KG entity-type annotation.

Three annotators labelled 200 blind mentions with an OPEN label set, and reached Fleiss
kappa 0.234 at 27.8% raw agreement. Inspection of the disagreements shows most of them are
two names for one concept (a repeated typo, a diacritics variant, hoc phan vs mon hoc)
rather than two readings of the mention. Collapsing those is a legitimate, documented
adjudication step; collapsing a genuine distinction is not, so this script does not decide.

It emits every pair of labels that COLLIDE - two annotators labelling the SAME mention
differently - because only colliding pairs can move kappa. Each pair carries its evidence and
an empty DECISION column for the author. Tiers order the review, they do not authorise a merge:

  T1_typo      near-identical once diacritics and case are stripped (edit distance <= 2).
               Example: TenGiengAnhMonHoc / TenTiengAnhMonHoc, a one-character slip repeated.
  T2_morph     shares at least half its CamelCase tokens. Example: MaHocPhan / MaMonHoc.
               Usually the same concept under two Vietnamese words, but read each one.
  T3_semantic  collides repeatedly with no string evidence. Example: CongTy / DoanhNghiep.
               These need a human: some are synonyms, some are a real granularity choice.
  T4_weak      collides once, no string evidence. Most of these are ordinary disagreement.

Usage:
  python scripts/build_synonym_candidates.py --sheets A.csv B.csv C.csv --out candidates.csv
"""
import argparse, csv, re, sys, unicodedata
from collections import Counter, defaultdict
from itertools import combinations


def deaccent(s):
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def key(s):
    return re.sub(r"[^a-z0-9]", "", deaccent(s).lower())


def tokens(label):
    """Split CamelCase / snake_case into lowercase morphemes."""
    s = deaccent(label or "")
    s = re.sub(r"[_\-/]+", " ", s)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    return {t.lower() for t in s.split() if t}


def edit_distance(a, b, cap=3):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load(path):
    rows = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            lab = (r.get("gold_type") or "").strip()
            if lab:
                rows[(r.get("id") or "").strip()] = (lab, (r.get("mention") or "").strip())
    return rows


def classify(l1, l2):
    k1, k2 = key(l1), key(l2)
    if k1 == k2:
        return "T1_typo", "identical once diacritics/case are stripped"
    d = edit_distance(k1, k2)
    if d <= 2:
        return "T1_typo", f"edit distance {d} after stripping diacritics"
    t1, t2 = tokens(l1), tokens(l2)
    if t1 and t2:
        inter, union = t1 & t2, t1 | t2
        if inter and len(inter) / len(union) >= 0.5:
            return "T2_morph", "shares tokens {%s}" % ", ".join(sorted(inter))
    return None, "no string evidence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", nargs="+", required=True)
    ap.add_argument("--out", default="synonym_candidates.csv")
    ap.add_argument("--min-collisions", type=int, default=1)
    args = ap.parse_args()

    sheets = [load(p) for p in args.sheets]
    names = [p.replace("\\", "/").rsplit("/", 1)[-1] for p in args.sheets]

    freq = Counter()
    for s in sheets:
        for lab, _ in s.values():
            freq[lab] += 1

    collide = Counter()
    examples = defaultdict(list)
    for rid in set().union(*[set(s) for s in sheets]):
        labs = {}
        for nm, s in zip(names, sheets):
            if rid in s:
                labs[nm] = s[rid]
        for (n1, (l1, m1)), (n2, (l2, _)) in combinations(sorted(labs.items()), 2):
            if key(l1) != key(l2) or l1 != l2:
                if l1 == l2:
                    continue
                pair = tuple(sorted((l1, l2)))
                collide[pair] += 1
                if len(examples[pair]) < 3:
                    examples[pair].append(f"#{rid} {m1[:40]}")

    rows = []
    for (l1, l2), n in collide.items():
        tier, why = classify(l1, l2)
        if tier is None:
            tier = "T3_semantic" if n >= 2 else "T4_weak"
        canonical = l1 if freq[l1] >= freq[l2] else l2
        rows.append({
            "tier": tier, "collisions": n,
            "label_1": l1, "n_1": freq[l1], "label_2": l2, "n_2": freq[l2],
            "suggested_canonical": canonical, "evidence": why,
            "examples": " | ".join(examples[(l1, l2)]),
            "DECISION_merge_or_keep": "", "NOTE": "",
        })

    order = {"T1_typo": 0, "T2_morph": 1, "T3_semantic": 2, "T4_weak": 3}
    rows.sort(key=lambda r: (order[r["tier"]], -r["collisions"], r["label_1"]))
    rows = [r for r in rows if r["collisions"] >= args.min_collisions]

    with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tally = Counter(r["tier"] for r in rows)
    lost = sum(r["collisions"] for r in rows)
    print(f"distinct labels across {len(sheets)} sheets: {len(freq)}")
    print(f"colliding pairs: {len(rows)}  covering {lost} disagreeing row-pairs")
    for t in ("T1_typo", "T2_morph", "T3_semantic", "T4_weak"):
        n_pairs = tally.get(t, 0)
        n_rows = sum(r["collisions"] for r in rows if r["tier"] == t)
        print(f"  {t:12s} pairs={n_pairs:4d}  row-pairs={n_rows:4d}")
    print(f"\nwritten: {args.out}")
    print("Fill DECISION_merge_or_keep with merge / keep. Nothing is merged until you do.")


if __name__ == "__main__":
    sys.exit(main())
