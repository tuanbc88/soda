"""Apply the author's reviewed synonym decisions to the annotation sheets.

The decisions in synonym_candidates_response.xlsx are PAIRWISE: "where one annotator wrote
label X and another wrote label Y on the same mention, those two mean the same thing here."
They are not statements that X and Y are globally interchangeable, and the difference is not
cosmetic. Taking the 31 approved pairs as a global equivalence relation makes `Thing` a bridge
between every specific type it was ever paired with, and the transitive closure fuses 25 labels
into one class containing SinhVien, TruongDaiHoc, QuyDinh and MonHoc together. That is the
prevalence collapse that drove Fleiss kappa to 0.031 in the all-merge simulation: agreement
looks excellent because almost every label has become the same label.

So merges are applied ROW-LOCALLY. On each mention, if two annotators' labels form an approved
pair, both are rewritten to that pair's canonical, iterated to a fixpoint within that row only.
Every approved decision is honoured exactly where it was made, and no equivalence leaks into
rows the reviewer never looked at.

Usage:
  python scripts/apply_synonym_decisions.py --decisions response.xlsx \
      --sheets A.csv B.csv C.csv --outdir reconciled/
"""
import argparse, csv, os, sys


def load_decisions(path):
    """Return {(label_a, label_b): canonical} for approved merges, keys order-insensitive."""
    import openpyxl
    ws = openpyxl.load_workbook(path).active
    rows = list(ws.values)
    hdr = list(rows[0])
    ix = {k: hdr.index(k) for k in
          ("label_1", "label_2", "suggested_canonical", "DECISION_merge_or_keep")}
    out, kept = {}, 0
    for r in rows[1:]:
        dec = r[ix["DECISION_merge_or_keep"]]
        dec = dec.strip().lower() if isinstance(dec, str) else ""
        l1, l2 = (r[ix["label_1"]] or "").strip(), (r[ix["label_2"]] or "").strip()
        if dec != "merge":
            kept += 1
            continue
        canon = (r[ix["suggested_canonical"]] or "").strip() or l1
        out[frozenset((l1, l2))] = canon
    return out, kept


def load_sheet(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def reconcile(sheets, merges):
    """Rewrite labels row by row. Returns (new_sheets, n_rows_touched, n_rewrites)."""
    by_id = [{(r.get("id") or "").strip(): r for r in s} for s in sheets]
    ids = set().union(*[set(d) for d in by_id])
    touched = rewrites = 0
    for rid in ids:
        present = [d[rid] for d in by_id if rid in d]
        labelled = [r for r in present if (r.get("gold_type") or "").strip()]
        if len(labelled) < 2:
            continue
        changed_here = False
        for _ in range(len(labelled)):          # fixpoint; at most one pass per label
            hit = False
            for i in range(len(labelled)):
                for j in range(i + 1, len(labelled)):
                    a = labelled[i]["gold_type"].strip()
                    b = labelled[j]["gold_type"].strip()
                    if a == b:
                        continue
                    canon = merges.get(frozenset((a, b)))
                    if canon:
                        for r in (labelled[i], labelled[j]):
                            if r["gold_type"].strip() != canon:
                                r["gold_type"] = canon
                                rewrites += 1
                        hit = changed_here = True
            if not hit:
                break
        touched += bool(changed_here)
    return sheets, touched, rewrites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--sheets", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    merges, kept = load_decisions(args.decisions)
    print(f"approved merges: {len(merges)}   kept apart: {kept}")

    sheets = [load_sheet(p) for p in args.sheets]
    sheets, touched, rewrites = reconcile(sheets, merges)
    print(f"rows reconciled: {touched}   individual labels rewritten: {rewrites}")

    os.makedirs(args.outdir, exist_ok=True)
    for src, rows in zip(args.sheets, sheets):
        name = os.path.basename(src).replace(".csv", "") + "__reconciled.csv"
        dst = os.path.join(args.outdir, name)
        with open(dst, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("  ->", dst)


if __name__ == "__main__":
    sys.exit(main())
