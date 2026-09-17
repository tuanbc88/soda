"""Compare two OIE_MISS diagnoses (measure-first: open-template re-run vs the 0622
primed-few-shot baseline). Prints a side-by-side table + the measure-first headline
(genuine-miss buckets C+D and OIE_MISS% before/after) and writes a self-contained report.

Background (DECISIONS 2026-06-26e/27): the OIE template closed-vocab leak into Mode-1 was
fixed (open extraction). The question: was the closed-vocab x fixed-few-shot combo an
OIE_MISS driver? If genuine extraction-miss (rollup_genuine_extraction_% = C_entity_miss +
D_pairing_miss) DROPS on the open re-run -> yes; if flat -> dynamic per-input few-shot
(KATE) is the next lever.

Each input is the diagnosis OUTPUT dir (the --output_dir you gave oie_miss_diagnosis.py), a
run dir, or the oie_miss_diagnosis.json itself; oie_metrics.json (entity recall) is picked
up automatically if present nearby.

Usage:
  python soda/evaluate/oie_miss_compare.py <open_dir> <baseline_dir> \
      [--label-a open] [--label-b 0622] [--out report.txt]
"""
import argparse
import json
import os


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _find(path, fname, subdirs):
    """Resolve `fname` from `path` which may be the file, its dir, or a run dir."""
    if path.endswith(".json") and os.path.isfile(path):
        return path
    cands = [os.path.join(path, fname)]
    for sd in subdirs:
        cands.append(os.path.join(path, sd, fname))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def _diag(path):
    p = _find(path, "oie_miss_diagnosis.json",
              ["eval_oie_miss", "iter0/eval_oie_miss"])
    return _load_json(p) if p else None


def _metrics(path):
    p = _find(path, "oie_metrics.json", ["eval_oie", "iter0/eval_oie"])
    return _load_json(p) if p else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="open re-run diagnosis dir/file (the NEW template)")
    ap.add_argument("b", help="baseline diagnosis dir/file (e.g. 0622 primed-few-shot)")
    ap.add_argument("--label-a", default="open")
    ap.add_argument("--label-b", default="0622")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    da, db = _diag(args.a), _diag(args.b)
    if da is None or db is None:
        raise SystemExit(f"[oie_miss_compare] diagnosis json not found "
                         f"(a={da is not None}, b={db is not None}); pass the oie_miss_diagnosis "
                         f"output dir / run dir / json.")
    oa, ob = da.get("overall", {}), db.get("overall", {})
    ma, mb = _metrics(args.a) or {}, _metrics(args.b) or {}

    la, lb = args.label_a, args.label_b
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    def row(label, va, vb, pct=False, flip_good="down"):
        """One comparison line. delta = a - b. flip_good marks which direction is 'better'."""
        if va is None or vb is None:
            emit(f"  {label:38s} {str(va):>10} {str(vb):>10} {'-':>10}")
            return
        d = va - vb
        arrow = ""
        if pct and abs(d) >= 0.01:
            better = (d < 0) if flip_good == "down" else (d > 0)
            arrow = "  (better)" if better else "  (worse)"
        ds = f"{d:+.2f}" if pct else f"{d:+d}" if isinstance(d, int) else f"{d:+.4f}"
        emit(f"  {label:38s} {va:>10} {vb:>10} {ds:>10}{arrow}")

    emit("=" * 78)
    emit(f"OIE_MISS DIAGNOSIS COMPARE   {la} (a)  vs  {lb} (b)   [delta = a - b]")
    emit(f"  a: {args.a}")
    emit(f"  b: {args.b}")
    emit("=" * 78)
    emit(f"  n_records a={oa.get('n_records')} b={ob.get('n_records')} | "
         f"n_gold a={oa.get('n_gold')} b={ob.get('n_gold')}")
    emit("")
    emit(f"  {'metric':38s} {la:>10} {lb:>10} {'delta':>10}")
    emit("  " + "-" * 72)

    # ---- headline ----
    emit("  [HEADLINE]")
    row("OIE_MISS %", oa.get("MISS_%"), ob.get("MISS_%"), pct=True, flip_good="down")
    row("CORRECT %", oa.get("CORRECT_%"), ob.get("CORRECT_%"), pct=True, flip_good="up")
    row("rollup GENUINE extraction % (C+D)", oa.get("rollup_genuine_extraction_%"),
        ob.get("rollup_genuine_extraction_%"), pct=True, flip_good="down")
    row("rollup surface/format % (B+F)", oa.get("rollup_surface_or_format_%"),
        ob.get("rollup_surface_or_format_%"), pct=True, flip_good="down")
    row("rollup ceiling/linking % (A+E)", oa.get("rollup_ceiling_or_linking_%"),
        ob.get("rollup_ceiling_or_linking_%"), pct=True, flip_good="down")

    # ---- per-bucket ----
    emit("")
    emit("  [BUCKETS]  (% of gold)")
    cats = ["A_unrealized", "B_surface", "C_entity_miss", "D_pairing_miss",
            "E_coref", "F_literal_format"]
    for c in cats:
        row(f"{c} %", oa.get(f"{c}_%"), ob.get(f"{c}_%"), pct=True, flip_good="down")

    # ---- implicit / direction ----
    emit("")
    emit("  [IMPLICIT / DIRECTION]")
    row("relation_implicit % of gold", oa.get("relation_implicit_%"),
        ob.get("relation_implicit_%"), pct=True, flip_good="down")
    row("rel_implicit missed % of miss", oa.get("relation_implicit_missed_%_of_miss"),
        ob.get("relation_implicit_missed_%_of_miss"), pct=True, flip_good="down")
    row("direction_swapped %", oa.get("direction_swapped_%"),
        ob.get("direction_swapped_%"), pct=True, flip_good="down")

    # ---- entity recall (if oie_metrics.json present) ----
    if ma or mb:
        emit("")
        emit("  [ENTITY RECALL]  (oie_metrics.json)")
        row("entity_recall", ma.get("entity_recall"), mb.get("entity_recall"),
            pct=False, flip_good="up")
        row("entities_missed", ma.get("entities_missed"), mb.get("entities_missed"))
        row("entities_gold_total", ma.get("entities_gold_total"), mb.get("entities_gold_total"))
        row("entity_recall_per_item_median", ma.get("entity_recall_per_item_median"),
            mb.get("entity_recall_per_item_median"), pct=False, flip_good="up")
        row("frac items recall<0.5", ma.get("frac_items_entity_recall_below_0.5"),
            mb.get("frac_items_entity_recall_below_0.5"), pct=False, flip_good="down")
    else:
        emit("")
        emit("  [ENTITY RECALL] oie_metrics.json not found near either input "
             "(run oie_metric.py if you want it).")

    # ---- verdict ----
    emit("")
    emit("=" * 78)
    ga, gb = oa.get("rollup_genuine_extraction_%"), ob.get("rollup_genuine_extraction_%")
    moa, mob = oa.get("MISS_%"), ob.get("MISS_%")
    if ga is not None and gb is not None:
        dg = ga - gb
        dm = (moa - mob) if (moa is not None and mob is not None) else None
        if dg <= -0.5:
            emit(f"VERDICT: genuine extraction-miss DROPPED {dg:+.2f} pts (OIE_MISS {dm:+.2f}) ->"
                 f" the closed-vocab x fixed-few-shot WAS an OIE_MISS driver; open template helps.")
        elif dg >= 0.5:
            emit(f"VERDICT: genuine extraction-miss ROSE {dg:+.2f} pts -> open template did not help OIE"
                 f" (investigate; maybe the primed few-shot was actually aiding format).")
        else:
            emit(f"VERDICT: genuine extraction-miss ~flat ({dg:+.2f} pts) -> closed-vocab was NOT the"
                 f" main OIE_MISS driver; dynamic per-input few-shot (KATE) is the next lever.")
    emit("=" * 78)

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            print(f"\n[report written: {args.out}]")
        except Exception as e:
            print(f"[could not write {args.out}: {e}]")


if __name__ == "__main__":
    main()
