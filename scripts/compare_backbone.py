#!/usr/bin/env python3
"""
Compare a swapped BACKBONE against the Qwen3-8B headline at the identical config
(Mode-3 / item / bf16 / bge-m3 / 0627open), on the three gold benchmarks, and print the
row that goes into the paper's tab:edc plus the delta vs EDC-no-refine.

Only valid when the two runs differ ONLY in the backbone -- that is what
run_backbone_headline.sh sets up. (The T4 size-sweep runs are 4-bit + minilm and are NOT
headline-comparable; don't point this at them.)

Usage
  python scripts/compare_backbone.py                       # qwen2.5-7b vs qwen3-8b headline
  python scripts/compare_backbone.py --backbone sailor2-8b
  python scripts/compare_backbone.py --out output/backbone_vs_headline.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

# EDC published baselines (Target Alignment = Mode 3), from COMPARISON_vs_EDC_paper.md
EDC = {
    "webnlg":   {"no_refine": (0.746, 0.688, 0.713), "refine": (0.794, 0.753, 0.772), "G": 159},
    "rebel":    {"no_refine": (0.506, 0.449, 0.473), "refine": (0.559, 0.516, 0.529), "G": 196},
    "wiki-nre": {"no_refine": (0.647, 0.638, 0.640), "refine": (0.693, 0.685, 0.657), "G": 45},
}
PATTERN = "output/{ds}_selfcanon2_mode3_item_{tag}_bgem3_{gpu}_{date}/iter0"


def read_t1(d):
    p = os.path.join(d, "eval", "table1_closed_schema.csv")
    if not os.path.isfile(p):
        return {}
    out = {}
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        try:
            out[r["case"]] = (float(r["partial"]), float(r["strict"]), float(r["exact"]))
        except (KeyError, ValueError):
            pass
    return out


def read_b3(d):
    p = os.path.join(d, "eval_clustering", "clustering_metrics.csv")
    if not os.path.isfile(p):
        return {}
    out = {}
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        try:
            out[r["case"]] = float(r["bcubed_f1"])
        except (KeyError, ValueError):
            pass
    return out


def best_llm(t1, key=1):
    """best LLM case (phi>=2, i.e. exclude case1 no-LLM threshold) by strict (key=1)."""
    cands = {c: v for c, v in t1.items() if not c.startswith("case1_")}
    if not cands:
        return None, None
    c = max(cands, key=lambda c: cands[c][key])
    return c, cands[c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="qwen2.5-7b", help="MODEL_TAG of the swapped backbone")
    ap.add_argument("--headline", default="qwen3-8b", help="MODEL_TAG of the headline backbone")
    ap.add_argument("--gpu", default="A100")
    ap.add_argument("--date", default="0627open")
    ap.add_argument("--datasets", nargs="+", default=["webnlg", "rebel", "wiki-nre"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    print("=" * 92)
    print(f"Backbone swap: {args.backbone}  vs  headline {args.headline}   "
          f"(Mode-3 / item / bf16 / bge-m3 / {args.date})")
    print("=" * 92)
    any_found = False
    for ds in args.datasets:
        dn = PATTERN.format(ds=ds, tag=args.backbone, gpu=args.gpu, date=args.date)
        dh = PATTERN.format(ds=ds, tag=args.headline, gpu=args.gpu, date=args.date)
        tn, th = read_t1(dn), read_t1(dh)
        bn, bh = read_b3(dn), read_b3(dh)
        print(f"\n--- {ds} (G={EDC[ds]['G']}) ---")
        if not tn:
            print(f"  [PENDING] backbone run not found: {dn}")
            continue
        any_found = True
        cn, vn = best_llm(tn)
        edc_p, edc_s, edc_e = EDC[ds]["no_refine"]
        print(f"  {args.backbone:12s} best {cn:24s} P {vn[0]:.3f}  S {vn[1]:.3f}  E {vn[2]:.3f}"
              f"   B3 {bn.get(cn, float('nan')):.3f}")
        if th:
            ch, vh = best_llm(th)
            print(f"  {args.headline:12s} best {ch:24s} P {vh[0]:.3f}  S {vh[1]:.3f}  E {vh[2]:.3f}"
                  f"   B3 {bh.get(ch, float('nan')):.3f}")
            print(f"  {'delta vs headline':37s} P {vn[0]-vh[0]:+.3f}  S {vn[1]-vh[1]:+.3f}  E {vn[2]-vh[2]:+.3f}")
        else:
            print(f"  [headline run missing: {dh}]")
            ch, vh = None, None
        print(f"  {'EDC no-refine':12s} {'':29s} P {edc_p:.3f}  S {edc_s:.3f}  E {edc_e:.3f}")
        d_edc = vn[1] - edc_s
        verdict = "BEATS EDC-no-refine" if d_edc > 0 else "below EDC-no-refine"
        print(f"  -> {args.backbone} strict {vn[1]:.3f} vs EDC {edc_s:.3f} = {d_edc:+.3f}  ** {verdict} **")
        rows.append({
            "dataset": ds, "backbone": args.backbone, "best_case": cn,
            "partial": round(vn[0], 4), "strict": round(vn[1], 4), "exact": round(vn[2], 4),
            "bcubed_f1": round(bn.get(cn, float("nan")), 4),
            "headline_strict": round(vh[1], 4) if vh else "",
            "delta_vs_headline": round(vn[1] - vh[1], 4) if vh else "",
            "edc_norefine_strict": edc_s, "delta_vs_edc": round(d_edc, 4),
        })

    if not any_found:
        print("\nNo backbone runs found yet — run `bash run_backbone_headline.sh` first.")
        return 1

    print("\n" + "=" * 92)
    print("tab:edc row to paste (SODA with the swapped backbone, no-refine):")
    print("=" * 92)
    for r in rows:
        phi = r["best_case"].split("_")[0].replace("case", "")
        print(f"  {r['dataset']:9s}  SODA ({args.backbone}, no-refine) & $\\phi{{=}}{phi}$ "
              f"& {r['partial']:.3f} & {r['strict']:.3f} & {r['exact']:.3f} \\\\"
              f"   % vs EDC {r['delta_vs_edc']:+.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
