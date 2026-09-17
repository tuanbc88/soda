#!/usr/bin/env python3
"""
T7 -- aggregate the >=3-seed variance runs into mean +- std, and decide whether the
headline claims (the canon-signal FLIP, the Mode3-Mode1 schema-policy gap) survive
the error bars.

Contract (RUN_T7_SEEDS.md):
  dirs    output/webnlg_selfcanon2_mode{1,3}_item_qwen3-8b_bgem3_A100_0627open_sdt0.3_seed{1,2,3}/iter0
  seeds   SD sampled at SD_TEMPERATURE=0.3; OIE+SC greedy; OIE reused across seeds
  rule    the FLIP / gap is CONFIRMED only if the winner's interval clears the runner-up's.

Metrics read per (mode, seed):
  Mode 1  aligned_triple_f1  eval_mode1/mode1_metrics.csv     (name-drift robust; the M1 axis)
          bcubed_f1          eval_clustering/clustering_metrics.csv
          strict             eval/table1_closed_schema.csv
  Mode 3  strict/partial/exact  eval/table1_closed_schema.csv
          bcubed_f1          eval_clustering/clustering_metrics.csv

std = population stdev (statistics.pstdev), matching the RUN_T7_SEEDS.md snippet.
Winner analysis uses LLM cases only (phi>=2); case1 (no-LLM cosine threshold) is
reported separately as the no-canonicalization reference (it wins B-cubed by
under-merging, so including it would be misleading).

Usage
  # after all 3 seeds land
  python scripts/aggregate_t7_seeds.py
  # partial (e.g. only seeds 1-2 done so far)
  python scripts/aggregate_t7_seeds.py --seeds 1 2
  # validate the parser against the existing single greedy run (std will be 0)
  python scripts/aggregate_t7_seeds.py --self-test
  # write a CSV to paste into the paper
  python scripts/aggregate_t7_seeds.py --out output/t7_seed_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
from collections import defaultdict

DEFAULT_PATTERN = (
    "output/webnlg_selfcanon2_mode{mode}_item_qwen3-8b_bgem3_A100_0627open"
    "_sdt0.3_seed{seed}/iter0"
)
# for --self-test: the committed greedy runs (no seed suffix); {seed} is ignored
GREEDY_PATTERN = "output/webnlg_selfcanon2_mode{mode}_item_qwen3-8b_bgem3_A100_0627open/iter0"

# metric key -> (relative csv path, case-column, value-column, modes it applies to)
METRICS = {
    "aligned_f1": ("eval_mode1/mode1_metrics.csv", "case", "aligned_triple_f1", {1}),
    "bcubed_f1": ("eval_clustering/clustering_metrics.csv", "case", "bcubed_f1", {1, 2, 3}),
    "strict": ("eval/table1_closed_schema.csv", "case", "strict", {1, 2, 3}),
    "partial": ("eval/table1_closed_schema.csv", "case", "partial", {3}),
    "exact": ("eval/table1_closed_schema.csv", "case", "exact", {3}),
}
# which metric is the headline axis per mode (used for the winner verdict)
HEADLINE = {1: ["bcubed_f1", "aligned_f1"], 3: ["bcubed_f1", "strict"]}


def read_metric(iter_dir, rel_path, case_col, val_col):
    """-> {case: float}; returns {} if the file is absent."""
    path = os.path.join(iter_dir, rel_path)
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            case = (row.get(case_col) or "").strip()
            raw = (row.get(val_col) or "").strip()
            if not case or not raw:
                continue
            try:
                out[case] = float(raw)
            except ValueError:
                pass
    return out


def collect(pattern, modes, seeds):
    """-> data[mode][metric][case] = [v_seed1, ...], plus a report of what was found.

    `pattern` may be a single template or a LIST of templates, tried in order until one
    resolves to an existing dir. That is needed whenever the legs of one comparison were
    produced on different boxes, since GPU_TAG is part of the directory name: the REBEL
    Mode-3 seeds ran on the A100 while Mode-1 ran on the H100 (2026-07-27b). Passing both
    templates keeps them in a single aggregation instead of forcing a rename that would
    falsify the recorded provenance.
    """
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    found, missing = [], []
    for mode in modes:
        for seed in seeds:
            cands = [p.format(mode=mode, seed=seed) for p in patterns]
            d = next((c for c in cands if os.path.isdir(c)), None)
            if d is None:
                missing.append((mode, seed, " | ".join(cands)))
                continue
            found.append((mode, seed, d))
            for mkey, (rel, ccol, vcol, mset) in METRICS.items():
                if mode not in mset:
                    continue
                for case, val in read_metric(d, rel, ccol, vcol).items():
                    data[mode][mkey][case].append(val)
    return data, found, missing


def agg(vals):
    n = len(vals)
    if n == 0:
        return None
    return {
        "n": n,
        "mean": st.mean(vals),
        "std": st.pstdev(vals) if n > 1 else 0.0,
        "vals": vals,
    }


def is_llm_case(case):
    """case1 = the no-LLM cosine threshold reference; exclude from winner analysis."""
    return not case.startswith("case1_")


def verdict(ranked):
    """ranked: [(case, stats), ...] desc by mean, LLM cases only.
    CONFIRMED iff winner's [mean-std] > runner-up's [mean+std] (disjoint intervals)."""
    if len(ranked) < 2:
        return "n/a (need >=2 cases)", None
    (w, ws), (r, rs) = ranked[0], ranked[1]
    lo_w = ws["mean"] - ws["std"]
    hi_r = rs["mean"] + rs["std"]
    margin = lo_w - hi_r
    if ws["n"] < 2:
        return f"UNDECIDED (single seed; no error bar) -- winner {w}", margin
    if margin > 0:
        return f"CONFIRMED: {w} clears {r} by {margin:+.4f}", margin
    return (
        f"NOT SEPARATED: {w} vs {r} intervals overlap by {-margin:.4f} "
        f"(need more seeds or the difference is noise)"
    ), margin


def fmt(s):
    return f"{s['mean']:.4f} +- {s['std']:.4f}" if s else "--"


def main():
    ap = argparse.ArgumentParser(description="Aggregate T7 seed runs into mean +- std.")
    ap.add_argument("--pattern", default=[DEFAULT_PATTERN], nargs="+",
                    help="dir template(s) with {mode} and {seed}; several are tried in order "
                         "until one exists, for legs run on different boxes (GPU_TAG differs). "
                         "Default: T7 webnlg 8B sdt0.3")
    ap.add_argument("--modes", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default=None, help="write a tidy CSV here")
    ap.add_argument("--self-test", action="store_true",
                    help="parse the existing greedy 0627open runs as a 1-seed sanity check")
    args = ap.parse_args()

    pattern, seeds = args.pattern, args.seeds
    if args.self_test:
        pattern, seeds = GREEDY_PATTERN, [1]
        print("[self-test] parsing the greedy 0627open runs as a single pseudo-seed "
              "(std must be 0.0000 and means must match the paper tables)\n")

    data, found, missing = collect(pattern, args.modes, seeds)

    print("=" * 78)
    print("T7 seed aggregation")
    print("=" * 78)
    for mode, seed, d in found:
        print(f"  [found]   mode {mode} seed {seed}  {d}")
    for mode, seed, d in missing:
        print(f"  [MISSING] mode {mode} seed {seed}  {d}")
    if not found:
        print("\nNo seed dirs found. Are the runs downloaded to local output/ yet?")
        return 1
    n_by_mode = {m: len({s for (mm, s, _) in found if mm == m}) for m in args.modes}
    print(f"\nseeds present per mode: {n_by_mode}")
    if any(n < 3 for n in n_by_mode.values()):
        print("WARNING: <3 seeds -> error bars are provisional; T7 needs >=3 to gate a headline.")

    rows = []
    for mode in args.modes:
        if mode not in data:
            continue
        for mkey in METRICS:
            if mode not in METRICS[mkey][3] or mkey not in data[mode]:
                continue
            stats = {c: agg(v) for c, v in data[mode][mkey].items()}
            stats = {c: s for c, s in stats.items() if s}
            if not stats:
                continue
            print("\n" + "-" * 78)
            print(f"Mode {mode} | {mkey}")
            print("-" * 78)
            for case in sorted(stats, key=lambda c: -stats[c]["mean"]):
                s = stats[case]
                tag = "" if is_llm_case(case) else "   (no-LLM ref)"
                print(f"  {case:28s} {fmt(s):>20s}   n={s['n']}{tag}")
                rows.append({
                    "mode": mode, "metric": mkey, "case": case,
                    "mean": round(s["mean"], 4), "std": round(s["std"], 4),
                    "n_seeds": s["n"],
                    "values": " ".join(f"{v:.4f}" for v in s["vals"]),
                })
            llm_ranked = sorted(
                [(c, s) for c, s in stats.items() if is_llm_case(c)],
                key=lambda kv: -kv[1]["mean"],
            )
            v, _ = verdict(llm_ranked)
            star = " *headline axis*" if mkey in HEADLINE.get(mode, []) else ""
            print(f"  -> winner verdict{star}: {v}")

    # Mode3 - Mode1 schema-policy gap, with propagated std, on the shared metrics
    for mkey in ("bcubed_f1", "strict"):
        if 1 in data and 3 in data and mkey in data[1] and mkey in data[3]:
            def best(mode):
                stats = {c: agg(v) for c, v in data[mode][mkey].items() if is_llm_case(c)}
                stats = {c: s for c, s in stats.items() if s}
                if not stats:
                    return None, None
                c = max(stats, key=lambda c: stats[c]["mean"])
                return c, stats[c]
            c1, s1 = best(1)
            c3, s3 = best(3)
            if s1 and s3:
                gap = s3["mean"] - s1["mean"]
                gstd = (s1["std"] ** 2 + s3["std"] ** 2) ** 0.5
                print("\n" + "-" * 78)
                print(f"Schema-policy gap (Mode3 - Mode1) | {mkey}")
                print("-" * 78)
                print(f"  M1 best {c1:26s} {fmt(s1)}")
                print(f"  M3 best {c3:26s} {fmt(s3)}")
                print(f"  gap = {gap:+.4f} +- {gstd:.4f} (std propagated in quadrature)")
                print(f"  -> {'REAL (gap exceeds its error bar)' if gap - gstd > 0 else 'NOT SEPARATED from 0'}")
                rows.append({
                    "mode": "M3-M1", "metric": mkey, "case": f"{c3} - {c1}",
                    "mean": round(gap, 4), "std": round(gstd, 4),
                    "n_seeds": min(s1["n"], s3["n"]), "values": "",
                })

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["mode", "metric", "case", "mean", "std",
                                               "n_seeds", "values"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
