"""Compare two SODA runs (SC batching OFF vs ON) and write a self-contained report.

Used by run_sc_batch_validate.sh and re-runnable locally on copied-back output. It
answers, in order:

  1. Are the SC INPUTS identical?  (oie_total.json + sd_total.json) - if not, the two
     runs diverged upstream (OIE/SD GPU nondeterminism), so any SC diff is confounded;
     re-run the ON side with --resume_from sc reusing the OFF base to isolate SC.
  2. Is the FINAL per-case canon KG identical?  (canon_kg_{case}.json/.txt)
  3. Where (if anywhere) do per-triplet SC decisions differ?  (from sc_mcp_logs.jsonl):
     for every (idx, case, triplet) it compares the raw LLM output AND the parsed
     pred_relation - distinguishing harmless raw-text fp jitter (same pred) from a real
     decision flip (different pred). This is the decisive greedy-reproduce check.
  4. SC wall-clock + token totals OFF vs ON (the speedup).

Usage:  python scripts/sc_batch_compare.py <off_dir> <on_dir> [--out report.txt]
        (each dir is a run root containing iter0/)
"""
import argparse
import json
import os
import sys


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_jsonl(path):
    recs = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    except Exception:
        return None
    return recs


def _iter(d):
    return os.path.join(d, "iter0")


def _discover_cases(off_iter):
    cases = []
    for fn in sorted(os.listdir(off_iter)):
        if fn.startswith("canon_kg_") and fn.endswith(".json") and "with_entity" not in fn:
            cases.append(fn[len("canon_kg_"):-len(".json")])
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("off_dir")
    ap.add_argument("on_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    off_i, on_i = _iter(args.off_dir), _iter(args.on_dir)
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("=" * 70)
    emit("SC-BATCHING OFF-vs-ON COMPARISON")
    emit(f"  OFF: {args.off_dir}")
    emit(f"  ON : {args.on_dir}")
    emit("=" * 70)

    # ---- 1. upstream identity (SC inputs) ----
    emit("\n[1] SC INPUTS (must match for a clean SC comparison)")
    upstream_ok = True
    for fn in ("oie_total.json", "sd_total.json"):
        a, b = _load_json(os.path.join(off_i, fn)), _load_json(os.path.join(on_i, fn))
        if a is None or b is None:
            emit(f"    {fn:16s}: MISSING on one side")
            upstream_ok = False
        elif a == b:
            emit(f"    {fn:16s}:  identical [ok]")
        else:
            emit(f"    {fn:16s}: DIFFER [DIFF]  (upstream nondeterminism - SC diff confounded)")
            upstream_ok = False
    if not upstream_ok:
        emit("    !! OIE/SD differ between runs -> re-run ON with --resume_from sc reusing the")
        emit("       OFF base to isolate SC batching (otherwise the SC diff mixes two causes).")

    # ---- 2. final per-case canon KG ----
    if not os.path.isdir(off_i):
        emit(f"\n[ERROR] {off_i} not found"); _write(args, lines); sys.exit(2)
    cases = _discover_cases(off_i)
    emit(f"\n[2] FINAL canon KG per case ({len(cases)} cases)")
    kg_diff_cases = []
    for case in cases:
        a = _load_json(os.path.join(off_i, f"canon_kg_{case}.json"))
        b = _load_json(os.path.join(on_i, f"canon_kg_{case}.json"))
        same = (a == b)
        if not same:
            kg_diff_cases.append(case)
        emit(f"    {case:28s}: {'identical [ok]' if same else 'DIFFER [DIFF]'}")

    # ---- 3. per-triplet decision + raw-output (the decisive check) ----
    emit("\n[3] PER-TRIPLET SC decisions (from sc_mcp_logs.jsonl)")
    off_log = _load_jsonl(os.path.join(off_i, "sc_mcp_logs.jsonl"))
    on_log = _load_jsonl(os.path.join(on_i, "sc_mcp_logs.jsonl"))
    n_pred_diff = None
    flip_cases = set()
    if off_log is None or on_log is None:
        emit("    sc_mcp_logs.jsonl missing on one side (set EDC_DEBUG_LOGS=1). Skipping.")
    else:
        def index(log):
            d = {}
            for rec in log:
                key = (rec.get("idx"), rec.get("case"))
                d[key] = rec.get("triplets", [])
            return d
        oi, ni = index(off_log), index(on_log)
        keys = sorted(set(oi) | set(ni), key=lambda k: (k[0] if k[0] is not None else -1, str(k[1])))
        n_trip = n_raw_diff = n_pred_diff = 0
        pred_flips = []
        raw_only = []
        for key in keys:
            ot, nt = oi.get(key, []), ni.get(key, [])
            m = {t.get("t_idx"): t for t in ot}
            for t in nt:
                ti = t.get("t_idx")
                o = m.get(ti)
                if o is None:
                    continue
                n_trip += 1
                raw_diff = (o.get("raw_output") != t.get("raw_output"))
                pred_diff = (o.get("pred_relation") != t.get("pred_relation"))
                if raw_diff:
                    n_raw_diff += 1
                if pred_diff:
                    n_pred_diff += 1
                    flip_cases.add(key[1])
                    pred_flips.append((key, ti, o, t))
                elif raw_diff:
                    raw_only.append((key, ti, o, t))
        emit(f"    triplet-case decisions compared : {n_trip}")
        emit(f"    raw LLM output differs           : {n_raw_diff}")
        emit(f"    pred_relation FLIPS (real diff)  : {n_pred_diff}")
        emit(f"    raw differs but pred SAME (fp ok): {len(raw_only)}")

        if pred_flips:
            emit("\n    --- pred_relation flips (idx, case, t_idx) ---")
            for (idx, case), ti, o, t in pred_flips[:40]:
                emit(f"    idx={idx} case={case} t_idx={ti} triplet={o.get('input_triplet')}")
                emit(f"        OFF pred={o.get('pred_relation')!r} dec={o.get('decision')!r} "
                     f"raw={str(o.get('raw_output'))[:80]!r}")
                emit(f"        ON  pred={t.get('pred_relation')!r} dec={t.get('decision')!r} "
                     f"raw={str(t.get('raw_output'))[:80]!r}")
            if len(pred_flips) > 40:
                emit(f"    ... +{len(pred_flips) - 40} more flips")
        if raw_only:
            emit(f"\n    --- {len(raw_only)} triplet(s): raw text differs but decision unchanged "
                 f"(harmless fp jitter); first 5 ---")
            for (idx, case), ti, o, t in raw_only[:5]:
                emit(f"    idx={idx} case={case} t_idx={ti}: OFF raw={str(o.get('raw_output'))[:50]!r}"
                     f" | ON raw={str(t.get('raw_output'))[:50]!r}")

    # ---- 4. timing + tokens ----
    emit("\n[4] SC wall-clock + tokens")
    ot = _load_json(os.path.join(off_i, "stage_timing.json")) or {}
    nt = _load_json(os.path.join(on_i, "stage_timing.json")) or {}
    osec, nsec = ot.get("sc_sec"), nt.get("sc_sec")
    emit(f"    sc_sec     : OFF={osec}  ON={nsec}"
         + (f"  speedup={osec / nsec:.2f}x" if (osec and nsec) else ""))
    emit(f"    sc_tokens  : OFF={ot.get('sc_tokens')}  ON={nt.get('sc_tokens')}")

    # ---- verdict ----
    emit("\n" + "=" * 70)
    kg_ok = (len(kg_diff_cases) == 0)
    if n_pred_diff is not None:
        if kg_ok and n_pred_diff == 0:
            emit("VERDICT: PASS - batched == off (no pred flips; final KG identical).")
        elif n_pred_diff == 0 and not kg_ok:
            emit(f"VERDICT: CHECK - no per-triplet pred flips but KG json differs on "
                 f"{kg_diff_cases} (likely schema ordering/aliases; inspect).")
        else:
            emit(f"VERDICT: CHECK - {n_pred_diff} pred flip(s) on cases {sorted(flip_cases)}; "
                 f"KG differs on {kg_diff_cases}. Few + scattered = fp; many/systematic = report.")
    else:
        emit(f"VERDICT: PARTIAL - KG differ cases={kg_diff_cases}; per-triplet trace unavailable "
             f"(set EDC_DEBUG_LOGS=1).")
    emit("=" * 70)

    _write(args, lines)


def _write(args, lines):
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            print(f"\n[report written: {args.out}]")
        except Exception as e:
            print(f"[could not write {args.out}: {e}]")


if __name__ == "__main__":
    main()
