"""Prefix-slice cached stage totals (oie_total.json / sd_total.json) for the scale study.

The webnlg_full_{1k,3k,6k,full} subsets are NESTED prefixes (build_full_dataset.py: same
seed-42 shuffle, smaller = prefix of larger), and OIE + SD are per-item, greedy-deterministic.
So a size-N run's OIE/SD is byte-identical to the first N entries of the `full` run's totals.
This lets the scale study compute OIE+SD ONCE on `full` and reuse a prefix for the smaller sizes.
Saves the redundant OIE+SD (SD ~= 69% of wall-clock).

★★ CORRECTION 2026-07-17 — this docstring used to end with:
      "(only SC must re-run per size, because the Mode-1 schema GROWS with the corpus = the §5b signal)"
   **That was WRONG**, and it cost the 4B curve ~10,000 redundant SC items and nearly cost the 8B
   window ~9,000 more. The schema growing is exactly *why the prefix is valid*, not a reason to re-run:
   SC is sequential and greedy, so item i depends only on the schema built from items 0..i-1, which
   means the `full` run **passes through every smaller size's end-state on its way**. Slicing full's SC
   at N reproduces the size-N run exactly — PROVEN 27/27 byte-identical (9 cases x 1k/3k/6k on the 4B
   data), schema/#rel reconstruction exact too.
   ⇒ **Use `datasets/slice_sc_results.py` for SC.** This file (OIE/SD) remains useful only when you must
   run SC for a size that `full` has NOT reached yet (e.g. the 32B window, whose SD stopped at 6,400).

Usage (place sliced totals where REUSE_STAGE_DIR expects them, under <out>/iter0/):
  python datasets/slice_stage_totals.py \
      --src output/webnlg_full_full_selfcanon2_mode1_item_qwen3-8b_bgem3_A100_20260627open/iter0 \
      --n 1000 \
      --out output/_scale_reuse_slices/webnlg_full_1k/iter0

Then run the size-N cell with:  REUSE_STAGE_DIR=output/_scale_reuse_slices/webnlg_full_1k \
      RESUME_FROM=sc DATASET=webnlg_full_1k ... bash run_soda.sh
(run_track3_scale_reuse.sh wires all of this automatically).

SAFETY: reuse is only valid if the size-N dataset really is the first-N prefix of `full`.
Cheap check with an existing pair (e.g. the 4B run):
  python datasets/slice_stage_totals.py --verify_prefix \
      --src <full/iter0> --other <1k/iter0> --n 1000
"""
import os
import json
import argparse

STAGE_FILES = ["oie_total.json", "sd_total.json"]


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)   # == edc_framework save format


def slice_totals(src, n, out):
    os.makedirs(out, exist_ok=True)
    for fn in STAGE_FILES:
        sp = os.path.join(src, fn)
        if not os.path.exists(sp):
            raise FileNotFoundError(
                f"missing {sp}. The `full` run must have finished OIE+SD first "
                f"(oie_total.json + sd_total.json in its iter0)."
            )
        data = _load(sp)
        if not isinstance(data, list):
            raise ValueError(f"{sp} is not a list (got {type(data).__name__}); "
                             f"cannot prefix-slice by item.")
        if len(data) < n:
            raise ValueError(f"{sp} has {len(data)} items < requested n={n}. "
                             f"Is --src really the FULL run?")
        _dump(data[:n], os.path.join(out, fn))
        print(f"  [{fn}] {len(data)} -> {n}  ->  {os.path.join(out, fn)}")
    print(f"[slice] done -> {out}")


def verify_prefix(src, other, n):
    """Assert the first-n entries of src's totals == other's totals (byte-equal after json)."""
    ok = True
    for fn in STAGE_FILES:
        a = _load(os.path.join(src, fn))[:n]
        b = _load(os.path.join(other, fn))
        same = (len(b) == n) and (a == b)
        print(f"  [{fn}] full[:{n}] == other ? {same}  (other len={len(b)})")
        ok = ok and same
    print("[verify] PREFIX-REUSE VALID" if ok else "[verify] MISMATCH — do NOT reuse")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the FULL run's iter dir (has oie_total.json + sd_total.json)")
    ap.add_argument("--n", type=int, help="prefix length (n_items of the smaller size)")
    ap.add_argument("--out", help="target iter dir to write the sliced totals into")
    ap.add_argument("--verify_prefix", action="store_true",
                    help="instead of slicing, check that --other == --src[:n]")
    ap.add_argument("--other", help="the smaller run's iter dir (for --verify_prefix)")
    args = ap.parse_args()

    if args.verify_prefix:
        if not (args.other and args.n):
            ap.error("--verify_prefix needs --other and --n")
        raise SystemExit(0 if verify_prefix(args.src, args.other, args.n) else 1)

    if not (args.n and args.out):
        ap.error("slicing needs --n and --out (or use --verify_prefix)")
    slice_totals(args.src, args.n, args.out)


if __name__ == "__main__":
    main()
