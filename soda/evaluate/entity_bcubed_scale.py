"""Entity B-cubed across the scale grid, reconstructed from a run's EC decision log. No GPU.

WHY THIS EXISTS
    `tab:scale_ent` reports #ent, a schema-SIZE statistic. A count above the gold G is equally
    consistent with over-splitting gold types and with a legitimately finer taxonomy that gold does
    not draw, and a model can over-split some types while merging others -- one net scalar cannot
    separate those. B-cubed precision and recall can, and in opposite directions:

        low PRECISION -> a predicted type lumps mentions gold keeps apart   (OVER-MERGE)
        low RECALL    -> a gold type is scattered across predicted types    (OVER-SPLIT)

    So this is what turns tab:scale_ent from a calibration statement into an accuracy one.

WHY THE CUTS CAN BE RECONSTRUCTED
    Entity canonicalization is sequential and greedy over a schema that starts EMPTY and only grows,
    so the full run passes through every smaller cut's end state on its way. Slicing its per-item
    decision log (`entity_canon_{ec}.json`, one entry per item) to the first N items therefore
    reproduces a native size-N run. This is the same prefix argument `slice_sc_results.py` makes for
    SC and for the entity SCHEMA SIZE; here it is extended to the per-mention predictions that
    B-cubed needs. The gold cuts are byte-prefixes of each other (verified: md5 of the first 1,000
    lines of webnlg_full_full.txt equals webnlg_full_1k.txt).

    --verify closes the loop: reconstruct at the source run's OWN size and diff against the
    clustering_entity_metrics.csv that run produced natively. Run it after any change to EC.

USAGE
    python soda/evaluate/entity_bcubed_scale.py \
        --src output/webnlg_full_full_selfcanon2_mode1_item_qwen3-4b_bgem3_A100_20260627open/iter0 \
        --sizes 1k=1000,3k=3000,6k=6000,13k=13211 --verify
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from soda.evaluate import clustering_metric as cm  # noqa: E402

GOLD_FULL = "./soda/evaluate/references/webnlg_full_full.txt"
GT_SCHEMA = "./schemas/webnlg_full_schema.json"
GROUND = "./schemas/gold_entity_types_webnlg_full.json"
# canonical per-cut gold files, preferred over slicing so the check uses an independent artifact
CUT_GOLD = {"1k": "./soda/evaluate/references/webnlg_full_1k.txt",
            "3k": "./soda/evaluate/references/webnlg_full_3k.txt",
            "6k": "./soda/evaluate/references/webnlg_full_6k.txt",
            "13k": GOLD_FULL, "full": GOLD_FULL}


def gold_for(label, n, tmp):
    """The cut's gold KG: the canonical file when one exists, else the first n lines of full."""
    p = CUT_GOLD.get(label)
    if p and os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            if sum(1 for _ in f) == n:
                return p, "canonical"
    out = os.path.join(tmp, f"gold_{label}.txt")
    with open(GOLD_FULL, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
        for i, line in enumerate(f):
            if i >= n:
                break
            g.write(line)
    return out, "sliced"


def score_cut(src, n, label, tmp):
    """Slice the EC decision log to n items, then run the stock entity metric on it."""
    pred = os.path.join(tmp, f"pred_{label}")
    outd = os.path.join(tmp, f"out_{label}")
    os.makedirs(pred, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(src, "entity_canon_ec*.json"))):
        data = json.load(open(p, encoding="utf-8"))
        if len(data) < n:
            raise SystemExit(f"[fatal] {os.path.basename(p)} has {len(data)} items < requested {n}; "
                             f"a cut cannot be reconstructed from a shorter run")
        json.dump(data[:n], open(os.path.join(pred, os.path.basename(p)), "w", encoding="utf-8"),
                  ensure_ascii=False)
    gk, how = gold_for(label, n, tmp)
    cm.run_entity(gt_kg=gk, gt_schema=GT_SCHEMA, pred_dir=pred, output_dir=outd,
                  ground_path=GROUND, p31_path=None)
    rows = {}
    csvp = os.path.join(outd, "clustering_entity_metrics.csv")
    if os.path.exists(csvp):
        for r in csv.DictReader(open(csvp, encoding="utf-8")):
            rows[r["ec_case"]] = r
    return rows, how


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="iter dir holding entity_canon_ec*.json")
    ap.add_argument("--sizes", required=True, help="label=N list, e.g. 1k=1000,3k=3000")
    ap.add_argument("--verify", action="store_true",
                    help="also reconstruct at the source run's own size and diff against its "
                         "native clustering_entity_metrics.csv")
    ap.add_argument("--out", default=None, help="write a tidy CSV here")
    a = ap.parse_args()

    sizes = []
    for tok in a.sizes.split(","):
        lab, _, n = tok.partition("=")
        sizes.append((lab.strip(), int(n)))

    tmp = tempfile.mkdtemp(prefix="entb3_")
    results = {}
    try:
        for lab, n in sizes:
            print(f"\n===== cut {lab} (n={n}) =====")
            results[lab], how = score_cut(a.src, n, lab, tmp)
            print(f"  [gold] {how}")

        print("\n" + "=" * 78)
        print(f"Entity B-cubed by cut   src={a.src}")
        print("=" * 78)
        cases = sorted({c for r in results.values() for c in r})
        hdr = "  ".join(f"{lab:>18}" for lab, _ in sizes)
        print(f"{'phi_e':<22} {hdr}")
        for c in cases:
            cells = []
            for lab, _ in sizes:
                r = results[lab].get(c)
                cells.append("--".rjust(18) if not r else
                             f"P{float(r['bcubed_p']):.3f} R{float(r['bcubed_r']):.3f}".rjust(18))
            print(f"{c:<22} " + "  ".join(cells))
        print(f"\n{'phi_e':<22} " + "  ".join(f"{lab:>18}" for lab, _ in sizes) + "   (F1 / #predType)")
        for c in cases:
            cells = []
            for lab, _ in sizes:
                r = results[lab].get(c)
                cells.append("--".rjust(18) if not r else
                             f"{float(r['bcubed_f1']):.4f} / {r['n_pred_clusters']:>4}".rjust(18))
            print(f"{c:<22} " + "  ".join(cells))

        if a.verify:
            native = os.path.join(a.src, "eval_clustering", "clustering_entity_metrics.csv")
            if not os.path.exists(native):
                print(f"\n[verify] SKIPPED: no native csv at {native}")
            else:
                any_ec = sorted(glob.glob(os.path.join(a.src, "entity_canon_ec*.json")))[0]
                n_native = len(json.load(open(any_ec, encoding="utf-8")))
                print(f"\n[verify] reconstructing at the source's own N={n_native}")
                rec, _ = score_cut(a.src, n_native, "_verify", tmp)
                nat = {r["ec_case"]: r for r in csv.DictReader(open(native, encoding="utf-8"))}
                bad = 0
                for c, nr in nat.items():
                    rr = rec.get(c)
                    if not rr:
                        print(f"  {c:<24} MISSING in reconstruction"); bad += 1; continue
                    for k in ("bcubed_p", "bcubed_r", "bcubed_f1"):
                        d = abs(float(nr[k]) - float(rr[k]))
                        if d > 5e-4:
                            print(f"  {c:<24} {k} native={nr[k]} recon={rr[k]} diff={d:.5f}  MISMATCH")
                            bad += 1
                    if not bad:
                        print(f"  {c:<24} F1 {float(nr['bcubed_f1']):.4f} == {float(rr['bcubed_f1']):.4f}  OK")
                if bad:
                    print(f"\n[verify] FAILED ({bad} mismatches) -- do NOT report these numbers")
                    return 1
                print("\n[verify] PASS: reconstruction reproduces the native run")

        if a.out:
            with open(a.out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["src", "cut", "n", "ec_case", "bcubed_p", "bcubed_r", "bcubed_f1",
                            "n_pred_clusters", "n_gold_clusters", "n_mentions"])
                for lab, n in sizes:
                    for c, r in sorted(results[lab].items()):
                        w.writerow([a.src, lab, n, c, r["bcubed_p"], r["bcubed_r"], r["bcubed_f1"],
                                    r["n_pred_clusters"], r["n_gold_clusters"], r["n_mentions"]])
            print(f"\nwrote {a.out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
