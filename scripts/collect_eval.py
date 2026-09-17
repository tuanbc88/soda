"""Collect the SMALL eval JSONs from a run's iter dir into one bundle to share.

`output/` is per-machine and gitignored, so to send results for analysis just run this
on the server and paste/attach the printed bundle (or the written file).

Usage:
  python scripts/collect_eval.py output/webnlg_selfcanon2_mode1_item_qwen7B4bit_minilm_T4_20260516/iter0
  python scripts/collect_eval.py <iter_dir> --out /tmp/eval_bundle.json
"""
import os
import sys
import json
import glob
import argparse

# only the small metric files — never the big KG dumps (canon_kg_*, oie_total, sd_total)
PATTERNS = ["eval_*/*.json", "eval_*/*.csv", "oie_refine_cost.json", "coref_total.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("iter_dir", help="a run's iterN dir (the one holding eval_*/)")
    ap.add_argument("--out", default=None, help="write bundle here (else print)")
    a = ap.parse_args()

    bundle = {"run": os.path.normpath(a.iter_dir), "files": {}}
    for pat in PATTERNS:
        for p in sorted(glob.glob(os.path.join(a.iter_dir, pat))):
            rel = os.path.relpath(p, a.iter_dir).replace("\\", "/")
            try:
                if p.endswith(".json"):
                    bundle["files"][rel] = json.load(open(p, encoding="utf-8"))
                else:  # .csv → keep as text
                    bundle["files"][rel] = open(p, encoding="utf-8").read()
            except Exception as e:
                bundle["files"][rel] = f"[read error: {e}]"

    text = json.dumps(bundle, ensure_ascii=False, indent=2)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)
        kb = os.path.getsize(a.out) / 1024
        print(f"[collect_eval] {len(bundle['files'])} files -> {a.out} ({kb:.0f} KB)")
    else:
        print(text)


if __name__ == "__main__":
    main()
