"""
Build FULL / larger SODA datasets from the HuggingFace source corpora, for the
scale study (RESEARCH_QUESTIONS.md §5b: does retrieval matter at full scale?).

Emits SODA's two aligned files per dataset name:
  - datasets/{name}.txt                      raw text, ONE record per line
  - soda/evaluate/references/{name}.txt      gold triples, Python-list-of-lists, 1/line
and copies the base dataset's few-shot dir so the pipeline can run on the new name.

**Scope (decided 2026-06-18): webnlg ONLY.** webnlg is the only one of the 3 with full gold at
every scale; scaling rebel/wiki-nre would require building gold for tens of thousands of records
(expensive + undercuts the "automatic" premise). rebel/wiki-nre are left as stubs on purpose —
for the scale study use them as low-end pool-size anchors instead (RESEARCH_QUESTIONS.md §5b).
Implemented: **webnlg** (release_v3.0_en — 13,211 train triple-sets, 372 rel).

Run from the repo root (webnlg uses ONLY stdlib — downloads + parses the official XML,
so it does NOT need the `datasets` library, avoiding its fsspec version bug):
  python datasets/build_full_dataset.py --dataset webnlg --split train \
      --fractions 1000,3000,6000,all --seed 42

Then (Mode 1 — no seed schema needed):
  DATASET=webnlg_full_3k EMB_TAG=minilm SC_EMBEDDER=sentence-transformers/all-MiniLM-L6-v2 \
      bash run_matrix.sh A100      # (or just exp1: runa.py --dataset_name webnlg_full_3k --no_schema ...)
"""

import os
import re
import ast
import io
import zipfile
import urllib.request
import shutil
import argparse
import xml.etree.ElementTree as ET

# Official WebNLG v3.0 release (the same archive the HF loader script pulls). Parsed
# directly here with stdlib XML so we DON'T depend on the `datasets`/`fsspec` library
# (whose version mismatch raises "Loading a dataset cached in a LocalFileSystem is not
# supported"). Deterministic: pinned commit.
WEBNLG_ZIP_URL = ("https://gitlab.com/shimorina/webnlg-dataset/-/archive/"
                  "587fa698bec705efbefe72a235a6019c2b9b8b6c/"
                  "webnlg-dataset-587fa698bec705efbefe72a235a6019c2b9b8b6c.zip")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "datasets")
REF_DIR = os.path.join(REPO, "soda", "evaluate", "references")
FS_DIR = os.path.join(REPO, "soda", "few_shot_examples")


def _clean_text(t):
    """One record per line -> collapse any newlines/whitespace."""
    return re.sub(r"\s+", " ", str(t).replace("\n", " ").replace("\r", " ")).strip()


def _frac_tag(n):
    if n is None:
        return "full"
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


# ---------------------------------------------------------------------------
# per-source extractors -> list[ (text, [[s,p,o], ...]) ]
# ---------------------------------------------------------------------------
def _parse_webnlg_entry(entry):
    """<entry> -> (text, [[s,p,o],...]) or None. Uses modifiedtripleset + best lex."""
    triples = []
    for mt in entry.findall(".//mtriple"):
        s = (mt.text or "").strip()
        parts = [p.strip() for p in s.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append(parts)
    if not triples:
        return None
    text = None
    for lex in entry.findall("lex"):
        tx = _clean_text(lex.text or "")
        if not tx:
            continue
        if lex.get("comment") == "good":   # prefer a good-rated lexicalization
            text = tx; break
        if text is None:
            text = tx                       # else keep the first non-empty
    return (text, triples) if text else None


def extract_webnlg(split):
    """Download the official WebNLG v3.0 archive and parse the XML for `split`
    (train/dev/test). stdlib only — no `datasets`/`fsspec` dependency."""
    print(f"Downloading WebNLG v3.0 archive (~25MB) ...")
    raw = urllib.request.urlopen(WEBNLG_ZIP_URL, timeout=120).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    want = f"/release_v3.0/en/{split}/"
    xmls = [n for n in zf.namelist() if want in n and n.endswith(".xml")]
    if not xmls:
        raise SystemExit(f"[webnlg] no XML under {want} in the archive "
                         f"(splits: train/dev/test). Got e.g. {zf.namelist()[:3]}")
    print(f"  parsing {len(xmls)} XML files under {want}")
    out = []
    for name in sorted(xmls):
        root = ET.fromstring(zf.read(name))
        for entry in root.findall(".//entry"):
            rec = _parse_webnlg_entry(entry)
            if rec:
                out.append(rec)
    return out, "webnlg"


def extract_rebel(split):
    raise SystemExit(
        "[build] rebel not implemented yet. Verify the HF field layout first:\n"
        "  load_dataset('Babelscape/rebel-dataset', split='train') -> inspect ds.features.\n"
        "  REBEL stores triplets in a linearized string ('triplets' field, <triplet>/<subj>/<obj>\n"
        "  markers) OR structured 'entities'/'relations' depending on the loader; write a parser\n"
        "  to [[subj, rel, obj], ...] then reuse the same writer. Subsample 10-50k from the 174k\n"
        "  test split (schema is capped ~220 -> tests Mode-1 self-gen bloat).")


def extract_wikinre(split):
    raise SystemExit(
        "[build] wiki-nre not implemented yet. Verify the HF field layout first:\n"
        "  load_dataset('Saibo-creator/wiki-nre', split=...) -> inspect ds.features for the\n"
        "  sentence text + triple fields, map to [[subj, rel, obj], ...]. ~29,619 test entries.")


EXTRACTORS = {"webnlg": extract_webnlg, "rebel": extract_rebel, "wiki-nre": extract_wikinre}


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------
def write_dataset(name, records, base_fewshot):
    os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(REF_DIR, exist_ok=True)
    in_path = os.path.join(DATA_DIR, f"{name}.txt")
    ref_path = os.path.join(REF_DIR, f"{name}.txt")
    with open(in_path, "w", encoding="utf-8") as fi, open(ref_path, "w", encoding="utf-8") as fr:
        for text, triples in records:
            fi.write(text + "\n")
            fr.write(repr([[a, b, c] for a, b, c in triples]) + "\n")   # ast.literal_eval-able
    # few-shot dir: copy the base dataset's (generic; lets the pipeline find few-shots)
    src = os.path.join(FS_DIR, base_fewshot)
    dst = os.path.join(FS_DIR, name)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        fs_msg = f"few-shots copied {base_fewshot} -> {name}"
    else:
        fs_msg = f"WARN: base few-shot dir {src} missing"
    # sanity: re-read the gold to confirm it parses
    n_tr = sum(len(ast.literal_eval(l)) for l in open(ref_path, encoding="utf-8"))
    print(f"  [{name}] records={len(records)} triples={n_tr} | {fs_msg}")
    print(f"      -> {in_path}")
    print(f"      -> {ref_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(EXTRACTORS))
    ap.add_argument("--split", default="train", help="train|dev|test (webnlg full schema = train)")
    ap.add_argument("--fractions", default="1000,3000,6000,all",
                    help="comma list of record counts; 'all' = full split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name_prefix", default=None, help="default {dataset}_full")
    args = ap.parse_args()

    records, base_fewshot = EXTRACTORS[args.dataset](args.split)
    print(f"Loaded {len(records)} records from {args.dataset}:{args.split}")

    import random
    rng = random.Random(args.seed)
    shuffled = records[:]
    rng.shuffle(shuffled)   # fixed-seed shuffle so the 1k subset is a subset of 3k, etc.

    prefix = args.name_prefix or f"{args.dataset}_full"
    fracs = []
    for f in args.fractions.split(","):
        f = f.strip()
        fracs.append(None if f.lower() == "all" else int(f))
    # nested subsets: smaller fractions are prefixes of larger ones (same shuffle)
    for n in sorted(fracs, key=lambda x: (x is None, x)):
        subset = shuffled if n is None else shuffled[:n]
        name = f"{prefix}_{_frac_tag(n)}"
        write_dataset(name, subset, base_fewshot)

    print("\nDone. Generated files are gitignored (regenerate with this script).")
    print("Scale study = **Mode 1 only** (Mode 2/3 would need the extended 372-rel seed schema;")
    print("the 159-rel webnlg_schema is too small for full data -> ignore exp3-6 at scale).")
    print("Run per fraction (the matrix exp1=Mode1-item, exp2=Mode1-itemcluster are the cells):")
    print(f"  DATASET={prefix}_3k EMB_TAG=minilm "
          f"SC_EMBEDDER=sentence-transformers/all-MiniLM-L6-v2 bash run_matrix.sh A100")
    print("Metrics resolve gold by convention from soda/evaluate/references/{DATASET}.txt.")
    print("Compare across fractions: eval_mode1/ (aligned-F1 + #pred relations) + "
          "eval_clustering/ (B-cubed + #clusters) + eval_oie/ — watch the self-generated schema "
          "size grow with #records (the §5b driver).")


if __name__ == "__main__":
    main()
