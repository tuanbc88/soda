"""Build the KATE demo pools (DECISIONS 2026-07-03b/c) -> datasets/kate_pool/{ds}_pool.jsonl

One JSON per line: {"text": ..., "triplets": [[h, r, t], ...]} — the candidate
(text, gold-triples) demonstrations that `soda/utils/kate_selector.py` retrieves
from at OIE time. Pools live in their own folder (author request) so KATE demo
versions never mix with the static few-shot files.

SOURCES + GUARDS:
  webnlg (local, no GPU/network):
    - datasets/webnlg_full_full.txt + soda/evaluate/references/webnlg_full_full.txt
      = the WebNLG+2020 TRAIN split (13,211 triple-sets; build_full_dataset.py)
      -> disjoint from the 1165-item test set by construction; a normalized-text
      dedup vs datasets/webnlg.txt is applied anyway (belt and braces).
  rebel (SERVER; needs `pip install datasets`):
    - HF Babelscape/rebel-dataset, --split validation (default; train also works),
      linearized triplets decoded with the canonical Babelscape parser;
    - GUARD 1: drop any sample whose text (normalized) appears in datasets/rebel.txt
      (our test 1000-sample);
    - GUARD 2 (gold-quality, per the distant-supervision finding): keep a sample
      only if EVERY triple's head AND tail literally appear in its text
      (NOT_VERBALIZED demos would teach unverbalized extraction);
    - --cap (default 20000) after filtering, fixed-seed shuffle.

RUN:
  python datasets/build_kate_pool.py --dataset webnlg
  python datasets/build_kate_pool.py --dataset rebel --split validation   # server
"""
import os
import re
import ast
import json
import random
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "kate_pool")


def _norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def _write(rows, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{name}_pool.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_tr = sum(len(r["triplets"]) for r in rows)
    print(f"[{name}] {len(rows)} demos ({n_tr} triples, "
          f"avg {n_tr / max(1, len(rows)):.1f}/demo) -> {out}")


def build_webnlg():
    texts = [l.rstrip("\n") for l in open(os.path.join(HERE, "webnlg_full_full.txt"), encoding="utf-8")]
    golds = [l.rstrip("\n") for l in open(os.path.join(
        ROOT, "soda", "evaluate", "references", "webnlg_full_full.txt"), encoding="utf-8")]
    assert len(texts) == len(golds), "train text/gold line mismatch"
    test_texts = {_norm(l) for l in open(os.path.join(HERE, "webnlg.txt"), encoding="utf-8")}
    rows, seen, n_skip = [], set(), 0
    for text, gold in zip(texts, golds):
        key = _norm(text)
        if key in test_texts:            # belt-and-braces (train should be disjoint)
            n_skip += 1
            continue
        if key in seen or not text.strip():
            continue
        seen.add(key)
        try:
            trips = [list(map(str, t)) for t in ast.literal_eval(gold) if len(t) == 3]
        except Exception:
            continue
        if trips:
            rows.append({"text": text, "triplets": trips})
    print(f"[webnlg] skipped {n_skip} test-overlapping lines")
    _write(rows, "webnlg")


# canonical Babelscape linearized-triplet parser (rebel-large model card)
def _decode_rebel(text):
    triplets = []
    subject, relation, object_, current = "", "", "", None
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = "t"
            if relation != "":
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if subject != "" and relation != "" and object_ != "":
        triplets.append((subject.strip(), relation.strip(), object_.strip()))
    return triplets


def build_rebel(split, cap, seed):
    from datasets import load_dataset          # server dependency
    try:
        ds = load_dataset("Babelscape/rebel-dataset", split=split)
    except RuntimeError:
        # datasets>=4.x dropped script-based loaders ("Dataset scripts are no longer supported,
        # but found rebel-dataset.py"). The HF auto-converted parquet branch carries the same
        # context/triplets schema, so KATE's rebel pool builds identically off it. Verified on
        # datasets 4.5.0 (172,860 validation rows, cols id/title/context/triplets). 2026-07-21.
        ds = load_dataset("Babelscape/rebel-dataset", split=split, revision="refs/convert/parquet")
    test_texts = {_norm(l) for l in open(os.path.join(HERE, "rebel.txt"), encoding="utf-8")}
    rows, seen = [], set()
    n_unverb = n_test = 0
    for s in ds:
        text = re.sub(r"\s+", " ", s.get("context") or "").strip()
        key = _norm(text)
        if not text or key in seen:
            continue
        if key in test_texts:                                  # GUARD 1
            n_test += 1
            continue
        trips = [[h, r, t] for h, r, t in _decode_rebel(s.get("triplets") or "")]
        if not trips:
            continue
        if not all((h in text) and (t in text) for h, _, t in trips):   # GUARD 2
            n_unverb += 1
            continue
        seen.add(key)
        rows.append({"text": text, "triplets": trips})
    print(f"[rebel] dropped: {n_test} test-overlap, {n_unverb} not-verbalized")
    random.Random(seed).shuffle(rows)
    _write(rows[:cap], "rebel")


def build_wikinre(split, cap, seed):
    """wiki-nre (SERVER; needs `pip install datasets`).
    Source: HF Saibo-creator/wiki-nre, --split train (default). Our 999-item test set is that
    dataset's `stratified_test_1K` split, so `train` is disjoint by construction; we still apply
    the normalized-text overlap guard against datasets/wiki-nre.txt (belt-and-braces). Triples are
    read from the structured `triplets` field (subject/predicate/object surfaceforms). Same two
    guards as rebel: GUARD 1 test-overlap dedup, GUARD 2 keep only fully-verbalized triples
    (head AND tail surfaceform literally in text) so demos never teach unverbalized extraction."""
    from datasets import load_dataset          # server dependency
    ds = load_dataset("Saibo-creator/wiki-nre", split=split)
    test_texts = {_norm(l) for l in open(os.path.join(HERE, "wiki-nre.txt"), encoding="utf-8")}
    rows, seen = [], set()
    n_unverb = n_test = 0

    def _surf(node):
        # node may be a dict {'surfaceform':..,'uri':..} or already a string
        if isinstance(node, dict):
            return (node.get("surfaceform") or "").strip()
        return str(node or "").strip()

    for s in ds:
        text = re.sub(r"\s+", " ", s.get("text") or "").strip()
        key = _norm(text)
        if not text or key in seen:
            continue
        if key in test_texts:                                  # GUARD 1
            n_test += 1
            continue
        raw = s.get("triplets")
        if isinstance(raw, str):
            try:
                raw = ast.literal_eval(raw)
            except Exception:
                continue
        trips = []
        for t in (raw or []):
            h, r, o = _surf(t.get("subject")), _surf(t.get("predicate")), _surf(t.get("object"))
            if h and r and o:
                trips.append([h, r, o])
        if not trips:
            continue
        if not all((h in text) and (t in text) for h, _, t in trips):   # GUARD 2
            n_unverb += 1
            continue
        seen.add(key)
        rows.append({"text": text, "triplets": trips})
    print(f"[wiki-nre] dropped: {n_test} test-overlap, {n_unverb} not-verbalized")
    random.Random(seed).shuffle(rows)
    _write(rows[:cap], "wiki-nre")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["webnlg", "rebel", "wiki-nre"], required=True)
    ap.add_argument("--split", default="validation", help="rebel HF split (validation|train); wiki-nre uses train")
    ap.add_argument("--cap", type=int, default=20000, help="rebel/wiki-nre pool cap after filtering")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.dataset == "webnlg":
        build_webnlg()
    elif args.dataset == "rebel":
        build_rebel(args.split, args.cap, args.seed)
    else:
        wsplit = "train" if args.split == "validation" else args.split
        build_wikinre(wsplit, args.cap, args.seed)


if __name__ == "__main__":
    main()
