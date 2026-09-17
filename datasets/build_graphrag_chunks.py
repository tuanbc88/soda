"""Convert HotpotQA / MuSiQue DISTRACTOR dev sets into SODA input.

DISTRACTOR setting: each question ships its own bounded context (hotpot = 10
paragraphs, musique = 20). We do NOT concatenate a question's paragraphs (they are
long and about different entities — concatenating would overflow the OIE token cap
and make coref link across unrelated topics). Instead:

    **1 paragraph = 1 line** in `{name}.txt`  (so 1 question/item -> many lines)

and `{name}_note.json` keeps the TRACE: which lines belong to which question, and
which paragraphs are GOLD (supporting facts). Paragraphs are NOT deduped across
questions (each question keeps its own), so every line maps to exactly one question
— see `--dedup` to collapse identical titles if you want to cut KGC cost.

note.json structure:
  {
    "dataset": str, "n_questions": int, "n_chunks": int,
    "questions": [{"q_index", "qid", "question", "answer", "gold_titles"}],
    "chunks":    [{"chunk_id"(==line), "q_index", "para_index", "title", "is_supporting"}]
  }

Sources (gitignored, under kg2rag/v_original/data/):
  hotpot : hotpotqa/hotpot_dev_distractor_v1.json
  musique: MuSiQue/musique_ans_v1.0_dev_mapped.jsonl

Usage:
  python datasets/build_graphrag_chunks.py --dataset both
  python datasets/build_graphrag_chunks.py --dataset hotpot --max_questions 300
"""
import os
import re
import json
import argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HOTPOT_SRC = os.path.join(ROOT, "kg2rag/v_original/data/hotpotqa/hotpot_dev_distractor_v1.json")
MUSIQUE_SRC = os.path.join(ROOT, "kg2rag/v_original/data/MuSiQue/musique_ans_v1.0_dev_mapped.jsonl")


def _norm(text):
    """One chunk = one line: collapse all whitespace/newlines to single spaces."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _write(questions, paragraphs, name, dataset):
    """paragraphs: list of (q_index, para_index, title, text, is_supporting, extra)
    where extra is a dict of source-aligned fields (e.g. musique idx/seq) merged into
    the chunk record — needed to map SODA triples back to the KG2RAG KG keys."""
    txt_path = os.path.join(HERE, f"{name}.txt")
    note_path = os.path.join(HERE, f"{name}_note.json")
    chunks = []
    with open(txt_path, "w", encoding="utf-8") as f:
        for cid, (qi, pi, title, text, is_sup, extra) in enumerate(paragraphs):
            f.write(text + "\n")
            rec = {"chunk_id": cid, "q_index": qi, "para_index": pi,
                   "title": title, "is_supporting": bool(is_sup)}
            rec.update(extra or {})
            chunks.append(rec)
    note = {"dataset": dataset, "n_questions": len(questions), "n_chunks": len(chunks),
            "questions": questions, "chunks": chunks}
    with open(note_path, "w", encoding="utf-8") as f:
        json.dump(note, f, ensure_ascii=False, indent=2)
    n_gold = sum(1 for c in chunks if c["is_supporting"])
    print(f"[{name}] {len(questions)} questions -> {len(chunks)} paragraph-lines "
          f"(avg {len(chunks)/max(1,len(questions)):.1f}/q) | {n_gold} gold paragraphs")
    print(f"        -> {txt_path}")
    print(f"        -> {note_path}")


def build_hotpot(max_questions=None, src=None, dedup=False):
    with open(src or HOTPOT_SRC, encoding="utf-8") as f:
        data = json.load(f)
    if max_questions:
        data = data[:max_questions]
    seen_chunks = set()

    questions, paragraphs = [], []
    for qi, s in enumerate(data):
        sup_titles = {t for t, _ in s.get("supporting_facts", [])}
        questions.append({"q_index": qi, "qid": s["_id"], "question": s["question"],
                          "answer": s.get("answer"), "gold_titles": sorted(sup_titles)})
        for pi, (title, sents) in enumerate(s["context"]):
            title = _norm(title)
            body = _norm("".join(sents))
            text = f"{title}. {body}" if body else title
            if dedup:
                # duplicate (title, text) across questions -> ONE SODA record is
                # enough (the KG2RAG KG is keyed per title; the adapter merges) --
                # big KGC saver on distractor sets where titles repeat
                k = (title, text)
                if k in seen_chunks:
                    continue
                seen_chunks.add(k)
            paragraphs.append((qi, pi, title, text, title in sup_titles, {}))
    _write(questions, paragraphs, "hotpot_chunk", "hotpotqa_distractor")


def build_musique(max_questions=None, src=None, dedup=False):
    with open(src or MUSIQUE_SRC, encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    if max_questions:
        data = data[:max_questions]
    seen_chunks = set()

    questions, paragraphs = [], []
    for qi, s in enumerate(data):
        gold = []
        for pi, p in enumerate(s["paragraphs"]):
            title = _norm(p["title"])
            body = _norm(p["paragraph_text"])
            text = f"{title}. {body}" if body else title
            is_sup = bool(p.get("is_supporting"))
            if is_sup:
                gold.append(title)
            # idx/seq are the KG2RAG musique KG keys (kg[title][seq]); keep them in the trace
            if dedup:
                k = (title, str(p.get("seq")))
                if k in seen_chunks:
                    continue
                seen_chunks.add(k)
            paragraphs.append((qi, pi, title, text, is_sup,
                               {"idx": p.get("idx"), "seq": p.get("seq")}))
        questions.append({"q_index": qi, "qid": s["id"], "question": s["question"],
                          "answer": s.get("answer"), "gold_titles": sorted(set(gold))})
    _write(questions, paragraphs, "musique_chunk", "musique_ans_dev")


def build_eduhcmut(src=None):
    """Emit `edu_hcmut01_chunk_note.json` for the ND02 eduhcmut arm.

    Different in kind from the hotpot/musique builders above, in two ways.

    1. **It does NOT write the .txt.** `datasets/edu_hcmut01.txt` already exists and the 36 SODA KGs
       (2026-07-20) are line-aligned to it. Regenerating it risks a different order, which would
       silently mis-key every triple in those KGs. So this VERIFIES alignment against the existing
       file and refuses to write a note if it does not match, rather than re-deriving the corpus.
    2. **There are no questions here.** eduhcmut's questions live in the `8-` retrieval gold
       (ND02), not in the chunk file, so `questions` is empty and `is_supporting` is always false.
       The note exists only to give `soda_to_kg2rag.py` its chunk -> (title, seq) map.

    `title` = **doc_id**, matching `extract_kg_upstream.extract_eduhcmut(--title_field doc_id)`, which
    is what built the KG2RAG-original baseline. That choice is load-bearing: doc_id makes a document
    group its chunks (the analogue of a hotpot article grouping its sentences), whereas chunk_id would
    make every chunk its own document and fragment the graph. `seq` is the running index within a
    title in file order, which reproduces that function's grouping exactly, so an SODA KG converted
    through the adapter lands on the SAME keys as the baseline and the two arms stay comparable.
    """
    src = src or os.path.join(HERE, "eduhcmut_gold", "2-chunks.json")
    txt_path = os.path.join(HERE, "edu_hcmut01.txt")
    note_path = os.path.join(HERE, "edu_hcmut01_chunk_note.json")

    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"expected a JSON list of chunks, got {type(data).__name__}")
    for k in ("chunk_id", "chunk_content", "doc_id"):
        if k not in data[0]:
            raise SystemExit(f"chunk missing '{k}': keys={list(data[0])}")

    if not os.path.exists(txt_path):
        raise SystemExit(f"[eduhcmut] {txt_path} does not exist; this builder aligns to it, "
                         f"it does not create it")
    with open(txt_path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    if len(lines) != len(data):
        raise SystemExit(f"[eduhcmut] REFUSING: {txt_path} has {len(lines)} lines but {src} has "
                         f"{len(data)} chunks. A note built on a mismatched corpus would mis-key "
                         f"every triple in the existing KGs.")
    bad = [i for i in range(len(data)) if _norm(data[i]["chunk_content"]) != _norm(lines[i])]
    if bad:
        raise SystemExit(f"[eduhcmut] REFUSING: {len(bad)} line(s) differ between {txt_path} and "
                         f"{src} (first at index {bad[0]}). The KGs are aligned to the .txt, so the "
                         f"note must be built from the same ordering.")

    seq_of = defaultdict(int)
    chunks = []
    for cid, ch in enumerate(data):
        title = str(ch["doc_id"])
        seq = seq_of[title]
        seq_of[title] += 1
        chunks.append({"chunk_id": cid, "q_index": None, "para_index": seq,
                       "title": title, "is_supporting": False,
                       "seq": seq, "src_chunk_id": ch["chunk_id"]})

    note = {"dataset": "eduhcmut", "n_questions": 0, "n_chunks": len(chunks),
            "questions": [], "chunks": chunks}
    with open(note_path, "w", encoding="utf-8") as f:
        json.dump(note, f, ensure_ascii=False, indent=2)
    print(f"[eduhcmut] {len(chunks)} chunks -> {len(seq_of)} titles (by doc_id); "
          f"alignment to {os.path.basename(txt_path)} verified on all lines")
    print(f"        -> {note_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hotpot", "musique", "eduhcmut", "both"], default="both")
    # Subsample the FIRST N questions; same canonical name (so the fewshot dir resolves).
    ap.add_argument("--max_questions", type=int, default=None)
    # RQ3 paired-subset support: point at a subset file (make_subset.py output) so the
    # SODA chunks cover EXACTLY the baseline subset; --dedup drops duplicate paragraphs
    # across questions (KG is per-title -> identical KG, much less SODA compute).
    ap.add_argument("--src", default=None, help="override the source data file (subset json/jsonl)")
    ap.add_argument("--dedup", action="store_true", help="dedup identical (title,text)/(title,seq) chunks")
    args = ap.parse_args()
    if args.dataset in ("hotpot", "both"):
        build_hotpot(args.max_questions, src=args.src, dedup=args.dedup)
    if args.dataset in ("musique", "both"):
        build_musique(args.max_questions, src=args.src, dedup=args.dedup)
    # NOT in "both": eduhcmut takes a different source shape (a flat chunk list, no questions) and
    # aligns to an existing .txt, so it must be asked for explicitly.
    if args.dataset == "eduhcmut":
        build_eduhcmut(src=args.src)
