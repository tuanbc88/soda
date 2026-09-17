"""GOLD-TRIPLE AUDIT — Ragas-style entailment check of the BENCHMARK GOLD itself
(DIAGNOSIS-ONLY; changes no pipeline stage and no headline number).

WHY (author 2026-07-02, DECISIONS 2026-07-02f): evaluation quality is bounded by
gold quality. Test-set label errors are pervasive and destabilize benchmark
rankings (Northcutt, Athalye & Mueller, NeurIPS 2021 Datasets & Benchmarks,
arXiv 2103.14749: >=3.3% avg label errors across 10 major benchmarks). Our KGC
benchmarks are especially exposed: REBEL and Wiki-NRE "gold" is DISTANT
SUPERVISION (Wikipedia-Wikidata alignment), not human annotation, so a gold
triple may not be expressed in its source text at all. The lexical A_unrealized
bucket of oie_miss_diagnosis.py (~5.5% webnlg / ~5.3% rebel) is a cheap proxy;
this tool is the clean LLM version, and its flagged list is the PUBLISHED
EVIDENCE for the paper's judge/gold-audit appendix.

METHOD — CheckEval-style binary checklist (Lee et al., EMNLP 2025) with a
deterministic verdict rule, i.e. granular decomposition of "is this gold triple
right for this text?" (RAGAs faithfulness pattern, Es et al., EACL 2024 demos):
  Q1: is the HEAD entity mentioned in the sentence (name, alias, pronoun,
      or definite description)?
  Q2: is the TAIL entity mentioned likewise?
  Q3: does the sentence STATE the relation between them, or is it strictly
      INFERABLE from the sentence content, or NEITHER?
Verdict rule (in code, not by the LLM):
  q1=no or q2=no  -> NOT_VERBALIZED  (distant-supervision artifact: the fact is
                     true in the KB but absent from this text -> extraction
                     CANNOT recover it; a per-dataset extraction ceiling)
  q3=no           -> UNSUPPORTED    (label-error candidate: entities present but
                     the text does not support the relation)
  q3=inferable    -> IMPLICIT_OK    (true + present, but requires inference; ties
                     to the relation_expression judge's implicit label)
  q3=stated       -> SUPPORTED
Protocol guards mirror relation_expression_judge.py (DECISIONS 2026-07-02d/e):
greedy T=0, per-triple resume, fixed-seed random --sample for the human
validation gate (author fills human_verdict; --human_csv computes kappa), and an
optional PoLL-style JURY (Verga et al. 2024, arXiv 2404.18796: a panel of
diverse judges beats a single large judge with less intra-model bias): run this
script once per judge model, then --merge the csvs -> majority verdict +
pairwise inter-judge kappa (caveat: correlated errors can undermine panels,
arXiv 2605.29800 -> report the inter-judge agreement, not just the majority).

RUN:
  # single judge (validation sample first, then full; resumable)
  python soda/evaluate/gold_triple_audit.py --dataset rebel --backend hf \
      --model Qwen/Qwen2.5-72B-Instruct --sample 50
  python soda/evaluate/gold_triple_audit.py --dataset rebel --backend hf --model ...
  # jury merge (after runs with different --model into different --output_dir)
  python soda/evaluate/gold_triple_audit.py --merge out_qwen out_llama out_gemma \
      --output_dir out_jury
OUTPUT: <out>/gold_audit.csv (idx,h,r,t,q1,q2,q3,verdict,human_verdict)
        <out>/gold_audit_summary.json (% per verdict = the headline evidence)
        <out>/gold_audit_flagged.csv (UNSUPPORTED + NOT_VERBALIZED, for release)
"""
import os
import re
import csv
import sys
import json
import random
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from soda.evaluate.retrieval_recall_metric import GT_CONFIG, load_kg_txt
    from soda.evaluate.relation_expression_judge import make_judge, load_src
except ImportError:
    from retrieval_recall_metric import GT_CONFIG, load_kg_txt
    from relation_expression_judge import make_judge, load_src

VERDICTS = ("SUPPORTED", "IMPLICIT_OK", "UNSUPPORTED", "NOT_VERBALIZED")

SYSTEM = ("You are a careful annotator auditing a knowledge-graph benchmark. You check whether "
          "a gold triple is actually expressed by its source sentence. Be strict and literal.")

USER_TMPL = """Sentence:
"{sentence}"

Gold triple to audit: ({subject}, {relation}, {object})

Answer THREE questions:
Q1: Is the head entity "{subject}" mentioned in the sentence (by name, alias, pronoun,
    or definite description)?
Q2: Is the tail entity "{object}" mentioned in the sentence likewise?
Q3: Does the sentence express the relation "{relation}" between them?
    - stated    : the relation is READABLE from the sentence surface via ANY construction
      (verb, copula, apposition, preposition, parenthetical, genitive, or compound),
      needing no outside knowledge
    - inferable : NOT surface-expressed, but strictly follows from the sentence content
      by genuine inference
    - no        : the sentence does not support this relation

Answer in EXACTLY this format, nothing else:
q1: <yes|no>
q2: <yes|no>
q3: <stated|inferable|no>
reason: <one short clause>"""

FEWSHOT = [
    ("Liselotte Grschebina was born in Karlsruhe.", "Liselotte_Grschebina", "bornIn", "Karlsruhe",
     "yes", "yes", "stated", "'was born in' states the relation; both entities named"),
    ("Ayam penyet is a food found in Java, a Singaporean island.", "Ayam_penyet", "country", "Singapore",
     "yes", "yes", "stated", "the apposition 'a Singaporean island' surface-states the country"),
    ("Alan Shepard held U.S. citizenship.", "Alan_Shepard", "nationality", "United_States",
     "yes", "yes", "inferable", "'American' never surfaces; nationality follows by inference from 'U.S. citizenship'"),
    ("The A-Rosa Luna is 125.8 m long.", "A-Rosa_Luna", "builder", "Neptun_Werft",
     "yes", "no", "no", "the builder is never mentioned in this sentence"),
]


def verdict_rule(q1, q2, q3):
    if q1 == "no" or q2 == "no":
        return "NOT_VERBALIZED"
    if q3 == "no":
        return "UNSUPPORTED"
    if q3 == "inferable":
        return "IMPLICIT_OK"
    return "SUPPORTED"


def build_messages(sentence, s, r, o):
    msgs = [{"role": "system", "content": SYSTEM}]
    for (sent, ss, rr, oo, q1, q2, q3, rea) in FEWSHOT:
        msgs.append({"role": "user", "content": USER_TMPL.format(sentence=sent, subject=ss, relation=rr, object=oo)})
        msgs.append({"role": "assistant", "content": f"q1: {q1}\nq2: {q2}\nq3: {q3}\nreason: {rea}"})
    msgs.append({"role": "user", "content": USER_TMPL.format(sentence=sentence, subject=s, relation=r, object=o)})
    return msgs


def parse_answer(text):
    low = (text or "").lower()
    m1 = re.search(r"q1:\s*(yes|no)", low)
    m2 = re.search(r"q2:\s*(yes|no)", low)
    m3 = re.search(r"q3:\s*(stated|inferable|no)", low)
    rm = re.search(r"reason:\s*(.+)", text or "", re.IGNORECASE)
    reason = rm.group(1).strip()[:200] if rm else ""
    if not (m1 and m2 and m3):
        return None, None, None, None, reason
    q1, q2, q3 = m1.group(1), m2.group(1), m3.group(1)
    return q1, q2, q3, verdict_rule(q1, q2, q3), reason


def cohen_kappa(pairs, labels):
    n = len(pairs)
    if not n:
        return 0.0, 0.0
    po = sum(1 for a, b in pairs if a == b) / float(n)
    pa = {l: sum(1 for a, _ in pairs if a == l) / float(n) for l in labels}
    pb = {l: sum(1 for _, b in pairs if b == l) / float(n) for l in labels}
    pe = sum(pa[l] * pb[l] for l in labels)
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return po, k


def write_outputs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cols = ["idx", "h", "r", "t", "q1", "q2", "q3", "verdict", "reason", "human_verdict"]
    with open(os.path.join(out_dir, "gold_audit.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    flagged = [r for r in rows if r.get("verdict") in ("UNSUPPORTED", "NOT_VERBALIZED")]
    with open(os.path.join(out_dir, "gold_audit_flagged.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in flagged:
            w.writerow({c: r.get(c, "") for c in cols})
    n = len(rows) or 1
    dist = {v: sum(1 for r in rows if r.get("verdict") == v) for v in VERDICTS}
    summary = {"n_audited": len(rows),
               "verdict_%": {v: round(100.0 * dist[v] / n, 2) for v in VERDICTS},
               "suspect_%_(UNSUPPORTED+NOT_VERBALIZED)":
                   round(100.0 * (dist["UNSUPPORTED"] + dist["NOT_VERBALIZED"]) / n, 2)}
    with open(os.path.join(out_dir, "gold_audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[audit] saved -> {out_dir}/gold_audit.csv + _flagged.csv + _summary.json")


def merge_jury(dirs, out_dir):
    """PoLL-style majority verdict over N single-judge audits + pairwise kappa."""
    panels = []
    for d in dirs:
        rows = {}
        for r in csv.DictReader(open(os.path.join(d, "gold_audit.csv"), encoding="utf-8")):
            rows[(r["idx"], r["h"], r["r"], r["t"])] = r
        panels.append(rows)
    keys = set(panels[0]).intersection(*panels[1:])
    merged = []
    for k in sorted(keys, key=lambda x: (int(x[0]), x[1])):
        votes = [p[k]["verdict"] for p in panels]
        maj = max(set(votes), key=votes.count)
        base = dict(panels[0][k])
        base["verdict"] = maj
        base["reason"] = f"jury {votes}"
        merged.append(base)
    write_outputs(merged, out_dir)
    kappas = {}
    for i in range(len(panels)):
        for j in range(i + 1, len(panels)):
            pairs = [(panels[i][k]["verdict"], panels[j][k]["verdict"]) for k in keys]
            po, kap = cohen_kappa(pairs, VERDICTS)
            kappas[f"judge{i}_vs_judge{j}"] = {"agreement": round(po, 3), "kappa": round(kap, 3)}
    with open(os.path.join(out_dir, "gold_audit_jury_agreement.json"), "w", encoding="utf-8") as f:
        json.dump(kappas, f, indent=2)
    print(json.dumps(kappas, indent=2))
    print("[jury] caveat: correlated judge errors can undermine panels (arXiv 2605.29800) — "
          "report the pairwise agreement alongside the majority verdicts.")


def human_agreement(human_csv, out_dir):
    rows = [r for r in csv.DictReader(open(human_csv, encoding="utf-8"))
            if r.get("verdict") in VERDICTS and (r.get("human_verdict") or "").strip().upper() in VERDICTS]
    if not rows:
        raise SystemExit(f"[agreement] no rows with verdict + human_verdict in {human_csv}")
    pairs = [(r["verdict"], r["human_verdict"].strip().upper()) for r in rows]
    po, kap = cohen_kappa(pairs, VERDICTS)
    res = {"n_labeled": len(rows), "percent_agreement": round(100 * po, 1),
           "cohens_kappa": round(kap, 3),
           "gate": "PASS (>=0.7)" if kap >= 0.7 else "FAIL (<0.7 -> revise rubric / do not trust the full audit)"}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gold_audit_agreement.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--gt_kg")
    ap.add_argument("--src_text")
    ap.add_argument("--backend", choices=["openai", "anthropic", "hf"])
    ap.add_argument("--model")
    ap.add_argument("--output_dir")
    ap.add_argument("--sample", type=int, default=0, help="fixed-seed random validation sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--human_csv", help="agreement mode: gold_audit.csv copy with human_verdict filled")
    ap.add_argument("--merge", nargs="*", help="jury mode: merge N single-judge output dirs (majority)")
    args = ap.parse_args()

    if args.human_csv:
        human_agreement(args.human_csv, args.output_dir or os.path.dirname(args.human_csv) or ".")
        return
    if args.merge:
        if not args.output_dir:
            raise SystemExit("--merge needs --output_dir")
        merge_jury(args.merge, args.output_dir)
        return
    if not (args.backend and args.model and (args.dataset or (args.gt_kg and args.src_text))):
        raise SystemExit("need --backend/--model + --dataset (or --gt_kg/--src_text); "
                         "or --human_csv / --merge modes")

    gt_kg = args.gt_kg or GT_CONFIG.get(args.dataset) or f"./soda/evaluate/references/{args.dataset}.txt"
    src_text = args.src_text or f"./datasets/{args.dataset}.txt"
    out_dir = args.output_dir or f"./output/gold_audit_{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "gold_audit_log.jsonl")

    gold = load_kg_txt(gt_kg)
    src = load_src(src_text)
    n = min(len(gold), len(src))
    items = []
    for i in range(n):
        for t in (gold[i] or []):
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                items.append({"idx": i, "h": str(t[0]), "r": str(t[1]), "t": str(t[2]), "sent": src[i]})
    if args.sample and args.sample < len(items):
        items = random.Random(args.seed).sample(items, args.sample)
        print(f"[sample] {len(items)} triples, uniform random (seed={args.seed})")
    print(f"[audit] {len(items)} gold triples ({args.backend}:{args.model})")

    done = {}
    if os.path.exists(log_path):
        for line in open(log_path, encoding="utf-8"):
            try:
                r = json.loads(line); done[(str(r["idx"]), r["h"], r["r"], r["t"])] = r
            except Exception:
                pass
        print(f"[resume] {len(done)} already audited")

    judge = make_judge(args.backend, args.model)
    rows = []
    with open(log_path, "a", encoding="utf-8") as flog:
        for k, it in enumerate(items):
            key = (str(it["idx"]), it["h"], it["r"], it["t"])
            if key in done:
                rows.append(done[key]); continue
            try:
                out = judge(build_messages(it["sent"], it["h"], it["r"], it["t"]))
            except Exception as e:
                out = ""
                print(f"  [err] idx={it['idx']} {str(e)[:80]}")
            q1, q2, q3, verdict, reason = parse_answer(out)
            rec = {"idx": it["idx"], "h": it["h"], "r": it["r"], "t": it["t"],
                   "q1": q1 or "?", "q2": q2 or "?", "q3": q3 or "?",
                   "verdict": verdict or "UNPARSED", "reason": reason}
            flog.write(json.dumps(rec, ensure_ascii=False) + "\n"); flog.flush()
            rows.append(rec)
            if k % 50 == 0:
                print(f"  ... {k}/{len(items)}")
    write_outputs(rows, out_dir)


if __name__ == "__main__":
    main()
