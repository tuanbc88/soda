"""LLM-judge: classify HOW each gold relation is expressed in its source sentence
(explicit / coref / implicit) — a DIAGNOSIS-ONLY tool (NOT part of the construction
pipeline; it does not touch OIE/SD/SC/EC and does not affect any headline number).

WHY: the lexical implicit-tag in oie_miss_diagnosis.py over-counts (it flags any relation
whose predicate is not surfaced, which conflates schema-name-vs-surface mismatch with true
inference). A strong LLM gives a clean explicit/coref/implicit split, so we can (a) report
the TRUE implicit (inference-required) rate, (b) cross-tab it with the OIE_MISS buckets to
see how much of the "genuine"/ceiling miss is actually a reasoning task (not extraction),
and (c) list the specific record idx for inspection.

JUDGE MODEL: must be STRONG (a small model is not a reliable judge). Matches the project's
precedent (retrieval gold + entity-type gold used Claude/GPT-4 as judge; Qwen3-8B is only the
PIPELINE model, so using a strong judge here does NOT affect the budget-asymmetric claim).
Backends: openai (gpt-4o) | anthropic (claude-sonnet-4-6 / claude-opus-4-8) | hf (large local, e.g. Qwen2.5-72B).
  - ⚠️ A Claude **Pro** subscription is NOT API access — the anthropic/openai backends need a
    pay-per-token API key + credit (console.anthropic.com), billed separately from Pro.
  - 💲 Cost (webnlg ~4k gold triples, ~900 in + ~50 out per call; prompt-cache does NOT apply,
    prefix < 2048 tok): Sonnet 4.6 ~$14 (~$7 via Batches), Opus 4.8 ~$23 (~$11.5 Batches).
  - ✅ RECOMMENDED given a local GPU server: `--backend hf --model Qwen/Qwen2.5-72B-Instruct`
    → $0 API cost, no key. Else Sonnet 4.6 (+Batches) for the best price/quality via API.
Validate against a small human-labeled sample (use --sample 50, ~$0.20) before the full run.

INPUT : gold KG .txt + source text .txt (line-aligned by record index; like oie_miss_diagnosis).
OUTPUT: <out>/relation_expression_judge.csv (idx,h,r,t,label,reason)
        <out>/relation_expression_summary.json (% explicit/coref/implicit; cross-tab with miss)
        <out>/relation_expression_log.jsonl (resume checkpoint)

METHOD / PROTOCOL (named + referenced; DECISIONS 2026-07-02d) — this is an
**LLM-as-annotator** protocol (classifying dataset text against a fixed linguistic
taxonomy), NOT pairwise LLM-as-judge of model outputs:
  1. Fixed 3-label taxonomy with decision rules in the prompt (explicit / coref /
     implicit), mutually exclusive; the explicit/implicit distinction follows the
     implicit-relation IE literature already cited in the paper
     (beckerman2019implicit; stramiglio2025explicit).
  2. **Explain-then-annotate few-shot** (AnnoLLM: He et al., NAACL 2024 Industry,
     arXiv 2303.16854): one anchor demonstration per label, each with its reason;
     the judge must output label + one-clause reason (the reason field doubles as
     an audit trail).
  3. Deterministic annotation: greedy T=0, structured `label:/reason:` output,
     regex-parsed; per-triple resume log (annotation runs are restartable).
  4. **Human-agreement validation gate** (Gilardi et al., PNAS 2023,
     10.1073/pnas.2305016120 — LLM annotation is only trusted after agreement vs
     human labels; Zheng et al., NeurIPS 2023 D&B, arXiv 2306.05685 — judge
     validity is established via agreement): `--sample N` draws a fixed-seed
     RANDOM sample (stratified by OIE_MISS bucket when --miss_csv is given), the
     author labels the same rows (`human_label` column), then `--human_csv`
     computes percent agreement + Cohen's kappa (Cohen 1960) with the
     Landis-Koch 1977 reading; GATE: kappa >= 0.7 ("substantial") before the
     full run is trusted. Mirrors this project's precedent (eduhcmut retrieval
     gold: LLM-judge -> SV human review; entity-type gold: LLM-infer ->
     KB-ground -> human adjudicate).
  5. Bias notes (Zheng et al. 2023): position/verbosity bias do not apply
     (single-label classification, no pairwise comparison, no free-form scoring);
     self-enhancement does not apply (the judge classifies GOLD dataset triples
     against source text and never sees any pipeline output). The judge model is
     deliberately a stronger model than the 8B pipeline model and is
     DIAGNOSIS-only, so the budget-asymmetric claim is unaffected.

RUN (the 3-step protocol):
  # 1. stratified validation sample (seed=42), author labels human_label in the csv copy
  python soda/evaluate/relation_expression_judge.py --dataset webnlg --backend hf \
      --model Qwen/Qwen2.5-72B-Instruct --sample 50 \
      --miss_csv ./output/<run>/iter0/eval_oie_miss/oie_miss_per_triple.csv
  # 2. agreement gate (no GPU): kappa >= 0.7 -> proceed
  python soda/evaluate/relation_expression_judge.py --human_csv <judge_csv copy with human_label> \
      --output_dir <same out dir>
  # 3. full run (resumable)
  OPENAI_KEY=... python soda/evaluate/relation_expression_judge.py --dataset webnlg \
      --backend openai --model gpt-4o --miss_csv .../oie_miss_per_triple.csv
"""
import os
import re
import csv
import sys
import json
import random
import argparse

# allow `python soda/evaluate/relation_expression_judge.py` to import soda.utils.*
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from soda.evaluate.retrieval_recall_metric import GT_CONFIG, load_kg_txt
except ImportError:
    from retrieval_recall_metric import GT_CONFIG, load_kg_txt

LABELS = ("explicit", "coref", "implicit")

SYSTEM = ("You are a careful linguistic annotator for information extraction. You judge HOW a "
          "true (subject, relation, object) fact is expressed in a sentence. Be strict and literal.")

USER_TMPL = """Sentence:
"{sentence}"

The following triple is TRUE for this sentence:
({subject}, {relation}, {object})

Classify HOW the RELATION is expressed in the sentence, with exactly one label:
- explicit : the fact is READABLE from the sentence surface -- via a verb OR any other construction
  (copula "is a", apposition "X, a town in Y", a preposition "in/of/at", a parenthetical "(born 1950)",
  a genitive "France's capital", or a compound "fantasy novel") -- needing NO outside knowledge.
  Both arguments are named. (e.g. "X was born in Y" / "X, a city in France" -> country /
  "X (born 1950)" -> date of birth / "X is a fantasy novel" -> genre : ALL explicit.)
- coref    : the fact is readable from the surface as above, BUT one argument appears ONLY as a
  pronoun or a definite description that must be resolved (e.g. "He died in ...", "the city has ...").
- implicit : the relation is NOT expressed on the surface at all; it can only be obtained by GENUINE
  world-knowledge inference (e.g. "X is a US citizen" -> nationality=American, where "American" is
  not in the text; a relation that is logically entailed but never surface-stated).

Answer in EXACTLY this format, nothing else:
label: <explicit|coref|implicit>
reason: <one short clause>"""

FEWSHOT = [
    ("Liselotte Grschebina was born in Karlsruhe.", "Liselotte_Grschebina", "bornIn", "Karlsruhe",
     "explicit", "the predicate 'was born in' states bornIn"),
    ("Ayam penyet is a food found in Java, a Singaporean island.", "Ayam_penyet", "country", "Singapore",
     "explicit", "the apposition 'a Singaporean island' surface-states the country"),
    ("Alan Shepard held U.S. citizenship.", "Alan_Shepard", "nationality", "United_States",
     "implicit", "'American' never surfaces; nationality is inferred from 'U.S. citizenship'"),
    ("Nie Haisheng was born in Zaoyang. He served as a fighter pilot.", "Nie_Haisheng", "occupation",
     "Fighter_pilot", "coref", "the occupation attaches to the pronoun 'He'"),
]

# ---- CheckEval-style checklist decomposition (Lee et al., EMNLP 2025, arXiv 2403.18771):
# the LLM answers VERIFIABLE binary sub-questions; the LABEL is derived by a
# deterministic rule in code, never scored holistically by the LLM. Decomposed
# binary questions raise cross-evaluator agreement and cut variance vs
# Likert/holistic judging (CheckEval: +0.45 agreement) — and each q1/q2 answer is
# an auditable intermediate claim (Ragas-style granular verification, Es et al.
# EACL 2024 demo; Creanga & Dinu, LoResLM 2026).
USER_TMPL_CHECK = """Sentence:
"{sentence}"

The following triple is TRUE for this sentence:
({subject}, {relation}, {object})

Answer TWO binary questions about HOW the relation "{relation}" is expressed:
Q1: Is the relation between the two arguments READABLE from the sentence surface -- via ANY
    construction (verb, copula, apposition, preposition, parenthetical, genitive, or compound),
    without needing outside world knowledge? Answer no ONLY if the relation is not
    surface-expressed and requires genuine inference.
Q2: Does at least one of the two arguments appear in the sentence ONLY as a pronoun or a
    definite description (i.e. it is never named in the sentence)?

Answer in EXACTLY this format, nothing else:
q1: <yes|no>
q2: <yes|no>
reason: <one short clause>"""

# same anchors, decomposed: (q1, q2) per the rule below
FEWSHOT_CHECK = [
    ("Liselotte Grschebina was born in Karlsruhe.", "Liselotte_Grschebina", "bornIn", "Karlsruhe",
     "yes", "no", "the predicate 'was born in' states bornIn; both arguments are named"),
    ("Ayam penyet is a food found in Java, a Singaporean island.", "Ayam_penyet", "country", "Singapore",
     "yes", "no", "the apposition 'a Singaporean island' surface-states the country"),
    ("Alan Shepard held U.S. citizenship.", "Alan_Shepard", "nationality", "United_States",
     "no", "no", "'American' never surfaces; nationality requires inference from 'U.S. citizenship'"),
    ("Nie Haisheng was born in Zaoyang. He served as a fighter pilot.", "Nie_Haisheng", "occupation",
     "Fighter_pilot", "yes", "yes", "'served as' states the occupation but the subject appears only as 'He'"),
]


def checklist_rule(q1, q2):
    """Deterministic label from the binary answers (the taxonomy, decomposed):
    no predicate stated -> implicit; stated but an argument is pronominalized ->
    coref; stated with named arguments -> explicit."""
    if q1 == "no":
        return "implicit"
    return "coref" if q2 == "yes" else "explicit"


def build_messages(sentence, s, r, o, style="checklist"):
    msgs = [{"role": "system", "content": SYSTEM}]
    if style == "holistic":
        for (sent, ss, rr, oo, lab, rea) in FEWSHOT:
            msgs.append({"role": "user", "content": USER_TMPL.format(sentence=sent, subject=ss, relation=rr, object=oo)})
            msgs.append({"role": "assistant", "content": f"label: {lab}\nreason: {rea}"})
        msgs.append({"role": "user", "content": USER_TMPL.format(sentence=sentence, subject=s, relation=r, object=o)})
    else:
        for (sent, ss, rr, oo, q1, q2, rea) in FEWSHOT_CHECK:
            msgs.append({"role": "user", "content": USER_TMPL_CHECK.format(sentence=sent, subject=ss, relation=rr, object=oo)})
            msgs.append({"role": "assistant", "content": f"q1: {q1}\nq2: {q2}\nreason: {rea}"})
        msgs.append({"role": "user", "content": USER_TMPL_CHECK.format(sentence=sentence, subject=s, relation=r, object=o)})
    return msgs


def parse_label(text, style="checklist"):
    if not text:
        return None, ""
    rm = re.search(r"reason:\s*(.+)", text, re.IGNORECASE)
    reason = rm.group(1).strip()[:200] if rm else ""
    low = text.lower()
    if style == "holistic":
        m = re.search(r"label:\s*(explicit|coref|implicit)", low)
        lab = m.group(1) if m else next((l for l in LABELS if re.search(rf"\b{l}\b", low)), None)
        return lab, reason
    m1 = re.search(r"q1:\s*(yes|no)", low)
    m2 = re.search(r"q2:\s*(yes|no)", low)
    if not m1:
        return None, reason
    lab = checklist_rule(m1.group(1), m2.group(1) if m2 else "no")
    # keep the raw binary answers in the audit trail
    reason = f"[q1={m1.group(1)} q2={m2.group(1) if m2 else '?'}] {reason}"[:200]
    return lab, reason


# ---------------- backends ----------------
def make_judge(backend, model):
    if backend == "openai":
        from soda.utils.llm_utils import openai_chat_completion

        def judge(messages):
            sys_p = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM)
            hist = [m for m in messages if m["role"] != "system"]
            return openai_chat_completion(model, sys_p, hist, temperature=0, max_tokens=120)
        return judge

    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY (API billing; NOT a Claude Pro subscription)

        def judge(messages):
            sys_p = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM)
            hist = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
            # NOTE: do NOT pass temperature — Sonnet 4.6 / Opus 4.8 removed it (400 if sent).
            # Default (no thinking field) = no thinking, which is what we want for cheap classification.
            resp = client.messages.create(model=model, max_tokens=120, system=sys_p, messages=hist)
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return judge

    if backend == "hf":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dt = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        q = os.environ.get("EDC_LOAD_IN_4BIT", "0") == "1"
        kw = dict(device_map="auto", low_cpu_mem_usage=True)
        if q:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True)
        else:
            kw["torch_dtype"] = dt
        mdl = AutoModelForCausalLM.from_pretrained(model, **kw)
        tok = AutoTokenizer.from_pretrained(model)
        try:
            from soda.utils.llm_utils import generate_completion_transformers as gen
        except ImportError:
            from llm_utils import generate_completion_transformers as gen

        def judge(messages):
            return gen(messages, mdl, tok, max_new_tokens=120, temperature=0.0)
        return judge

    raise SystemExit(f"unknown backend: {backend}")


def load_src(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--gt_kg")
    ap.add_argument("--src_text")
    ap.add_argument("--backend", choices=["openai", "anthropic", "hf"])
    ap.add_argument("--model", help="STRONG model (gpt-4o / claude-* / a large HF model)")
    ap.add_argument("--output_dir")
    ap.add_argument("--miss_csv", help="oie_miss_per_triple.csv -> cross-tab label x miss-bucket (+ sample strata)")
    ap.add_argument("--sample", type=int, default=0,
                    help="validation mode: judge a fixed-seed RANDOM sample of N triples "
                         "(stratified by miss-bucket when --miss_csv is given)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (fixed for reproducibility)")
    ap.add_argument("--human_csv",
                    help="agreement mode (no judging): a copy of relation_expression_judge csv with the "
                         "human_label column filled -> percent agreement + Cohen's kappa (gate >= 0.7)")
    ap.add_argument("--style", choices=["checklist", "holistic"], default="checklist",
                    help="checklist = CheckEval-style binary sub-questions + rule-derived label (default); "
                         "holistic = single-shot 3-way label. Run BOTH on the validation sample and pick "
                         "the higher-kappa style for the full run (pre-registered selection rule).")
    args = ap.parse_args()

    if args.human_csv:
        human_agreement(args.human_csv, args.output_dir or os.path.dirname(args.human_csv) or ".")
        return
    if not (args.backend and args.model):
        ap.error("--backend and --model are required (unless running --human_csv agreement mode)")

    gt_kg = args.gt_kg or GT_CONFIG.get(args.dataset) or (f"./soda/evaluate/references/{args.dataset}.txt" if args.dataset else None)
    src_text = args.src_text or (f"./datasets/{args.dataset}.txt" if args.dataset else None)
    out_dir = args.output_dir or (os.path.dirname(args.miss_csv) if args.miss_csv else ".")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"relation_expression_log_{args.style}.jsonl")

    gold = load_kg_txt(gt_kg)
    src = load_src(src_text)
    n = min(len(gold), len(src))

    # flatten gold triples with their record idx + sentence
    items = []
    for i in range(n):
        for t in (gold[i] or []):
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                items.append({"idx": i, "h": str(t[0]), "r": str(t[1]), "t": str(t[2]), "sent": src[i]})
    if args.sample and args.sample < len(items):
        rng = random.Random(args.seed)
        if args.miss_csv and os.path.exists(args.miss_csv):
            # stratify by OIE_MISS bucket so the validation sample covers the
            # categories the cross-tab depends on (proportional, >=1 per stratum)
            cat_of = {}
            for r in csv.DictReader(open(args.miss_csv, encoding="utf-8")):
                cat_of[(int(r["idx"]), r["h"], r["r"], r["t"])] = r["category"]
            strata = {}
            for it in items:
                c = cat_of.get((it["idx"], it["h"], it["r"], it["t"]), "?")
                strata.setdefault(c, []).append(it)
            picked = []
            for c, grp in sorted(strata.items()):
                k = max(1, round(args.sample * len(grp) / float(len(items))))
                picked += rng.sample(grp, min(k, len(grp)))
            rng.shuffle(picked)
            items = picked[:args.sample]
            print(f"[sample] {len(items)} triples, stratified over {len(strata)} miss-buckets (seed={args.seed})")
        else:
            items = rng.sample(items, args.sample)
            print(f"[sample] {len(items)} triples, uniform random (seed={args.seed})")
    print(f"[judge] {len(items)} gold triples to classify ({args.backend}:{args.model})")

    done = {}
    if os.path.exists(log_path):
        for line in open(log_path, encoding="utf-8"):
            try:
                r = json.loads(line); done[(r["idx"], r["h"], r["r"], r["t"])] = r
            except Exception:
                pass
        print(f"[resume] {len(done)} already judged")

    judge = make_judge(args.backend, args.model)
    with open(log_path, "a", encoding="utf-8") as flog:
        for k, it in enumerate(items):
            key = (it["idx"], it["h"], it["r"], it["t"])
            if key in done:
                continue
            try:
                out = judge(build_messages(it["sent"], it["h"], it["r"], it["t"], style=args.style))
            except Exception as e:
                out = ""
                print(f"  [err] idx={it['idx']} {str(e)[:80]}")
            lab, reason = parse_label(out, style=args.style)
            rec = {**key_to_dict(key), "label": lab or "UNPARSED", "reason": reason}
            flog.write(json.dumps(rec, ensure_ascii=False) + "\n"); flog.flush()
            if k % 50 == 0:
                print(f"  ... {k}/{len(items)}")

    finalize(log_path, items, out_dir, args.miss_csv, style=args.style)


def key_to_dict(key):
    return {"idx": key[0], "h": key[1], "r": key[2], "t": key[3]}


def human_agreement(human_csv, out_dir):
    """Agreement mode (validation gate; no GPU): percent agreement + Cohen's kappa
    between the judge `label` and the author-filled `human_label` on the sample.
    Kappa reading per Landis & Koch (1977): >=0.61 substantial, >=0.81 near-perfect.
    GATE: kappa >= 0.7 before the full run is trusted (else: revise the rubric
    few-shots and re-validate, or fall back to reporting the lexical tag only)."""
    rows = [r for r in csv.DictReader(open(human_csv, encoding="utf-8"))
            if r.get("label") in LABELS and (r.get("human_label") or "").strip().lower() in LABELS]
    if not rows:
        raise SystemExit(f"[agreement] no rows with both label and human_label filled in {human_csv}")
    n = len(rows)
    conf = {}
    agree = 0
    for r in rows:
        j, h = r["label"], r["human_label"].strip().lower()
        conf.setdefault(h, {}).setdefault(j, 0)
        conf[h][j] += 1
        agree += int(j == h)
    po = agree / float(n)
    pj = {l: sum(1 for r in rows if r["label"] == l) / float(n) for l in LABELS}
    ph = {l: sum(1 for r in rows if r["human_label"].strip().lower() == l) / float(n) for l in LABELS}
    pe = sum(pj[l] * ph[l] for l in LABELS)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    verdict = ("PASS (>=0.7 substantial -> full run trusted)" if kappa >= 0.7 else
               "FAIL (<0.7 -> revise rubric/few-shots and re-validate, or keep lexical-tag only)")
    result = {"n_labeled": n, "percent_agreement": round(100 * po, 1),
              "cohens_kappa": round(kappa, 3), "gate": verdict,
              "confusion_human_x_judge": conf,
              "note": "kappa reading per Landis & Koch 1977; protocol per Gilardi et al. PNAS 2023 / "
                      "AnnoLLM NAACL 2024 / Zheng et al. NeurIPS 2023 (see module docstring)"}
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "relation_expression_agreement.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[agreement] written -> {out_path}")


def finalize(log_path, items, out_dir, miss_csv, style="checklist"):
    rows = {}
    for line in open(log_path, encoding="utf-8"):
        r = json.loads(line); rows[(r["idx"], r["h"], r["r"], r["t"])] = r
    out = [rows.get((it["idx"], it["h"], it["r"], it["t"]),
                   {**it, "label": "UNJUDGED", "reason": ""}) for it in items]

    # human_label stays empty: the validation protocol copies this csv, the author
    # fills human_label, then --human_csv computes agreement (kappa gate >= 0.7).
    with open(os.path.join(out_dir, f"relation_expression_judge_{style}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "h", "r", "t", "label", "reason", "human_label"]); w.writeheader()
        for r in out:
            row = {k: r.get(k) for k in ["idx", "h", "r", "t", "label", "reason"]}
            row["human_label"] = ""
            w.writerow(row)

    n = len(out)
    dist = {l: sum(1 for r in out if r["label"] == l) for l in LABELS}
    summary = {"n_triples": n, "model_judged": n - sum(1 for r in out if r["label"] in ("UNJUDGED", "UNPARSED")),
               "distribution": dist,
               "distribution_%": {l: round(100.0 * dist[l] / n, 2) if n else 0.0 for l in LABELS}}

    # cross-tab with OIE_MISS bucket (label x category), if available
    if miss_csv and os.path.exists(miss_csv):
        miss = {}
        for r in csv.DictReader(open(miss_csv, encoding="utf-8")):
            miss[(int(r["idx"]), r["h"], r["r"], r["t"])] = r["category"]
        ct = {}
        impl_miss = impl_correct = miss_tot = 0
        for r in out:
            cat = miss.get((r["idx"], r["h"], r["r"], r["t"]))
            if cat is None:
                continue
            ct.setdefault(r["label"], {}).setdefault(cat, 0)
            ct[r["label"]][cat] += 1
            missed = cat != "CORRECT"
            if missed:
                miss_tot += 1
                if r["label"] == "implicit":
                    impl_miss += 1
            elif r["label"] == "implicit":
                impl_correct += 1
        summary["crosstab_label_x_misscategory"] = ct
        summary["implicit_among_missed_%"] = round(100.0 * impl_miss / miss_tot, 2) if miss_tot else None
        summary["implicit_recovered_(correct_despite_implicit)"] = impl_correct

    with open(os.path.join(out_dir, f"relation_expression_summary_{style}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n[judge] distribution %:", summary["distribution_%"])
    if "implicit_among_missed_%" in summary:
        print(f"[judge] TRUE implicit among MISSED triples: {summary['implicit_among_missed_%']}%")
    print(f"Saved -> {out_dir}/relation_expression_judge_{style}.csv + relation_expression_summary_{style}.json")


if __name__ == "__main__":
    main()
