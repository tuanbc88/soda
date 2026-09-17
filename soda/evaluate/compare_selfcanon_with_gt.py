#!/usr/bin/env python3
# ============================================================
# Compare Self Canonicalized relations with GT relations
# Cosine retrieval + LLM rerank (vLLM / HF)
#
# Author: TuanBC
# ============================================================

import csv
import argparse
from collections import defaultdict
from typing import List

import torch
from sentence_transformers import SentenceTransformer, util


# ============================================================
# Load helpers
# ============================================================
def load_gt_relations(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "relation": r["relation"],
                "definition": r.get("definition", "")
            })
    return rows


def load_schema_relations(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("in_data", "").lower() == "true":
                rows.append({
                    "relation": r["relation"],
                    "definition": r.get("definition", "")
                })
    return rows


# ============================================================
# Cosine retrieval
# ============================================================
def cosine_retrieval(
    gt_relations,
    schema_relations,
    embedder_name,
    top_k
):
    model = SentenceTransformer(embedder_name)

    gt_texts = [f"{r['relation']}: {r['definition']}" for r in gt_relations]
    sc_texts = [f"{r['relation']}: {r['definition']}" for r in schema_relations]

    gt_emb = model.encode(gt_texts, convert_to_tensor=True)
    sc_emb = model.encode(sc_texts, convert_to_tensor=True)

    sim = util.cos_sim(gt_emb, sc_emb)

    rows = []

    for i, gt in enumerate(gt_relations):
        sims = sim[i]
        k = min(top_k, sims.size(0))
        topk = sims.topk(k)

        for rank, (idx, score) in enumerate(
            zip(topk.indices, topk.values), start=1
        ):
            rows.append({
                "gold_relation": gt["relation"],
                "extract_relation": schema_relations[idx]["relation"],
                "score_cosine": round(score.item(), 4),
                "score_cosine_rank": rank,
                "llm_rerank": "",
                "llm_confidence": "",
            })

    return rows


# ============================================================
# LLM backends
# ============================================================
def load_llm_backend(backend, model_name):
    if backend == "hf":
        from transformers import pipeline
        return pipeline(
            "text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=128,
        )

    if backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=model_name)
        params = SamplingParams(
            temperature=0.0,
            max_tokens=128,
        )
        return llm, params

    return None


def llm_score_relation(llm, backend, prompt):
    if backend == "hf":
        out = llm(prompt)[0]["generated_text"]
        return out

    if backend == "vllm":
        llm_model, params = llm
        out = llm_model.generate([prompt], params)[0].outputs[0].text
        return out

    raise ValueError("Unsupported backend")


# ============================================================
# LLM rerank
# ============================================================
def llm_rerank(
    rows,
    backend,
    model_name,
):
    if backend == "none":
        # fallback: cosine top-1
        for r in rows:
            r["llm_rerank"] = 1 if r["score_cosine_rank"] == 1 else 0
            r["llm_confidence"] = r["score_cosine"]
        return rows

    print(f"[INFO] Loading LLM backend={backend}, model={model_name}")
    llm = load_llm_backend(backend, model_name)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["gold_relation"]].append(r)

    for gold, cands in grouped.items():
        prompt = (
            f"Gold relation: {gold}\n\n"
            f"Candidate relations:\n"
        )
        for i, c in enumerate(cands, 1):
            prompt += f"{i}. {c['extract_relation']}\n"

        prompt += (
            "\nQuestion: Which candidate best matches the gold relation?\n"
            "Answer with format: index, confidence (0-1)\n"
        )

        try:
            out = llm_score_relation(llm, backend, prompt)
            # simple parse: assume "1, 0.92"
            idx, conf = out.strip().split(",")
            best = int(idx)

        except Exception:
            best = 1
            conf = 0.0

        for i, c in enumerate(cands, 1):
            if i == best:
                c["llm_rerank"] = 1
                c["llm_confidence"] = float(conf)
            else:
                c["llm_rerank"] = 0
                c["llm_confidence"] = ""

    return rows


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Self Canon schema with GT relations"
    )

    parser.add_argument("--gt_csv", required=True)
    parser.add_argument("--schema_csv", required=True)
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output", required=True)

    # LLM
    parser.add_argument("--llm_backend", choices=["none", "hf", "vllm"], default="none")
    parser.add_argument("--llm_model", default="")

    args = parser.parse_args()

    gt = load_gt_relations(args.gt_csv)
    schema = load_schema_relations(args.schema_csv)

    rows = cosine_retrieval(
        gt,
        schema,
        args.embedder,
        args.top_k,
    )

    rows = llm_rerank(
        rows,
        backend=args.llm_backend,
        model_name=args.llm_model,
    )

    # add STT
    final = []
    stt = 0
    last = None
    for r in rows:
        if r["gold_relation"] != last:
            stt += 1
            last = r["gold_relation"]
        r["stt"] = stt
        final.append(r)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stt",
                "gold_relation",
                "extract_relation",
                "score_cosine",
                "score_cosine_rank",
                "llm_rerank",
                "llm_confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(final)

    print("================================")
    print("Self Canon vs GT comparison done")
    print(f"GT relations     : {len(gt)}")
    print(f"Schema relations : {len(schema)}")
    print(f"Output           : {args.output}")
    print("================================")
