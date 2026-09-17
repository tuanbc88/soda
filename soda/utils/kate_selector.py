"""KATE demo selector for OIE — dynamic per-input few-shot (DECISIONS 2026-07-03b/c).

Implements kNN demonstration selection (KATE, Liu et al. 2022 "What Makes Good
In-Context Examples for GPT-3?") for the OIE stage: for each input text, retrieve
the k most similar (text, gold-triples) pairs from a TRAIN-split pool and render
them in EXACTLY the static few-shot file format (only the demo CONTENT changes;
count, format, and template stay fixed -> FormatSpread-safe minimal change).

STATUS: an explicitly-labeled "dynamic-primed" ablation arm (option A, agreed
2026-07-03): demos carry gold-schema relation names, so this is per-input
vocabulary+format priming. The Mode-1 open headline keeps the static open
few-shot; KATE numbers are reported alongside open + static-primed.
Gate (pre-registered): genuine-miss (C+D) drop >= 1.5pt on rebel.

Pool file: datasets/kate_pool/{ds}_pool.jsonl ({"text":..., "triplets":[[h,r,t],..]}
per line; build with datasets/build_kate_pool.py). Embeddings are cached next to
the pool as <pool>.<embedder>.npy (gitignored).
"""
import os
import json

import numpy as np


class KateSelector:
    def __init__(self, pool_path, embedder_name="BAAI/bge-m3", k=6, device=None):
        from sentence_transformers import SentenceTransformer
        self.k = int(k)
        self.pool = [json.loads(l) for l in open(pool_path, encoding="utf-8") if l.strip()]
        if not self.pool:
            raise ValueError(f"KATE pool is empty: {pool_path}")
        self.model = SentenceTransformer(embedder_name, device=device)
        emb_path = f"{pool_path}.{embedder_name.split('/')[-1]}.npy"
        if os.path.exists(emb_path):
            self.emb = np.load(emb_path)
            if len(self.emb) != len(self.pool):
                raise ValueError(f"stale embedding cache {emb_path} "
                                 f"({len(self.emb)} != {len(self.pool)}); delete it")
        else:
            print(f"[KATE] embedding {len(self.pool)} pool texts (one-time; cached -> {emb_path})")
            self.emb = np.asarray(self.model.encode(
                [p["text"] for p in self.pool], batch_size=64,
                normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
            np.save(emb_path, self.emb)

    def render(self, input_text):
        """Top-k most similar pool demos, rendered in the static-fewshot format."""
        q = np.asarray(self.model.encode([input_text], normalize_embeddings=True),
                       dtype=np.float32)[0]
        top = np.argsort(-(self.emb @ q))[: self.k]
        blocks = []
        for i, idx in enumerate(top):
            p = self.pool[int(idx)]
            trips = [[str(h), str(r), str(t)] for h, r, t in p["triplets"]]
            blocks.append(f"Example {i + 1}:\nText: {p['text']}\nTriplets: {trips!r}")
        return "\n\n".join(blocks)
