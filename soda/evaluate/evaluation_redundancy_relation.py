#!/usr/bin/env python3
# ============================================================
# Compute Redundancy of Relation (RoR) for EDC
# Based on EMNLP 2024 EDC paper
# Author: TuanBC + ChatGPT
# ============================================================

import csv
import json
import argparse
from sentence_transformers import SentenceTransformer, util


def compute_relation_redundancy_semantic(
    schema_csv_path: str,
    embedder_name: str,
):
    """
    Compute Redundancy of Relation (semantic):
    Average cosine similarity between each relation and its nearest neighbor.
    """

    # ---------- Load schema ----------
    all_relations = []
    used_relations = []
    texts = []
    
    with open(schema_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            relation = row.get("relation") or row.get("name")
            definition = row.get("definition", "")
            in_data = row.get("in_data", "").lower() == "true"
    
            if not relation:
                continue
    
            # count total schema size
            all_relations.append(relation)
    
            # only use grounded relations for redundancy
            if in_data:
                used_relations.append(relation)
                texts.append(f"{relation}: {definition}")

    relations =  used_relations 
    if len(relations) < 2:
        return {
            "redundancy_semantic": 0.0,
            "num_relations": len(relations),
            "per_relation_best_sim": {},
        }

    # ---------- Embedding ----------
    model = SentenceTransformer(embedder_name)
    embeddings = model.encode(texts, convert_to_tensor=True)

    # ---------- Cosine similarity ----------
    sim_matrix = util.cos_sim(embeddings, embeddings)

    per_relation_best_sim = {}
    total = 0.0

    for i, rel in enumerate(relations):
        best_sim = -1.0
        for j in range(len(relations)):
            if i == j:
                continue
            score = sim_matrix[i][j].item()
            if score > best_sim:
                best_sim = score

        per_relation_best_sim[rel] = round(best_sim, 4)
        total += best_sim

    avg_redundancy = total / len(relations)

    return {
        "redundancy_semantic": round(avg_redundancy, 4),
        "num_relations": len(relations),
        "per_relation_best_sim": per_relation_best_sim,
    }


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Redundancy of Relation (semantic) for EDC schema"
    )
    parser.add_argument(
        "--schema_csv",
        required=True,
        help="Path to schema_dump.csv or schema_dump_with_coverage.csv",
    )
    parser.add_argument(
        "--embedder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON file",
    )

    args = parser.parse_args()

    result = compute_relation_redundancy_semantic(
        schema_csv_path=args.schema_csv,
        embedder_name=args.embedder,
    )

    print("================================")
    print("Redundancy of Relation (Semantic)")
    print("--------------------------------")
    print(f"Schema file   : {args.schema_csv}")
    print(f"Embedder      : {args.embedder}")
    print(f"#Relations    : {result['num_relations']}")
    print(f"RoR (avg cos) : {result['redundancy_semantic']}")
    print("================================")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        print(f"Result saved to: {args.output}")
