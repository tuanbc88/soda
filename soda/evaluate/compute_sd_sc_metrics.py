import json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import csv
import re

# =========================
# Utils
# =========================

def normalize_rel(r: str):
    return r.strip()

def strip_sd_key(k: str):
    # "1. location" -> "location"
    if "." in k:
        return k.split(".", 1)[1].strip()
    return k.strip()


# =========================
# Load data
# =========================
# canon_kg.txt
# def load_canon_kg(path):
#     """
#     Returns list of triplets [(h, r, t), ...]
#     """
#     triples = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line or line == "[]":
#                 continue
#             try:
#                 rows = ast.literal_eval(line)
#                 for h, r, t in rows:
#                     triples.append((h, r, t))
#             except Exception:
#                 continue
#     return triples

def load_canon_kg(path):
    triples = []
    triple_pat = re.compile(r"\[\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*\]")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line == "[]":
                continue

            matches = triple_pat.findall(line)
            for h, r, t in matches:
                triples.append((h, r, t))

    return triples
    
# result_at_each_stage.json
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# schema_canon.csv
def load_schema_canon(path):
    """
    Returns dict:
    {relation: row_dict}
    """
    schema = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row["relation"].strip()
            schema[rel] = row
    return schema

    
# =========================
# 2. Metric từ canon KG (txt)
# =========================
def compute_canon_kg_metrics(triples):
    entities = set()
    relations = Counter()

    for h, r, t in triples:
        entities.add(h)
        entities.add(t)
        relations[r] += 1

    num_triples = len(triples)
    num_entities = len(entities)
    num_relations = len(relations)

    singleton_rate = (
        sum(1 for c in relations.values() if c == 1) / num_relations
        if num_relations else 0.0
    )

    return {
        "num_triples": num_triples,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "avg_triples_per_relation": (
            num_triples / num_relations if num_relations else 0.0
        ),
        "singleton_relation_rate": singleton_rate,
    }


# =========================
# 2. Metric từ canon KG (txt)
# =========================

def compute_schema_kg_metrics(schema, triples):
    kg_relations = Counter(r for _, r, _ in triples)

    orphan = [r for r in schema if r not in kg_relations]

    utilization = {
        r: kg_relations.get(r, 0)
        for r in schema
    }

    return {
        "schema_size": len(schema),
        "orphan_relation_rate": len(orphan) / len(schema) if schema else 0.0,
        "schema_kg_consistency": (
            len(set(kg_relations) & set(schema)) / len(kg_relations)
            if kg_relations else 0.0
        ),
        "utilization": utilization,
    }

# ================================================================================================
# 3. metrics for json file
# ================================================================================================
    
# =========================
# SD Coverage
# =========================
# Instance-level SD coverage = với mỗi sample (index), tỷ lệ relation trong OIE có schema_definition
# vd: 
# ++ Mean = 0.9 → trung bình mỗi câu, 90% relation được define
# ++ Median = 1.0 → ít nhất 50% câu có SD đầy đủ

def compute_sd_coverage(data):
    inst_cov = []
    rel_stats = defaultdict(lambda: {"appear": 0, "defined": 0})

    for item in data:
        oie = item["oie"]
        sd = {strip_sd_key(k) for k in item["schema_definition"].keys()}

        covered = 0
        for _, r, _ in oie:
            r = normalize_rel(r)
            rel_stats[r]["appear"] += 1
            if r in sd:
                covered += 1
                rel_stats[r]["defined"] += 1

        inst_cov.append(covered / len(oie) if oie else 0.0)

    rel_cov = {
        r: v["defined"] / v["appear"]
        for r, v in rel_stats.items()
        if v["appear"] > 0
    }

    return inst_cov, rel_cov


# =========================
# Canonicalization Identity Rate
# =========================
# đo: % relation sau canonicalization vẫn giữ nguyên tên
# vd: 0.333 nghĩa là: chỉ ~33% relation không bị rename / merge. cần chạy lặp thêm

def compute_identity_rate(data):
    total = 0
    identity = 0
    missing_sd_total = 0
    missing_sd_identity = 0

    for item in data:
        oie = item["oie"]
        canon = item["schema_canonicalizaiton"]
        sd = {strip_sd_key(k) for k in item["schema_definition"].keys()}

        for (h, r, t), c in zip(oie, canon):
            total += 1
            if c is not None and c == [h, r, t]:
                identity += 1

            if r not in sd:
                missing_sd_total += 1
                if c is not None and c == [h, r, t]:
                    missing_sd_identity += 1

    return {
        "CIR_all": identity / total if total else 0.0,
        "CIR_missing_SD": (
            missing_sd_identity / missing_sd_total
            if missing_sd_total else 0.0
        ),
    }


# =========================
# Schema Growth & Stability
# =========================

def extract_schema_sets(data):
    """
    Returns list of schema sets per iteration (file order)
    """
    schemas = []
    current_schema = set()

    for item in data:
        canon = item["schema_canonicalizaiton"]
        for c in canon:
            if c is not None:
                current_schema.add(c[1])
        schemas.append(set(current_schema))

    return schemas


def compute_schema_growth(schemas):
    return [len(s) for s in schemas]


def compute_schema_stability(schemas):
    csr = []
    for i in range(len(schemas) - 1):
        if len(schemas[i]) == 0:
            csr.append(1.0)
        else:
            csr.append(
                len(schemas[i] & schemas[i + 1]) / len(schemas[i])
            )
    return csr


# =========================
# Canonical Name Entropy
# =========================

def compute_name_entropy(data):
    cluster_map = defaultdict(list)

    for item in data:
        oie = item["oie"]
        canon = item["schema_canonicalizaiton"]

        for (_, r, _), c in zip(oie, canon):
            if c is not None:
                cluster_map[c[1]].append(r)

    entropy = {}
    for canon_rel, members in cluster_map.items():
        counts = Counter(members)
        total = sum(counts.values())
        probs = [v / total for v in counts.values()]
        entropy[canon_rel] = -sum(p * np.log(p) for p in probs)

    return entropy


# =========================
# MAIN
# =========================

def main(json_path, kg_path=None, schema_path=None, output_path=None):
    print("===== LOADING DATA =====")

    results = {}

    # =====================
    # JSON: SD × SC metrics
    # =====================
    data = load_json(json_path)

    inst_cov, rel_cov = compute_sd_coverage(data)
    cir = compute_identity_rate(data)

    schemas = extract_schema_sets(data)
    growth = compute_schema_growth(schemas)
    csr = compute_schema_stability(schemas)

    entropy = compute_name_entropy(data)

    results["sd_coverage"] = {
        "instance_mean": float(np.mean(inst_cov)),
        "instance_median": float(np.median(inst_cov)),
        "relation_coverage": rel_cov,
    }

    results["sc_robustness"] = cir

    results["schema_dynamics"] = {
        "growth": growth,
        "stability_csr": csr,
    }

    results["canonical_name_entropy"] = entropy

    # -------- print --------
    print("\n===== SD COVERAGE =====")
    print(f"Instance-level mean:   {results['sd_coverage']['instance_mean']:.3f}")
    print(f"Instance-level median: {results['sd_coverage']['instance_median']:.3f}")

    print("\nTop 10 relations with LOWEST SD coverage:")
    for r, v in sorted(rel_cov.items(), key=lambda x: x[1])[:10]:
        print(f"{r:30s} {v:.3f}")

    print("\n===== SC ROBUSTNESS =====")
    for k, v in cir.items():
        print(f"{k}: {v:.3f}")

    # =====================
    # Canonical KG metrics
    # =====================
    if kg_path is not None:
        triples = load_canon_kg(kg_path)
        
        # print("DEBUG: raw KG triples =", len(triples))
        
        kg_metrics = compute_canon_kg_metrics(triples)

        results["canonical_kg"] = kg_metrics

        print("\n===== CANONICAL KG METRICS =====")
        for k, v in kg_metrics.items():
            print(f"{k}: {v}")

    # =====================
    # Schema × KG metrics
    # =====================
    if schema_path is not None and kg_path is not None:
        schema = load_schema_canon(schema_path)
        
        schema_metrics = compute_schema_kg_metrics(schema, triples)

        results["schema_kg"] = schema_metrics

        print("\n===== SCHEMA × KG METRICS (RAW) =====")
        print(f"schema_size: {schema_metrics['schema_size']}")
        print(f"orphan_relation_rate: {schema_metrics['orphan_relation_rate']:.3f}")
        print(f"schema_kg_consistency: {schema_metrics['schema_kg_consistency']:.3f}")

    # =====================
    # Dump JSON (optional)
    # =====================
    if output_path is not None:
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n[✓] Metrics dumped to: {output_path}")



def compute_sd_sc_metrics(
    canon_kg_path: str,
    schema_csv_path: str,
    json_path: str = None,
):
    """
    Library wrapper for SD/SC metrics.
    Returns metrics as dict (no print, no argparse).
    """

    results = {}

    # ---------- Canonical KG ----------
    if canon_kg_path is not None:
        triples = load_canon_kg(canon_kg_path)
        results["canonical_kg"] = compute_canon_kg_metrics(triples)
    else:
        triples = None

    # ---------- Schema × KG ----------
    if schema_csv_path is not None and triples is not None:
        schema = load_schema_canon(schema_csv_path)
        results["schema_kg"] = compute_schema_kg_metrics(schema, triples)

    # ---------- JSON-based metrics (optional) ----------
    if json_path is not None:
        data = load_json(json_path)

        inst_cov, rel_cov = compute_sd_coverage(data)
        cir = compute_identity_rate(data)

        results["sd_coverage"] = {
            "instance_mean": float(np.mean(inst_cov)),
            "instance_median": float(np.median(inst_cov)),
            "relation_coverage": rel_cov,
        }

        results["sc_robustness"] = cir

    return results



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute SD coverage and SC robustness metrics"
    )
    parser.add_argument("--json", required=True, help="EDC output JSON")
    parser.add_argument("--kg", default=None, help="Canonical KG txt file")
    parser.add_argument("--schema", default=None, help="Canonical schema csv")
    parser.add_argument("--output",default=None,  help="Optional output json file")

    args = parser.parse_args()

    main(
        json_path=args.json,
        kg_path=args.kg,
        schema_path=args.schema,
        output_path=args.output,
    )

