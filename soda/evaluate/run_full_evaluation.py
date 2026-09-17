import os
import ast
import json
import csv
import argparse

from evaluation_script import compute_metrics

# Literal/value canonicalizer (dates -> ISO, numbers -> canonical) applied SYMMETRICALLY
# to pred + gold for a 2nd "literal-normalized" strict/exact/partial table. The delta vs
# the raw table = the "literal-format gap" (RQ2 assess asset; DECISIONS 2026-06-24).
try:
    from literal_normalize import normalize_triplet
except ImportError:
    from soda.evaluate.literal_normalize import normalize_triplet

# Lazy/optional: only needed for Table 2 (schema-precision + SBERT redundancy).
# Table 1 (strict/exact/partial) works without it.
try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_SBERT = True
except Exception:
    SentenceTransformer = None; util = None
    _HAS_SBERT = False
from sklearn.cluster import AgglomerativeClustering


# =========================
# CONFIG
# =========================

CASES = [
    "case1_embed_threshold",
    "case2_name_only",
    "case3_name_gendef_edc",
    "case4_name_gendef_abstract",
    "case5_name_detail",
    "case6_name_detail_headtail",
    "case7_detail_typed",
    "case8_concat",
    "case9_weighted",
]


# =========================
# LOADERS
# =========================

def load_kg_txt(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            triples = ast.literal_eval(line.strip())
            data.append({"triplets": triples})

    return data


def load_schema_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gt_schema_csv(path):
    # Reference schema CSVs are headerless: column 0 = relation name,
    # column 1 = general definition. (Earlier this used DictReader expecting a
    # "relation" header, which silently returned an empty set.)
    rels = set()

    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                rels.add(row[0].strip())

    return rels


# =========================
# UTILS
# =========================

def get_relations_from_schema(schema):
    return list(schema.get("relation_types", {}).keys())


def normalize_rel(r):
    return r.lower().replace("_", "").replace(" ", "")


# =========================
# CLOSED-SCHEMA
# =========================

def normalize_kg(data):
    """Apply the literal canonicalizer to every triplet in a loaded KG (list of
    {"triplets": [...]}). Used to build the symmetric normalized pred + gold."""
    return [{"triplets": [normalize_triplet(t) for t in item.get("triplets", [])]}
            for item in data]


def evaluate_case(pred_data, gold_data, log_dir=None):

    exact_all = []
    strict_all = []
    partial_all = []

    total = len(pred_data)

    for i, (pred_item, gold_item) in enumerate(zip(pred_data, gold_data)):

        pred = pred_item.get("triplets", [])
        gold = gold_item.get("triplets", [])

        try:
            exact, strict, partial = compute_metrics(
                pred,
                gold,
                idx=i,
                log_dir=log_dir
            )

        except Exception as e:

            if log_dir is not None:

                err_dir = os.path.join(log_dir, "eval_logs")
                os.makedirs(err_dir, exist_ok=True)

                with open(
                    os.path.join(err_dir, f"{i}_FATAL.json"),
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump({
                        "idx": i,
                        "error": str(e),
                        "pred": pred,
                        "gold": gold
                    }, f, indent=2, ensure_ascii=False)

            exact, strict, partial = (
                {"f1": 0.0},
                {"f1": 0.0},
                {"f1": 0.0}
            )

        exact_all.append(exact["f1"])
        strict_all.append(strict["f1"])
        partial_all.append(partial["f1"])

        if i % 50 == 0:
            print(f"[Eval] {i}/{total}")

    if len(exact_all) == 0:
        return {
            "exact": 0.0,
            "strict": 0.0,
            "partial": 0.0,
        }

    return {
        "exact": sum(exact_all) / len(exact_all),
        "strict": sum(strict_all) / len(strict_all),
        "partial": sum(partial_all) / len(partial_all),
    }


# =========================
# OPEN-SCHEMA
# =========================

def compute_num_rel(schema):
    return len(set(get_relations_from_schema(schema)))


def compute_redundancy_lexical(schema):

    rels = get_relations_from_schema(schema)

    if len(rels) < 2:
        return 0.0

    norm = [normalize_rel(r) for r in rels]

    redundant = 0
    total = 0

    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):

            total += 1

            if (
                norm[i] == norm[j]
                or norm[i] in norm[j]
                or norm[j] in norm[i]
            ):
                redundant += 1

    return redundant / total if total else 0.0


def compute_redundancy_sbert(schema, model):

    rels = get_relations_from_schema(schema)

    if len(rels) < 2:
        return 0.0

    texts = []

    rel_dict = schema.get("relation_types", {})

    for r in rels:

        definition = rel_dict.get(r, {}).get("definition", "")

        texts.append(f"{r}: {definition}")

    emb = model.encode(texts, convert_to_tensor=True)

    sim = util.cos_sim(emb, emb)

    total = 0.0

    for i in range(len(rels)):

        best = max(
            sim[i][j].item()
            for j in range(len(rels))
            if i != j
        )

        total += best

    return total / len(rels)


def compute_redundancy_cluster(
    schema,
    model,
    threshold=0.8
):

    rels = get_relations_from_schema(schema)

    if len(rels) < 2:
        return 0.0

    texts = []

    rel_dict = schema.get("relation_types", {})

    for r in rels:

        definition = rel_dict.get(r, {}).get("definition", "")

        texts.append(f"{r}: {definition}")

    emb = model.encode(texts)

    clustering = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=1 - threshold,
        n_clusters=None
    )

    labels = clustering.fit_predict(emb)

    num_clusters = len(set(labels))

    return 1 - num_clusters / len(rels)


def compute_precision(
    pred_schema,
    gt_rels,
    normalize_flag=False
):

    pred = get_relations_from_schema(pred_schema)

    if normalize_flag:
        pred_set = set(normalize_rel(r) for r in pred)
        gt_set = set(normalize_rel(r) for r in gt_rels)
    else:
        pred_set = set(pred)
        gt_set = set(gt_rels)

    if not pred_set:
        return 0.0

    return len(pred_set & gt_set) / len(pred_set)


# =========================
# MAIN
# =========================

def run_evaluation(
    gt_kg,
    gt_schema_csv,
    pred_dir,
    output_dir
):

    os.makedirs(output_dir, exist_ok=True)

    print("======================================")
    print(">>> RUN EVALUATION")
    print("======================================")

    print(f"GT KG       : {gt_kg}")
    print(f"GT SCHEMA   : {gt_schema_csv}")
    print(f"PRED DIR    : {pred_dir}")
    print(f"OUTPUT DIR  : {output_dir}")

    print("======================================")

    gold_data = load_kg_txt(gt_kg)
    gold_data_norm = normalize_kg(gold_data)   # for the literal-normalized table

    gt_schema_rels = load_gt_schema_csv(gt_schema_csv)

    if _HAS_SBERT:
        # Table 2 redundancy embeds only the schema relation texts (~150 strings) -> tiny.
        # Default to CPU so it NEVER contends with a GPU still held by the pipeline/another
        # run (that contention surfaces as CUBLAS_STATUS_NOT_INITIALIZED). Override with
        # EDC_EVAL_SBERT_DEVICE=cuda if the GPU is free and you want speed.
        sbert_device = os.environ.get("EDC_EVAL_SBERT_DEVICE", "cpu")
        print(f"Loading SBERT model (device={sbert_device})...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=sbert_device)
    else:
        print("[warn] sentence_transformers not available -> Table 2 (schema "
              "precision/redundancy) skipped; Table 1 strict/exact/partial still computed.")
        model = None

    table1 = []
    table1_norm = []
    table1_gap = []
    table2 = []

    for case in CASES:

        print(f"\nRunning {case}...")

        kg_path = os.path.join(
            pred_dir,
            f"canon_kg_{case}.txt"
        )

        schema_path = os.path.join(
            pred_dir,
            f"canon_schema_{case}.json"
        )

        if not os.path.exists(kg_path):
            print(f"Missing: {kg_path}")
            continue

        if not os.path.exists(schema_path):
            print(f"Missing: {schema_path}")
            continue

        pred_data = load_kg_txt(kg_path)

        pred_schema = load_schema_json(schema_path)

        # =====================
        # TABLE 1
        # =====================

        scores = evaluate_case(
            pred_data,
            gold_data,
            output_dir
        )

        table1.append({
            "case": case,
            "partial": round(scores["partial"], 4),
            "strict": round(scores["strict"], 4),
            "exact": round(scores["exact"], 4),
        })

        # =====================
        # TABLE 1 (literal-normalized) — same scorer, dates/numbers canonicalized
        # symmetrically on pred + gold. Reported alongside raw; delta = literal-format gap.
        # =====================
        pred_data_norm = normalize_kg(pred_data)

        scores_norm = evaluate_case(
            pred_data_norm,
            gold_data_norm,
            None        # no eval_logs for the normalized pass (keep the dir clean)
        )

        table1_norm.append({
            "case": case,
            "partial": round(scores_norm["partial"], 4),
            "strict": round(scores_norm["strict"], 4),
            "exact": round(scores_norm["exact"], 4),
        })

        table1_gap.append({
            "case": case,
            "partial_raw": round(scores["partial"], 4),
            "partial_norm": round(scores_norm["partial"], 4),
            "partial_gap": round(scores_norm["partial"] - scores["partial"], 4),
            "strict_raw": round(scores["strict"], 4),
            "strict_norm": round(scores_norm["strict"], 4),
            "strict_gap": round(scores_norm["strict"] - scores["strict"], 4),
            "exact_raw": round(scores["exact"], 4),
            "exact_norm": round(scores_norm["exact"], 4),
            "exact_gap": round(scores_norm["exact"] - scores["exact"], 4),
        })

        # =====================
        # TABLE 2 (needs SBERT model; skipped if unavailable). Non-fatal: a GPU/SBERT
        # error here must NOT lose the Table-1 raw/normalized/gap CSVs (written after the loop).
        # =====================
        if model is None:
            continue

        try:
            table2.append({
                "case": case,

                "precision_strict":
                    round(
                        compute_precision(
                            pred_schema,
                            gt_schema_rels,
                            False
                        ),
                        4
                    ),

                "precision_norm":
                    round(
                        compute_precision(
                            pred_schema,
                            gt_schema_rels,
                            True
                        ),
                        4
                    ),

                "num_relations":
                    compute_num_rel(pred_schema),

                "redundancy_lexical":
                    round(
                        compute_redundancy_lexical(pred_schema),
                        4
                    ),

                "redundancy_sbert":
                    round(
                        compute_redundancy_sbert(
                            pred_schema,
                            model
                        ),
                        4
                    ),

                "redundancy_cluster":
                    round(
                        compute_redundancy_cluster(
                            pred_schema,
                            model
                        ),
                        4
                    ),
            })
        except Exception as e:
            print(f"[warn] Table 2 (schema redundancy) failed for {case}: {e}\n"
                  f"       -> skipping Table 2 for this case; Table 1 (raw/normalized/gap) is unaffected.")

    # =========================
    # SAVE
    # =========================

    t1_path = os.path.join(
        output_dir,
        "table1_closed_schema.csv"
    )

    t2_path = os.path.join(
        output_dir,
        "table2_open_schema.csv"
    )

    if table1:

        with open(
            t1_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=table1[0].keys()
            )

            writer.writeheader()
            writer.writerows(table1)

    if table2:

        with open(
            t2_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=table2[0].keys()
            )

            writer.writeheader()
            writer.writerows(table2)

    # literal-normalized table + the raw-vs-norm gap (the "literal-format gap" asset)
    t1n_path = os.path.join(output_dir, "table1_closed_schema_normalized.csv")
    gap_path = os.path.join(output_dir, "table1_literal_gap.csv")

    if table1_norm:
        with open(t1n_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=table1_norm[0].keys())
            writer.writeheader()
            writer.writerows(table1_norm)

    if table1_gap:
        with open(gap_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=table1_gap[0].keys())
            writer.writeheader()
            writer.writerows(table1_gap)

    print("\n======================================")
    print("DONE")
    print("======================================")

    print(t1_path)
    print(t1n_path)
    print(gap_path)
    print(t2_path)


# =========================
# ENTRY
# =========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # =====================
    # QUICK MODE
    # =====================

    parser.add_argument("--dataset")
    parser.add_argument("--method")
    parser.add_argument("--iter")

    # =====================
    # MANUAL OVERRIDE
    # =====================

    parser.add_argument("--gt_kg")
    parser.add_argument("--gt_schema_csv")
    parser.add_argument("--pred_dir")
    parser.add_argument("--output_dir")

    args = parser.parse_args()

    # =========================================
    # AUTO BUILD PATHS
    # =========================================

    if args.dataset and args.method and args.iter:

        # datasets with a gold KG + reference schema CSV (intrinsic eval)
        GT_CONFIG = {
            "webnlg":   ("./soda/evaluate/references/webnlg.txt",   "./schemas/webnlg_schema.csv"),
            "rebel":    ("./soda/evaluate/references/rebel.txt",    "./schemas/rebel_schema.csv"),
            "wiki-nre": ("./soda/evaluate/references/wiki-nre.txt", "./schemas/wiki-nre_schema.csv"),
        }

        if args.dataset in GT_CONFIG:
            gt_kg, gt_schema_csv = GT_CONFIG[args.dataset]
        else:
            raise ValueError(
                f"No default GT config for dataset={args.dataset} "
                f"(known: {list(GT_CONFIG)})"
            )

        pred_dir = (
            f"./output/"
            f"{args.dataset}_{args.method}/"
            f"{args.iter}"
        )

        output_dir = os.path.join(
            pred_dir,
            "eval"
        )

    else:

        gt_kg = args.gt_kg
        gt_schema_csv = args.gt_schema_csv
        pred_dir = args.pred_dir
        output_dir = args.output_dir

    run_evaluation(
        gt_kg=gt_kg,
        gt_schema_csv=gt_schema_csv,
        pred_dir=pred_dir,
        output_dir=output_dir
    )