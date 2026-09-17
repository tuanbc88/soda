"""Build entity-INSTANCE-canonicalized KGs from an EXISTING SODA run (RQ §8.5).

WHY THIS EXISTS (it saves ~36h of GPU per corpus)
-------------------------------------------------
The IC phase lives inside `extract_kg`, so the obvious way to get instance-canon KGs for hotpot/
musique is to re-run KGC with `--enable_instance_canon`. That would cost ~36h of A100 to recompute
OIE+SD+SC+EC that we ALREADY HAVE — and IC calls no LLM at all, only an embedder.

Everything IC needs is already on disk in the `rq3n100` runs:
    {case}.json                  -> per-sample triplets (saved by save_sc_results)
    sd_total.json                -> SD1 entity descriptions           (ic3 signal)
    entity_canon_{ec_case}.json  -> members + pred_entity_type        (ic2/ic3 signal)
so this rebuilds the IC artifacts offline, identically to what the in-pipeline phase would emit.
Minutes on GPU, ~an hour on CPU (one embedder pass over the unique surfaces).

The in-pipeline phase remains the reference implementation — this reads the SAME
EntityInstanceCanonicalizer, it does not reimplement the merge logic.

Emits, per ic_case, into the run's iter0 dir (never touching canon_kg_{case}.txt — §8.2 scope guard):
    instance_canon_{ic_case}.json         surface -> canonical surface
    instance_canon_stats_{ic_case}.json   merge rate / signals / purity / sample_merges
    canon_kg_instance_{ic_case}_{case}.txt

Usage:
    python scripts/build_instance_canon_kgs.py \
        --run_dir output/hotpot_chunk_selfcanon2_mode1_item_qwen3-8b_bgem3_A100_rq3n100 \
        --ec_case ec2_name_definition --embedder BAAI/bge-m3
"""
import argparse
import json
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soda import entity_instance_canonicalization as eic  # noqa: E402


def load_sc_case(iter_dir, case):
    """Read a saved sc_results case file -> per-sample [[h, r, t], ...].

    NB the SAVED schema is the simplified one (`input` / `pred`), not the in-memory
    `input_triplet` / `pred_relation` — see save_sc_results in edc_framework.
    """
    path = os.path.join(iter_dir, f"{case}.json")
    if not os.path.isfile(path):
        return None
    samples = json.load(open(path, encoding="utf-8"))
    out = []
    for s in samples:
        trips = []
        for t in s.get("triplets", []):
            pred = t.get("pred")
            if pred is None:
                continue
            h, _r, tail = t["input"]
            trips.append([h, pred, tail])
        out.append(trips)
    return out


def load_type_map(iter_dir, ec_case):
    """entity_canon_{ec_case}.json -> {entity_surface: canonical_entity_type}."""
    path = os.path.join(iter_dir, f"entity_canon_{ec_case}.json")
    if not os.path.isfile(path):
        return {}, path
    type_map = {}
    for sample in json.load(open(path, encoding="utf-8")):
        for ent in sample.get("entities", []):
            pred = ent.get("pred_entity_type")
            if not pred:
                continue
            for m in ent.get("members", []):
                type_map[str(m)] = pred
    return type_map, path


def discover_cases(iter_dir):
    """Relation cases present in this run, from canon_schema_{case}.json."""
    cases = []
    for p in sorted(glob.glob(os.path.join(iter_dir, "canon_schema_*.json"))):
        cases.append(os.path.basename(p)[len("canon_schema_"):-len(".json")])
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="an SODA run dir (parent of iter0) or the iter dir itself")
    ap.add_argument("--ec_case", default="ec2_name_definition",
                    help="which EC case supplies the canonical types for ic2/ic3")
    ap.add_argument("--embedder", default="BAAI/bge-m3",
                    help="must match the run's sc_embedder for the signals to be comparable")
    ap.add_argument("--ic_cases", default=None,
                    help="comma-separated subset of %s" % ",".join(eic.IC_ABLATION_CONFIG))
    ap.add_argument("--cases", default=None, help="comma-separated relation cases (default: all present)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: sentence-transformers picks)")
    args = ap.parse_args()

    iter_dir = args.run_dir
    if not os.path.basename(iter_dir).startswith("iter"):
        cand = os.path.join(iter_dir, "iter0")
        if os.path.isdir(cand):
            iter_dir = cand
    if not os.path.isdir(iter_dir):
        sys.exit(f"FATAL: {iter_dir} is not a directory")

    ic_cases = args.ic_cases.split(",") if args.ic_cases else list(eic.IC_ABLATION_CONFIG)
    for c in ic_cases:
        if c not in eic.IC_ABLATION_CONFIG:
            sys.exit(f"FATAL: unknown ic case {c!r}; expected {list(eic.IC_ABLATION_CONFIG)}")

    cases = args.cases.split(",") if args.cases else discover_cases(iter_dir)
    if not cases:
        sys.exit(f"FATAL: no canon_schema_*.json in {iter_dir} — is this an SODA run dir?")

    # ---- load what the in-pipeline phase would have had in memory ----
    triplets_by_case, per_sample_by_case, missing = {}, {}, []
    for case in cases:
        per_sample = load_sc_case(iter_dir, case)
        if per_sample is None:
            missing.append(case)
            continue
        per_sample_by_case[case] = per_sample
        triplets_by_case[case] = [t for trips in per_sample for t in trips]
    if missing:
        print(f"[warn] no sc_results file for {len(missing)} case(s), skipped: {', '.join(missing)}")
    if not triplets_by_case:
        sys.exit("FATAL: no usable case files found")

    type_map, ec_path = load_type_map(iter_dir, args.ec_case)
    if not type_map:
        print(f"⚠️  [IC] no entity types loaded from {ec_path} -> ic2/ic3 have NO type signal and")
        print("⚠️       DEGRADE TO ic1. Their numbers are NOT a valid test of the §8.5 hypothesis")
        print("⚠️       (DECISIONS 2026-07-16b). Was --enable_entity_canon on for this run?")
    else:
        print(f"[IC] {len(type_map)} typed surfaces from {os.path.basename(ec_path)}")

    sd_path = os.path.join(iter_dir, "sd_total.json")
    sd_dict_list = json.load(open(sd_path, encoding="utf-8")) if os.path.isfile(sd_path) else []
    desc_map = eic.collect_entity_descriptions(sd_dict_list)
    if not desc_map:
        print(f"⚠️  [IC] no SD1 descriptions from {sd_path} -> ic3 degrades to ic2.")
    else:
        print(f"[IC] {len(desc_map)} described surfaces from sd_total.json")

    surfaces_freq = eic.collect_surface_freq(triplets_by_case)
    print(f"[IC] {len(surfaces_freq)} unique surfaces over {len(triplets_by_case)} relation cases "
          f"(surfaces are relation-case-invariant; the union is the safe basis)")

    # ---- embed once, canonicalize per ic_case ----
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(args.embedder, **({"device": args.device} if args.device else {}))
    ic = eic.EntityInstanceCanonicalizer(embedder)

    for ic_case in ic_cases:
        canon_map, stats = ic.canonicalize(surfaces_freq, type_map, desc_map, ic_case)
        stats["ec_case_used"] = args.ec_case if type_map else None
        stats["built_by"] = "scripts/build_instance_canon_kgs.py (offline from an existing run)"

        with open(os.path.join(iter_dir, f"instance_canon_{ic_case}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(canon_map, f, indent=2, ensure_ascii=False)
        eic.save_stats(stats, os.path.join(iter_dir, f"instance_canon_stats_{ic_case}.json"))

        from soda.utils import schema_utils
        for case, per_sample in per_sample_by_case.items():
            rewritten = eic.rewrite_triplets_per_sample(per_sample, canon_map)
            schema_utils.dump_canon_kg_txt(
                rewritten, os.path.join(iter_dir, f"canon_kg_instance_{ic_case}_{case}.txt")
            )
        print(f"[IC] {ic_case}: wrote {len(per_sample_by_case)} KGs -> canon_kg_instance_{ic_case}_*.txt")

    print("\n[IC] done. Review the sample_merges in instance_canon_stats_*.json BEFORE trusting any")
    print("     downstream number — the thresholds are UNTUNED (RQ item #10).")


if __name__ == "__main__":
    main()
