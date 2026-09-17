"""Re-type the released Edu-KG with a chosen entity signal (default $\\phi^e{=}2$).

Why
---
`edc_framework` picks the entity-type map for the released graph with
`default_ec = list(ec_results.keys())[0]` -- the FIRST entity case, i.e. name-only ($\\phi^e{=}1$).
That is a default, not a decision, and it costs coverage: name-only leaves 28.5% of entities typed
`Thing` where name+definition leaves noticeably fewer, and the name+definition assignment is already
on disk. This script swaps the typing without touching the relations, the extraction, or the
canonicalization, none of which depend on the entity signal.

It reproduces the pipeline's two rules exactly:
  * surface -> canonical type is built by iterating chunks in order, LAST write wins
    (`entity_type_map_per_case[case][m] = final_type` in the EC loop);
  * an entity takes the type of its FIRST occurrence in the triple stream, falling back to the
    relation schema's head/tail type when the surface is absent from the map
    (`dump_canon_kg_with_entity_types`).

Because the `relations` list in each `canon_kg_with_entity_*.json` was written in that same triple
order, the entity block can be rebuilt from it exactly, with no re-run and no GPU.

Always run `--verify` first: it rebuilds with the ORIGINAL signal and checks the result is identical
to the file on disk. If that fails, the rules above no longer match the pipeline and the re-typing
must not be trusted.

Usage
-----
    python scripts/retype_edukg_release.py --verify                 # gate: reproduce ec1 exactly
    python scripts/retype_edukg_release.py --to ec2_name_definition # write the re-typed graphs
"""
import argparse
import json
import os
import shutil
from collections import Counter

RUN_DIR = "output/edu_kg_core_selfcanon2_mode1_item_qwen2.5-7b_bgem3vni_A100_edukg_release/iter0"
CASES = ["case1_embed_threshold", "case2_name_only", "case3_name_gendef_edc",
         "case4_name_gendef_abstract", "case5_name_detail", "case6_name_detail_headtail",
         "case7_detail_typed", "case8_concat", "case9_weighted"]
ORIGINAL_EC = "ec1_name_only"      # what the pipeline's default_ec resolved to for this run


def build_type_map(run_dir, ec_case):
    """surface -> canonical type, last write wins, mirroring the EC loop."""
    path = os.path.join(run_dir, f"entity_canon_{ec_case}.json")
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    tmap = {}
    for rec in records:
        for group in rec.get("entities", []):
            final_type = group.get("pred_entity_type")
            if final_type is None:
                continue
            for m in group.get("members", []):
                tmap[m] = final_type
    return tmap


def retype(kg, schema, tmap):
    """Rebuild the entity block from the relation stream, first occurrence wins."""
    entities = {}
    for rel in kg["relations"]:
        h, r, t = rel["head"], rel["relation"], rel["tail"]
        rel_schema = schema.get("relation_types", {}).get(r, {})
        if h not in entities:
            entities[h] = {"id": h,
                           "type": tmap.get(h, rel_schema.get("head_type")),
                           "attributes": {}}
        if t not in entities:
            entities[t] = {"id": t,
                           "type": tmap.get(t, rel_schema.get("tail_type")),
                           "attributes": {}}
    return {"entities": list(entities.values()),
            "relations": kg["relations"],
            "attributes": kg.get("attributes", [])}


def load(run_dir, case):
    with open(os.path.join(run_dir, f"canon_kg_with_entity_{case}.json"), encoding="utf-8") as f:
        kg = json.load(f)
    sp = os.path.join(run_dir, f"canon_schema_{case}.json")
    schema = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
    return kg, schema


def thing_stats(kg):
    c = Counter(e.get("type") for e in kg["entities"])
    n = len(kg["entities"])
    return c.get("Thing", 0), n, len(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=RUN_DIR)
    ap.add_argument("--to", default="ec2_name_definition", help="entity signal to type the release with")
    ap.add_argument("--verify", action="store_true",
                    help="rebuild with the ORIGINAL signal and check it reproduces the files on disk")
    ap.add_argument("--no_backup", action="store_true")
    args = ap.parse_args()

    if args.verify:
        tmap = build_type_map(args.run_dir, ORIGINAL_EC)
        print(f"verifying against {ORIGINAL_EC} ({len(tmap)} surfaces mapped)")
        ok = True
        for case in CASES:
            kg, schema = load(args.run_dir, case)
            rebuilt = retype(kg, schema, tmap)
            same = rebuilt["entities"] == kg["entities"]
            ok &= same
            th, n, k = thing_stats(kg)
            print(f"  {case:26} {'MATCH' if same else 'MISMATCH'}   "
                  f"entities={n} Thing={th} ({100*th/n:.1f}%) types={k}")
        print("\nVERIFY:", "PASS -- the rebuild rules match the pipeline" if ok else "FAIL -- do not re-type")
        return

    tmap = build_type_map(args.run_dir, args.to)
    print(f"re-typing with {args.to} ({len(tmap)} surfaces mapped)\n")
    for case in CASES:
        kg, schema = load(args.run_dir, case)
        before = thing_stats(kg)
        rebuilt = retype(kg, schema, tmap)
        after = thing_stats(rebuilt)
        path = os.path.join(args.run_dir, f"canon_kg_with_entity_{case}.json")
        if not args.no_backup:
            bak = path.replace(".json", f".{ORIGINAL_EC}.json")
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rebuilt, f, indent=2, ensure_ascii=False)
        print(f"  {case:26} Thing {before[0]:6d} ({100*before[0]/before[1]:.1f}%) -> "
              f"{after[0]:6d} ({100*after[0]/after[1]:.1f}%)   types {before[2]} -> {after[2]}")
    print(f"\noriginal files kept alongside as *.{ORIGINAL_EC}.json")


if __name__ == "__main__":
    main()
