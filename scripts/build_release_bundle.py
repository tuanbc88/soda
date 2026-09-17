"""Assemble the data-deposit bundle for the KBS submission (Zenodo / Mendeley Data).

KBS applies Elsevier's research-data Option C, under which a dataset that supports the paper is
deposited in a persistent repository and cited in the reference list with a [dataset] tag. A GitHub
URL does not satisfy "persistent"; a DOI does. This script collects exactly what the manuscript's
Data availability statement promises, so the author can upload one folder without hunting for files.

Deliberately NOT included:
  * the raw Vietnamese-education corpus, which the manuscript states is not released in full under
    institutional terms;
  * the Tier-A chunk shards, which are already public in the code repository and whose release
    terms are the author's call rather than a mechanical one;
  * code, which is cited by repository URL and does not belong in a data deposit.

Usage
-----
    python scripts/build_release_bundle.py            # writes assets/release_zenodo/
"""
import argparse
import json
import os
import shutil

EDU_RUN = "output/edu_kg_core_selfcanon2_mode1_item_qwen2.5-7b_bgem3vni_A100_edukg_release/iter0"

EDU_FILES = [
    # (source, name in the bundle, what it is)
    ("canon_kg_with_entity_case8_concat.json", "edu_kg_tierA_graph.json",
     "The released Edu-KG: 33,498 typed entities and their relations, relation signal phi=8, "
     "entity signal phi^e=2."),
    ("canon_schema_case8_concat.json", "edu_kg_relation_schema.json",
     "The discovered relation schema for that graph: type names, definitions and alias lists."),
    ("canon_entity_schema_ec2_name_definition.json", "edu_kg_entity_schema.json",
     "The discovered entity-type schema (phi^e=2), with definitions, parents and alias lists."),
]

SCHEMA_FILES = [
    ("gold_entity_types_webnlg.json", "typed entity-schema layer, WebNLG"),
    ("gold_entity_types_webnlg_full.json", "typed entity-schema layer, WebNLG full split"),
    ("gold_entity_types_rebel.json", "typed entity-schema layer, REBEL"),
    ("gold_entity_types_wiki-nre.json", "typed entity-schema layer, Wiki-NRE"),
    ("gold_entity_types_overrides.json", "human adjudications applied on top of KB grounding"),
    ("entity_type_wikidata_map.json", "schema type -> Wikidata QID / DBpedia class map"),
    ("gold_entity_p31_webnlg.json", "per-entity P31 identity gold, WebNLG"),
    ("gold_entity_p31_rebel.json", "per-entity P31 identity gold, REBEL"),
    ("gold_entity_p31_wiki-nre.json", "per-entity P31 identity gold, Wiki-NRE"),
]

DOC_FILES = [
    ("schemas/ENTITY_TYPE_SCHEMA.md", "how the typed entity-schema layer was built and adjudicated"),
    ("EDU_KG_PROVENANCE.md", "Edu-KG corpus provenance, tiering, and PII handling"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/release_zenodo")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "edu_kg"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "entity_type_layer"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "docs"), exist_ok=True)

    manifest = []
    for src, dst, desc in EDU_FILES:
        s = os.path.join(EDU_RUN, src)
        d = os.path.join(args.out, "edu_kg", dst)
        shutil.copy2(s, d)
        manifest.append(("edu_kg/" + dst, os.path.getsize(d), desc, s))
    for name, desc in SCHEMA_FILES:
        s = os.path.join("schemas", name)
        d = os.path.join(args.out, "entity_type_layer", name)
        shutil.copy2(s, d)
        manifest.append(("entity_type_layer/" + name, os.path.getsize(d), desc, s))
    for s, desc in DOC_FILES:
        d = os.path.join(args.out, "docs", os.path.basename(s))
        shutil.copy2(s, d)
        manifest.append(("docs/" + os.path.basename(s), os.path.getsize(d), desc, s))

    total = sum(m[1] for m in manifest)
    with open(os.path.join(args.out, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump([{"path": p, "bytes": b, "description": d, "source_in_repo": s}
                   for p, b, d, s in manifest], f, indent=2, ensure_ascii=False)

    for p, b, d, _ in manifest:
        print(f"  {b:>10,}  {p}")
    print(f"\n{len(manifest)} files, {total/1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
