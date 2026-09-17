import json
import os
import re
import logging

logger = logging.getLogger(__name__)


def load_json(path):

    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    return {}


def save_json(data, path):

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_schema(schema_path):

    import json

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    if "relations" not in schema:
        schema["relations"] = {}

    if "entities" not in schema:
        schema["entities"] = {}

    return schema


def merge_relation_schema__local(old_schema, new_schema):

    for r, info in new_schema.items():

        if r not in old_schema:
            old_schema[r] = info
            continue

        # merge aliases
        old_alias = set(old_schema[r].get("aliases", []))
        new_alias = set(info.get("aliases", []))

        old_schema[r]["aliases"] = list(old_alias | new_alias)

        # merge examples
        old_examples = old_schema[r].get("examples", [])
        new_examples = info.get("examples", [])

        old_schema[r]["examples"] = old_examples + new_examples

    return old_schema


def merge_entity_schema__local(old_schema, new_schema):

    for e, info in new_schema.items():

        if e not in old_schema:
            old_schema[e] = info
            continue

        old_alias = set(old_schema[e].get("aliases", []))
        new_alias = set(info.get("aliases", []))

        old_schema[e]["aliases"] = list(old_alias | new_alias)

        old_examples = old_schema[e].get("examples", [])
        new_examples = info.get("examples", [])

        old_schema[e]["examples"] = old_examples + new_examples

    return old_schema


def merge_relation_schema(global_schema, new_schema):

    if global_schema is None:
        global_schema = {}

    for rel, rel_info in new_schema.items():

        if rel not in global_schema:

            global_schema[rel] = rel_info
            continue

        existing = global_schema[rel]

        # definition
        if "definition" not in existing and "definition" in rel_info:
            existing["definition"] = rel_info["definition"]

        # head_type
        if "head_type" in rel_info:
            if "head_type" not in existing:
                existing["head_type"] = rel_info["head_type"]
            elif existing["head_type"] != rel_info["head_type"]:
                logger.warning(
                    f"Head type conflict for relation {rel}: "
                    f"{existing['head_type']} vs {rel_info['head_type']}"
                )

        # tail_type
        if "tail_type" in rel_info:
            if "tail_type" not in existing:
                existing["tail_type"] = rel_info["tail_type"]
            elif existing["tail_type"] != rel_info["tail_type"]:
                logger.warning(
                    f"Tail type conflict for relation {rel}: "
                    f"{existing['tail_type']} vs {rel_info['tail_type']}"
                )

        # aliases
        aliases = set(existing.get("aliases", []))
        aliases.update(rel_info.get("aliases", []))
        existing["aliases"] = list(aliases)

        # examples
        examples = existing.get("examples", [])
        new_examples = rel_info.get("examples", [])

        example_set = set(tuple(e) for e in examples)
        for ex in new_examples:
            if tuple(ex) not in example_set:
                examples.append(ex)

        existing["examples"] = examples

    return global_schema


def merge_entity_schema(global_schema, new_schema):

    if global_schema is None:
        global_schema = {}

    for ent, ent_info in new_schema.items():

        if ent not in global_schema:

            global_schema[ent] = ent_info
            continue

        existing = global_schema[ent]

        # definition
        if "definition" not in existing and "definition" in ent_info:
            existing["definition"] = ent_info["definition"]

        # aliases
        aliases = set(existing.get("aliases", []))
        aliases.update(ent_info.get("aliases", []))
        existing["aliases"] = list(aliases)

        # examples
        examples = existing.get("examples", [])
        new_examples = ent_info.get("examples", [])

        example_set = set(examples)
        for ex in new_examples:
            if ex not in example_set:
                examples.append(ex)

        existing["examples"] = examples

    return global_schema


def sanitize_llm_schema_output(llm_output: str):

    def extract_json(text):
        """Extract the first valid JSON object from text."""
        json_pattern = r"\{.*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)

        for m in matches:
            try:
                return json.loads(m)
            except Exception:
                continue

        raise ValueError("No valid JSON found in LLM output")


    schema = extract_json(llm_output)

    if not isinstance(schema, dict):
        schema = {}

    relations = schema.get("relations", {})
    entities = schema.get("entities", {})

    if not isinstance(relations, dict):
        relations = {}

    if not isinstance(entities, dict):
        entities = {}

    clean_relations = {}

    for r, rinfo in relations.items():

        if not isinstance(r, str):
            continue

        if not isinstance(rinfo, dict):
            continue

        definition = rinfo.get("definition", "")
        head_type = rinfo.get("head_type", "")
        tail_type = rinfo.get("tail_type", "")
        aliases = rinfo.get("aliases", [])
        examples = rinfo.get("examples", [])

        if not isinstance(definition, str):
            definition = str(definition)

        if not isinstance(head_type, str):
            head_type = str(head_type)

        if not isinstance(tail_type, str):
            tail_type = str(tail_type)

        if not isinstance(aliases, list):
            aliases = []

        if not isinstance(examples, list):
            examples = []

        clean_relations[r] = {
            "definition": definition.strip(),
            "head_type": head_type.strip(),
            "tail_type": tail_type.strip(),
            "aliases": aliases,
            "examples": examples,
        }

    clean_entities = {}

    for e, einfo in entities.items():

        if not isinstance(e, str):
            continue

        if not isinstance(einfo, dict):
            continue

        definition = einfo.get("definition", "")
        aliases = einfo.get("aliases", [])
        examples = einfo.get("examples", [])

        if not isinstance(definition, str):
            definition = str(definition)

        if not isinstance(aliases, list):
            aliases = []

        if not isinstance(examples, list):
            examples = []

        clean_entities[e] = {
            "definition": definition.strip(),
            "aliases": aliases,
            "examples": examples,
        }

    return {
        "relations": clean_relations,
        "entities": clean_entities,
    }

def merge_schema(global_schema, new_schema):

    # ---- entity types ----
    for etype, info in new_schema["entity_types"].items():

        if etype not in global_schema["entity_types"]:
            global_schema["entity_types"][etype] = info
        else:
            global_schema["entity_types"][etype]["attributes"].update(
                info.get("attributes", {})
            )

    # ---- relation types ----
    for rtype, info in new_schema["relation_types"].items():

        if rtype not in global_schema["relation_types"]:
            global_schema["relation_types"][rtype] = info
        else:
            global_schema["relation_types"][rtype]["attributes"].update(
                info.get("attributes", {})
            )

    return global_schema

def dump_canon_schema(schema, output_path):

    clean_schema = {
        "entity_types": schema.get("entity_types", {}),
        "relation_types": schema.get("relation_types", {}),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_schema, f, indent=2, ensure_ascii=False)


# JSON format
def dump_canon_kg(canon_triplets_list, schema, output_path):

    entities = {}
    relations = []
    attributes = []

    for triplets in canon_triplets_list:

        if triplets is None:
            continue

        for triple in triplets:

            if not isinstance(triple, (list, tuple)):
                continue

            if len(triple) < 3:
                continue

            h, r, t = triple[:3]

            # infer entity type từ schema
            rel_schema = schema["relation_types"].get(r, {})

            head_type = rel_schema.get("head_type")
            tail_type = rel_schema.get("tail_type")

            # add entities
            if h not in entities:
                entities[h] = {
                    "id": h,
                    "type": head_type,
                    "attributes": {}
                }

            if t not in entities:
                entities[t] = {
                    "id": t,
                    "type": tail_type,
                    "attributes": {}
                }

            relations.append({
                "head": h,
                "relation": r,
                "tail": t,
                "attributes": {}
            })

    kg = {
        "entities": list(entities.values()),
        "relations": relations,
        "attributes": attributes
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)

# TXT format
def dump_canon_kg_txt(triplets_per_sample, path):
    """
    Format mỗi dòng:
    [[h, r, t], [h, r, t], ...]
    """

    with open(path, "w", encoding="utf-8") as f:
        for sample in triplets_per_sample:
            f.write(str(sample) + "\n")


def dump_result_each_stage(
    input_text_list,
    oie_triplets_list,
    sd_dict_list,
    canon_triplets_list,
    canon_candidate_dict_list,
    output_path,
):

    results = []

    for idx in range(len(input_text_list)):

        result = {
            "index": idx,
            "text": input_text_list[idx],

            # 🔥 FIX HERE
            "oie": {
                "triples": oie_triplets_list[idx],   # list trực tiếp
            },

            "schema_definition": sd_dict_list[idx],

            "canonicalization": {
                "triples": canon_triplets_list[idx],
                "candidates": canon_candidate_dict_list[idx],
            },
        }

        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def compute_kg_statistics(canon_triplets_list):

    entities = set()
    relations = set()
    triple_count = 0

    for triplets in canon_triplets_list:

        if triplets is None:
            continue

        for triple in triplets:

            if not isinstance(triple, (list, tuple)):
                continue

            if len(triple) < 3:
                continue

            h, r, t = triple[:3]

            entities.add(h)
            entities.add(t)
            relations.add(r)

            triple_count += 1

    return {
        "num_triples": triple_count,
        "num_entities": len(entities),
        "num_relations": len(relations),
    }

def compute_schema_statistics(schema):

    entity_types = schema.get("entity_types", {})
    relation_types = schema.get("relation_types", {})

    attribute_count = 0

    for e in entity_types.values():
        attribute_count += len(e.get("attributes", {}))

    for r in relation_types.values():
        attribute_count += len(r.get("attributes", {}))

    return {
        "num_entity_types": len(entity_types),
        "num_relation_types": len(relation_types),
        "num_attributes": attribute_count,
    }


def evaluate_schema_coverage(schema, canon_triplets_list):

    used_relations = set()

    for triplets in canon_triplets_list:

        if triplets is None:
            continue

        for triple in triplets:

            if len(triple) >= 3:
                used_relations.add(triple[1])

    all_relations = set(schema.get("relation_types", {}).keys())

    grounded = used_relations.intersection(all_relations)

    return {
        "grounded_relations": len(grounded),
        "total_relations": len(all_relations),
        "coverage_ratio": len(grounded) / max(len(all_relations), 1),
    }


from sentence_transformers import SentenceTransformer
import numpy as np


def evaluate_relation_redundancy(schema, embedder_name="all-MiniLM-L6-v2"):

    model = SentenceTransformer(embedder_name)

    relations = list(schema["relation_types"].keys())

    if len(relations) < 2:
        return {"redundancy_score": 0}

    embeddings = model.encode(relations, normalize_embeddings=True)

    sims = np.matmul(embeddings, embeddings.T)

    redundancy_pairs = []

    for i in range(len(relations)):
        for j in range(i + 1, len(relations)):

            if sims[i][j] > 0.85:

                redundancy_pairs.append({
                    "relation1": relations[i],
                    "relation2": relations[j],
                    "similarity": float(sims[i][j])
                })

    return {
        "num_relations": len(relations),
        "redundant_pairs": redundancy_pairs,
        "redundancy_count": len(redundancy_pairs),
    }


def add_relation_alias(schema_case, canonical_rel, raw_rel):
    """
    Add raw relation name into aliases of canonical relation.
    """

    if canonical_rel not in schema_case["relation_types"]:
        return

    rel_obj = schema_case["relation_types"][canonical_rel]

    if "aliases" not in rel_obj:
        rel_obj["aliases"] = []

    # avoid duplicate
    if raw_rel != canonical_rel and raw_rel not in rel_obj["aliases"]:
        rel_obj["aliases"].append(raw_rel)


def add_entity_type_alias(schema_case, canonical_type, raw_type):
    """
    Add raw entity-type name into aliases of the canonical entity type.
    Twin of add_relation_alias, for the entity-canonicalization path.
    """

    if "entity_types" not in schema_case:
        return

    if canonical_type not in schema_case["entity_types"]:
        return

    ent_obj = schema_case["entity_types"][canonical_type]

    if "aliases" not in ent_obj:
        ent_obj["aliases"] = []

    if raw_type != canonical_type and raw_type not in ent_obj["aliases"]:
        ent_obj["aliases"].append(raw_type)


# JSON format (entity types taken from a canonical entity map instead of
# being inferred from the relation's head/tail types). Used only when the
# entity-canonicalization pass is enabled; the original dump_canon_kg is left
# untouched so existing outputs stay reproducible.
def dump_canon_kg_with_entity_types(
    canon_triplets_list,
    schema,
    entity_type_map,
    output_path,
):
    """
    entity_type_map: {entity_surface_name: canonical_entity_type}
    Falls back to the relation head/tail type when an entity is missing
    from the map.
    """

    entities = {}
    relations = []
    attributes = []

    entity_type_map = entity_type_map or {}

    for triplets in canon_triplets_list:

        if triplets is None:
            continue

        for triple in triplets:

            if not isinstance(triple, (list, tuple)):
                continue

            if len(triple) < 3:
                continue

            h, r, t = triple[:3]

            rel_schema = schema["relation_types"].get(r, {})
            head_type = entity_type_map.get(h, rel_schema.get("head_type"))
            tail_type = entity_type_map.get(t, rel_schema.get("tail_type"))

            if h not in entities:
                entities[h] = {"id": h, "type": head_type, "attributes": {}}

            if t not in entities:
                entities[t] = {"id": t, "type": tail_type, "attributes": {}}

            relations.append({
                "head": h,
                "relation": r,
                "tail": t,
                "attributes": {}
            })

    kg = {
        "entities": list(entities.values()),
        "relations": relations,
        "attributes": attributes
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)
