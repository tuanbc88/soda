from typing import List, Dict, Any
import logging
import os
import re
import ast
import json
import soda.utils.llm_utils as llm_utils
import soda.utils.log_utils as log_utils
import soda.utils.debug_logger as debug_logger
from soda.pipeline_logger import PipelineLogger

logger = logging.getLogger(__name__)


# =========================
# 🔹 PARSE UTILS
# =========================
def clean_output(text: str):
    text = text.strip()

    if text.startswith("1"):
        text = text[1:].strip()

    text = re.sub(r"```json|```", "", text)

    # 2. remove apostrophe trong word (student's → students)
    text = re.sub(r"(\w)'(\w)", r"\1\2", text)
    
    return text


def extract_list_block(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else None

def safe_literal_eval(text: str):
    # --- replace pattern: word' word → word\' word
    text = re.sub(
        r"([a-zA-Z])'([a-zA-Z])",
        r"\1\\'\2",
        text
    )
    
    for fn in [ast.literal_eval, json.loads]:
        try:
            return fn(text)
        except:
            continue
    return None


def _norm_token(s):
    """Loose key token for matching: case-insensitive, underscores↔spaces, collapsed ws.

    SD2/SD3 LLM output frequently drifts the surface form of subj/rel/obj
    (e.g. ``date_of_birth`` vs ``date of birth``). An exact ``==`` match then
    rejects valid definitions, which both loses them and triggers the retry
    storm. We match on this normalized form and canonicalize the accepted item
    back to the expected surface (see parsers below).
    """
    return re.sub(r"\s+", " ", str(s).strip().lower().replace("_", " "))


def _norm_key3(subj, rel, obj):
    return (_norm_token(subj), _norm_token(rel), _norm_token(obj))


def extract_entities_from_triples(triples):
    seen, entities = set(), []
    for h, _, t in triples:
        for e in (h, t):
            if e not in seen:
                seen.add(e)
                entities.append(e)
    return entities


# =========================
# 🔹 FALLBACKS (OPEN SAFE)
# =========================
def fallback_sd1_entities(entities):
    return [
        [e, "Thing", "Thing", f"{e} is an entity inferred from the text."]
        for e in entities
    ]


def fallback_sd2_relations(typed_triples):
    results = []

    for subj_type, r, obj_type in typed_triples:
        results.append([
            subj_type,
            r,
            obj_type,
            f"a relation between {subj_type} and {obj_type}",
            "unknown"
        ])

    # deduplicate theo full triple
    unique = {}
    for item in results:
        key = (item[0], item[1], item[2])
        unique[key] = item

    return list(unique.values())


def fallback_sd3_roles(relations):
    results = []
    for r in relations:
        subj, rel, obj, _, _, _ = r
        results.append([subj, rel, obj, "main", "self"])
    return results


# =========================
# 🔹 PARSERS
# =========================

def parse_sd1_entities(output, triples, allow_partial=True, **kwargs):
    output = clean_output(output)
    raw = extract_list_block(output)
    if not raw:
        return None

    data = safe_literal_eval(raw)
    if not isinstance(data, list):
        return None

    # expected entities từ triples
    expected = {x for t in triples for x in (t[0], t[2])}

    parsed = []
    for item in data:
        if not (isinstance(item, list) and len(item) == 4):
            continue

        ent, coarse, fine, desc = item

        # basic validation
        if not isinstance(ent, str):
            continue

        # normalize nhẹ
        ent_norm = ent.strip()

        # keep nếu:
        # 1. match expected
        # 2. hoặc allow_partial = True → vẫn giữ
        if ent_norm in expected or allow_partial:
            parsed.append([
                ent_norm,
                str(coarse).strip(),
                str(fine).strip(),
                str(desc).strip()
            ])

    # ❗ quan trọng: không return None nếu có parsed
    if parsed:
        return parsed

    return None

def parse_sd2_relations(output, triples, **kwargs):
    output = clean_output(output)
    raw = extract_list_block(output)
    if not raw:
        return None

    data = safe_literal_eval(raw)
    if not isinstance(data, list):
        return None

    # normalized key -> canonical expected (subj, rel, obj), to absorb surface drift
    expected = {_norm_key3(t[0], t[1], t[2]): (t[0], t[1], t[2]) for t in triples}

    parsed = []
    rejected = []

    for item in data:
        if isinstance(item, list) and len(item) == 5:
            canon = expected.get(_norm_key3(item[0], item[1], item[2]))
            if canon is not None:
                # rewrite surface form so downstream exact-match stays aligned
                item[0], item[1], item[2] = canon
                parsed.append(item)
            else:
                rejected.append(item)

    if rejected:
        logger.warning(f"[SD2 Parser] Rejected {len(rejected)} items:")
        for r in rejected:
            logger.warning(f"  - {r}")

    return parsed if parsed else None

def parse_sd2_general_relations(output, triples, **kwargs):
    """
    Parse GENERAL relation definition:
    [subject, relation, object, general_definition, semantic_hint]
    """
    output = clean_output(output)
    raw = extract_list_block(output)
    if not raw:
        return None

    data = safe_literal_eval(raw)
    if not isinstance(data, list):
        return None

    # normalized key -> canonical expected (subj, rel, obj), to absorb surface drift
    expected = {_norm_key3(t[0], t[1], t[2]): (t[0], t[1], t[2]) for t in triples}

    parsed = []
    rejected = []

    for item in data:
        if isinstance(item, list) and len(item) == 5:
            canon = expected.get(_norm_key3(item[0], item[1], item[2]))
            if canon is not None:
                # rewrite surface form so downstream exact-match stays aligned
                item[0], item[1], item[2] = canon
                parsed.append(item)
            else:
                rejected.append(item)

    if rejected:
        logger.warning(f"[SD2-GENERAL Parser] Rejected {len(rejected)} items:")
        for r in rejected:
            logger.warning(f"  - {r}")

    return parsed if parsed else None


def parse_sd3_roles(output, relations, **kwargs):
    output = clean_output(output)
    raw = extract_list_block(output)
    if not raw:
        return None

    data = safe_literal_eval(raw)
    if not isinstance(data, list):
        return None

    # normalized relation name -> canonical relation name, to absorb surface drift
    expected = {_norm_token(r[1]): r[1] for r in relations}

    parsed = []
    for item in data:
        if isinstance(item, list) and len(item) == 5:
            canon = expected.get(_norm_token(item[1]))
            if canon is not None:
                item[1] = canon
                parsed.append(item)

    return parsed if parsed else None


# =========================
# 🔹 HELPERS
# =========================
def ensure_all_entities(defined, original_entities):
    defined_names = {e[0] for e in defined}
    missing = [e for e in original_entities if e not in defined_names]

    if missing:
        logger.warning(f"Missing entities → fallback add: {missing}")
        defined += fallback_sd1_entities(missing)

    return defined

def find_missing_entities(parsed, expected_entities):
    parsed_names = {e[0] for e in parsed}
    return [e for e in expected_entities if e not in parsed_names]

def dedup_entities(ent_defs):
    seen = set()
    new_list = []
    for e in ent_defs:
        key = e[0]  # entity name
        if key not in seen:
            new_list.append(e)
            seen.add(key)
    return new_list

def find_missing_relations(parsed, typed_triples):
    parsed_keys = {(r[0], r[1], r[2]) for r in parsed}
    expected_keys = {(t[0], t[1], t[2]) for t in typed_triples}

    return [list(t) for t in expected_keys - parsed_keys]
    

def build_typed_triples(triples, entity_defs):
    ent_map = {e[0]: e[2] for e in entity_defs}  # fine_type

    typed = []
    for h, r, t in triples:
        subj_type = ent_map.get(h, "Thing")

        # 🔥 FIX isA collapse
        if r == "isA":
            obj_type = "Thing"   # hoặc giữ taxonomy riêng nếu bạn muốn
        else:
            obj_type = ent_map.get(t, "Thing")

        typed.append([subj_type, r, obj_type])

    return typed

def dedup_triples(triples):
    seen = set()
    new_list = []
    for t in triples:
        key = tuple(t)
        if key not in seen:
            new_list.append(t)
            seen.add(key)
    return new_list

def is_ontology_relation(rel: str):
    rel_norm = rel.lower().strip()

    # keywords = [
    #     "isa", "is_a", "type", "typeof", "instanceof",
    #     "subclass", "kindof", "category", "classof"
    # ]
    keywords = [ "isa", "is_a"]

    # exact or substring match
    return any(k in rel_norm for k in keywords)

def split_triples(triples):
    ontology_triples = []
    normal_triples = []

    for t in triples:
        h, r, o = t

        if is_ontology_relation(r):
            ontology_triples.append(t)
        else:
            normal_triples.append(t)

    # 🔥 logging rõ ràng
    if ontology_triples:
        logger.info(f"[Ontology Filter] removed {len(ontology_triples)} triples:")
        for t in ontology_triples:
            logger.info(f"  - {t}")

    return normal_triples, ontology_triples
    

# =========================
# 🔥 MAIN CLASS
# =========================
class SchemaDefiner:

    def __init__(self, model=None, tokenizer=None, openai_model=None, use_sd_general=None, pipeline_logger: PipelineLogger = None,iter_dir=None, sd_temperature=0.0):
        assert openai_model or (model and tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.openai_model = openai_model
        self.use_sd_general = use_sd_general
        self.pipeline_logger = pipeline_logger
        self.iter_dir = iter_dir
        # 0.0 = greedy (deterministic). >0 = sample the SD definition/typing stage (the
        # controlled variance source for the ≥3-seed study). OIE/SC stay greedy.
        self.sd_temperature = float(sd_temperature or 0.0)

    def _call_llm(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        if self.openai_model:
            return llm_utils.openai_chat_completion(
                self.openai_model, None, messages, temperature=self.sd_temperature
            )

        return llm_utils.generate_completion_transformers(
            messages, self.model, self.tokenizer, answer_prepend="",
            temperature=self.sd_temperature
        )

    def _retry_parse(self, prompt, parse_fn, max_retry=3, **kwargs):
        last_output = None

        for _ in range(max_retry):
            out = self._call_llm(prompt)

            # When SD is greedy (sd_temperature=0) re-running the same prompt yields the
            # SAME output → once we see a repeat, further retries are wasted compute, stop.
            # When SD samples (sd_temperature>0) outputs differ, so this never short-circuits
            # and a parse-fail genuinely gets a fresh sample (the intended retry behavior).
            if last_output is not None and out == last_output:
                break

            last_output = out

            try:
                parsed = parse_fn(out, **kwargs)
            except Exception as e:
                logger.warning(f"Parse error: {e}")
                parsed = None

            if parsed:
                return parsed, out, True

        return None, last_output, False

    def build_local_schema(self, ent_defs, rel_defs, roles):
        """
        Merge SD1 + SD2 + SD3 → unified schema
        """
    
        # =========================
        # ENTITY TYPES
        # =========================
        entity_types = {}
    
        for ent, coarse, fine, desc in ent_defs:
            if fine not in entity_types:
                entity_types[fine] = {
                    "parent": coarse,
                    "definition": desc,
                    "attributes": {}
                }
    
        # =========================
        # RELATION TYPES
        # =========================
        relation_types = {}
    
        # index roles by relation
        role_map = {}
        for r in roles:
            head, rel, tail, role, parent = r
            role_map.setdefault(rel, []).append(r)
    
        for subj_type, rel, obj_type, definition, rel_type in rel_defs:
    
            relation_types[rel] = {
                "definition": definition,
                "head_type": subj_type,
                "tail_type": obj_type,
                "attributes": {},
                "relation_type": rel_type
            }
    
            # attach attributes
            for r in role_map.get(rel, []):
                h, r_name, t, role, parent = r
    
                if role == "rel_attr":
                    relation_types[rel]["attributes"][t] = {
                        "type": "relation_attribute",
                        "value_type": t,
                        "parent": parent
                    }
    
                elif role == "subj_attr":
                    # attach vào entity type
                    if subj_type in entity_types:
                        entity_types[subj_type]["attributes"][t] = {
                            "type": "entity_attribute",
                            "value_type": t
                        }
    
        return {
            "entity_types": entity_types,
            "relation_types": relation_types
        }

    def merge_sd_outputs(ent_defs, rel_defs, roles):
        """
        Merge SD1 + SD2 + SD3 → local schema (clean + dedup)
        """
    
        # =========================
        # ENTITY TYPES
        # =========================
        entity_types = {}
    
        for ent, coarse, fine, desc in ent_defs:
            if fine not in entity_types:
                entity_types[fine] = {
                    "parent": coarse,
                    "definition": desc,
                    "attributes": {}
                }
            else:
                # 🔥 resolve conflict: keep better definition
                old = entity_types[fine]["definition"]
                if len(desc) > len(old):
                    entity_types[fine]["definition"] = desc
    
        # =========================
        # RELATION TYPES
        # =========================
        relation_types = {}
    
        for subj, rel, obj, definition, rel_type in rel_defs:
    
            if rel not in relation_types:
                relation_types[rel] = {
                    "definition": definition,
                    "head_type": subj,
                    "tail_type": obj,
                    "attributes": {},
                    "relation_type": rel_type
                }
            else:
                # 🔥 resolve conflict
                old = relation_types[rel]
    
                # ưu tiên cái có type đầy đủ
                score_old = int(bool(old["head_type"])) + int(bool(old["tail_type"]))
                score_new = int(bool(subj)) + int(bool(obj))
    
                if score_new > score_old or len(definition) > len(old["definition"]):
                    relation_types[rel].update({
                        "definition": definition,
                        "head_type": subj,
                        "tail_type": obj,
                    })
    
        # =========================
        # ATTACH ATTRIBUTES (SD3)
        # =========================
        for head, rel, tail, role, parent in roles:
    
            if role == "main":
                continue
    
            # relation attribute
            if role == "rel_attr":
                if parent in relation_types:
                    relation_types[parent]["attributes"][tail] = {
                        "value_type": tail,
                        "attr_type": "relation_attribute"
                    }
    
            # entity attribute
            elif role == "subj_attr":
                if head in entity_types:
                    entity_types[head]["attributes"][tail] = {
                        "value_type": tail,
                        "attr_type": "entity_attribute"
                    }
    
        return {
            "entity_types": entity_types,
            "relation_types": relation_types
        }   

    def merge_roles_into_rel_defs(self, rel_defs, roles_defs):

        if len(rel_defs) != len(roles_defs):
            logger.warning("SD2 vs SD3 size mismatch → fallback safe merge")
        
        role_map = {}
    
        for r in roles_defs:
            if len(r) >= 5:
                subj, rel, obj, role, parent = r
                key = (subj, rel, obj)
                role_map[key] = (role, parent)
    
        merged = []
    
        for rel_item in rel_defs:
            subj, rel, obj, definition, rel_type = rel_item
    
            key = (subj, rel, obj)
    
            role, parent = role_map.get(key, ("main", "self"))
    
            merged.append({
                "relation": rel,
                "head_type": subj,
                "tail_type": obj,
                "definition": definition,
                "relation_type": rel_type,
                "role": role,
                "parent": parent
            })
    
        return merged     
    
    def define_schema_full(
        self,
        text,
        triples,
        sd1_prompt, sd1_fs,
        sd2_prompt, sd2_fs,
        sd3_prompt, sd3_fs,
        sd_general_prompt=None,
        sd_general_fs=None,
        sd_general_edc_prompt=None,   # v2: EDC-style (Type A) general-def template (optional)
        sd_general_edc_fs=None,       # v2: EDC-style (Type A) general-def few-shot (optional)
        index: int = None,
    ):
        
        use_sd2=True
        use_sd3=True 
        # ================= SD1 =================
        entities = extract_entities_from_triples(triples)
        
        # print(f'----- idx: {index} - SD1: entity input--')
        # print(f' + entities no: {len(entities)}')
        # print(f' + entities: {entities}')
        # print('---------------------------')

        # --- First pass ---
        p1 = sd1_prompt.format_map({
            "text": text,
            "triples": triples,
            "entities": entities,
            "few_shot_examples": sd1_fs
        })

        p1_context = {
            "idx": index,
            "type": "SD1 entity definition prompt",
            "prompt": p1
        }
        debug_logger.jsonl_log(self.iter_dir, "sd_logs", p1_context)

        
        ent_defs, raw1, ok1 = self._retry_parse(
            p1, parse_sd1_entities, triples=triples
        )
        
        if not ent_defs:
            ent_defs = []
        
        # 🔥 tìm missing
        missing_entities = find_missing_entities(ent_defs, entities)
        
        if missing_entities:
            logger.info(f"SD1 missing → retry: {missing_entities}")
        
            p1_retry = sd1_prompt.format_map({
                "text": text,
                "triples": triples,
                "entities": missing_entities,   # 🔥 chỉ truyền missing
                "few_shot_examples": sd1_fs
            })
        
            extra_defs, _, ok_extra = self._retry_parse(
                p1_retry,
                parse_sd1_entities,
                triples=[[e, "", e] for e in missing_entities],
                allow_partial=False   # 🔥 FIX
            )
        
            if extra_defs:
                ent_defs.extend(extra_defs)
                
            ent_defs = dedup_entities(ent_defs)    
        
        # fallback cuối cùng
        ent_defs = ensure_all_entities(ent_defs, entities)
        
        sd1_output = {
            "idx": index,
            "type": "SD1 entity definition output",
            "output_ent_defs": ent_defs
        }
        debug_logger.jsonl_log(self.iter_dir, "sd_logs", sd1_output)
        
        
        if self.pipeline_logger:
            self.pipeline_logger.log(
                item_id=index,
                stage="SD1",
                data={
                    #"raw_output": completion,
                    "ent_defs": ent_defs
                }
            )


        # ================= SD2-GENERAL relation definition =================
        general_rel_defs = []
        raw2_general = None
        
        if self.use_sd_general and sd_general_prompt:
        
            p2g = sd_general_prompt.format_map({
                "text": text,
                "triples": triples,
                "few_shot_examples": sd_general_fs
            })

            # print(f'----- idx: {index} - SD2a: general relation prompt--')
            # print(f' + p2g: {p2g}')
            # print('---------------------------')

            p2a_context = {
                "idx": index,
                "type": "SD2a general relation definition prompt",
                "prompt": p2g
            }
            debug_logger.jsonl_log(self.iter_dir, "sd_logs", p2a_context)
            
        
            general_rel_defs, raw2_general, ok2g = self._retry_parse(
                p2g,
                parse_sd2_general_relations,
                triples=triples
            )
        
            if not general_rel_defs:
                general_rel_defs = []

            sd2a_output = {
                "idx": index,
                "type": "SD2a general relation definition output",
                "output_general_rel_defs": general_rel_defs
            }
            debug_logger.jsonl_log(self.iter_dir, "sd_logs", sd2a_output)
            
            if self.pipeline_logger:
                self.pipeline_logger.log(
                    item_id=index,
                    stage="SD2_GENERAL",
                    data={
                        "general_rel_defs": general_rel_defs
                    }
                )

        # ====== SD2a-EDC: EDC-style general def (Type A), v2 ======
        # Parallel pass to SD2a above, same prompt template but the EDC few-shot
        # ("The subject entity [verb] the object entity specified by ..."). Produces
        # general_definition_edc (case3). Only runs when an EDC few-shot is provided.
        general_edc_rel_defs = []
        # Dedicated Type-A template ONLY (no fallback to the abstract template — that would
        # produce abstract-styled edc defs). The framework guarantees both are present when
        # use_sd_general is on, so this pass runs in lock-step with the abstract SD2a pass.
        if self.use_sd_general and sd_general_edc_prompt and sd_general_edc_fs:

            p2g_edc = sd_general_edc_prompt.format_map({
                "text": text,
                "triples": triples,
                "few_shot_examples": sd_general_edc_fs
            })

            debug_logger.jsonl_log(self.iter_dir, "sd_logs", {
                "idx": index,
                "type": "SD2a-EDC general relation definition prompt",
                "prompt": p2g_edc
            })

            general_edc_rel_defs, _raw2_edc, _ok2g_edc = self._retry_parse(
                p2g_edc,
                parse_sd2_general_relations,
                triples=triples
            )

            if not general_edc_rel_defs:
                general_edc_rel_defs = []

            debug_logger.jsonl_log(self.iter_dir, "sd_logs", {
                "idx": index,
                "type": "SD2a-EDC general relation definition output",
                "output_general_edc_rel_defs": general_edc_rel_defs
            })

            if self.pipeline_logger:
                self.pipeline_logger.log(
                    item_id=index,
                    stage="SD2_GENERAL_EDC",
                    data={"general_edc_rel_defs": general_edc_rel_defs}
                )

        # ================= SD2 (detailed relation definition) =================
        relations = []
        raw2 = None

        # tách những relations dạng isA, ra khỏi danh sách define.    
        triples, ontology_triples = split_triples(triples)
        
        if use_sd2:
            typed_triples = build_typed_triples(triples, ent_defs)  # # build typed triples

            # print(f'----- idx: {index} - SD2: typed triples input--')
            # print(f' + typed_triples no: {len(typed_triples)}')
            # print(json.dumps(typed_triples, indent=2, ensure_ascii=False))
            # print('---------------------------')
            
            p2 = sd2_prompt.format_map({
                "text": text,
                "triples": triples,
                "entities": ent_defs,
                "typed_triples": typed_triples,
                "few_shot_examples": sd2_fs
            })

            p2b_context = {
                "idx": index,
                "type": "SD2b detailed relation prompt",
                "prompt": p2
            }
            debug_logger.jsonl_log(self.iter_dir, "sd_logs", p2b_context)

        
            rel_defs, raw2, ok2 = self._retry_parse(
                p2,
                parse_sd2_relations,
                triples=typed_triples   # 🔥 dùng typed_triples
            )
            
            if not rel_defs:
                rel_defs = []
            
            # 🔥 tìm missing
            missing_relations = find_missing_relations(rel_defs, typed_triples)
            
            if missing_relations:
                logger.info(f"SD2 missing → retry: {missing_relations}")
            
                p2_retry = sd2_prompt.format_map({
                    "text": text,
                    "triples": triples,              # giữ original context
                    "entities": ent_defs,
                    "typed_triples": missing_relations,   # 🔥 QUAN TRỌNG
                    "few_shot_examples": sd2_fs
                })
            
                extra_rels, _, ok_extra = self._retry_parse(
                    p2_retry,
                    parse_sd2_relations,
                    triples=missing_relations
                )
            
                if extra_rels:
                    rel_defs.extend(extra_rels)
                    
                rel_defs = dedup_triples(rel_defs)
            
            # fallback cuối cùng (chỉ phần còn thiếu)
            final_missing = find_missing_relations(rel_defs, typed_triples)
            
            if final_missing:
                logger.warning(f"SD2 fallback for: {final_missing}")
            
                fallback = fallback_sd2_relations(typed_triples)
                fb_map = {(r[0], r[1], r[2]): r for r in fallback}
            
                for m in final_missing:
                    key = tuple(m)
                    if key in fb_map:
                        rel_defs.append(fb_map[key])
            
            relations = rel_defs
        
            sd2b_output = {
                "idx": index,
                "type": "SD2b detalied relation definition output",
                "output_detailed_rel_defs": rel_defs
            }
            debug_logger.jsonl_log(self.iter_dir, "sd_logs", sd2b_output)
            
            
            if self.pipeline_logger:
                self.pipeline_logger.log(
                    item_id=index,
                    stage="SD2",
                    data={
                        #"raw_output": completion,
                        "rel_defs": rel_defs
                    }
                )
        # ================= SD3 =================
        roles = []
        raw3 = None

        if use_sd3 and relations:
            p3 = sd3_prompt.format_map({
                "relations": relations,
                "few_shot_examples": sd3_fs
            })

            p3_context = {
                "idx": index,
                "type": "SD3 roles of relation prompt",
                "prompt": p3
            }
            debug_logger.jsonl_log(self.iter_dir, "sd_logs", p3_context)

            role_defs, raw3, ok3 = self._retry_parse(
                p3, parse_sd3_roles, relations=relations
            )

            if not ok3:
                logger.warning("SD3 fallback triggered")
                role_defs = fallback_sd3_roles(relations)

            roles = role_defs


            if self.pipeline_logger:
                self.pipeline_logger.log(
                    item_id=index,
                    stage="SD3",
                    data={
                        #"raw_output": completion,
                        "relation_roles": roles
                    }
                )

            p3_output = {
                "idx": index,
                "type": "SD3 roles of relation - output",
                "output": roles
            }
            debug_logger.jsonl_log(self.iter_dir, "sd_logs", p3_output)

        # print(f'----- idx: {index} - SD3: defined roles output---------------')
        # print(f' + role_defs no: {len(role_defs)}')
        # print(json.dumps(role_defs, indent=2, ensure_ascii=False))
        # print('-----------------------------------------------------')
        
        
        final_rel_defs = self.merge_roles_into_rel_defs(rel_defs, roles)

        # ================= MERGE GENERAL =================
        if self.use_sd_general and (general_rel_defs or general_edc_rel_defs):

            # =========================
            # BUILD GENERAL MAPS (relation-level)
            # abstract (Type B, from SD2a) + edc (Type A, from SD2a-EDC)
            # =========================
            general_map = {}
            for r in general_rel_defs:
                if len(r) == 5:
                    _, rel, _, g_def, g_type = r
                    # 🔥 nếu nhiều instance → lấy cái đầu (hoặc bạn có thể vote sau)
                    if rel not in general_map:
                        general_map[rel] = (g_def, g_type)

            general_edc_map = {}
            for r in general_edc_rel_defs:
                if len(r) == 5:
                    _, rel, _, g_def, _g_type = r
                    if rel not in general_edc_map:
                        general_edc_map[rel] = g_def

            for rel_item in final_rel_defs:
                rel = rel_item["relation"]

                # abstract (Type B) -> general_definition_abstract
                if rel in general_map:
                    g_def, g_type = general_map[rel]
                    rel_item["general_definition_abstract"] = g_def
                    rel_item["semantic_hint"] = g_type
                else:
                    rel_item["general_definition_abstract"] = ""
                    rel_item["semantic_hint"] = "unknown"

                # edc (Type A) -> general_definition_edc
                rel_item["general_definition_edc"] = general_edc_map.get(rel, "")


        
        # print(f'----- idx: {index} - SD: full final output---------------')
        # print(f' + ent_defs no: {len(ent_defs)}')
        # print(f' + final_rel_defs no: {len(final_rel_defs)}')
        # print(json.dumps(ent_defs, indent=2, ensure_ascii=False))
        # print(json.dumps(final_rel_defs, indent=2, ensure_ascii=False))
        # print('----------------------------------------------------------')
        sd_output = {
                "idx": index,
                "type": "SD output",
                "output": {
                            "entities": ent_defs,
                            "relations": final_rel_defs
                            }
        }
                    
        debug_logger.jsonl_log(self.iter_dir, "sd_logs", sd_output)
                
        return {
            "entities": ent_defs,
            "relations": final_rel_defs,
            "sd1_raw": raw1,
            "sd2_general_raw": raw2_general, 
            "sd2_raw": raw2,
            "sd3_raw": raw3,
        }