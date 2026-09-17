"""
Entity-type canonicalization.

This is the entity-type twin of SchemaCanonicalizer. Where SC maps a locally
discovered RELATION (e.g. "purchased") onto a global relation type, EC maps a
locally discovered ENTITY fine-type (e.g. "University", produced by SD1) onto a
global entity type in schema["entity_types"], or decides it is new.

Entity types have no head/tail, so the ablation surface is smaller than the 7
relation cases. Three cases are provided:

    ec1_name_only          : embed name only;            verify shows name only
    ec2_name_definition    : embed name + definition;    verify shows name | definition
    ec3_definition_parent  : embed name + definition;    verify shows name | definition | parent,
                             plus a soft "parent" (coarse-type) constraint at retrieval
                             (analogous to the head/tail type penalty in relation case5).

Global entity_types entries look like:  {"definition": str, "attributes": {...}}
(and may optionally carry "parent" / "aliases"). Local SD1 entities are
[entity_name, coarse_type, fine_type, description]; the fine_type is the thing we
canonicalize, coarse_type is treated as its parent, description as its definition.

Cluster-augmented retrieval is shared with relations via soda.schema_clustering,
controlled by the same config keys (retrieval_mode / cluster_sim_threshold /
cluster_top_m / cluster_extra_k).
"""

import numpy as np
import logging
import re
from typing import Dict

import soda.utils.llm_utils as llm_utils
import soda.schema_clustering as schema_clustering

logger = logging.getLogger(__name__)


class EntityCanonicalizer:

    EC_ABLATION_CONFIG = {
        "ec1_name_only": {
            "embedding_style": "name",
            "use_parent_constraint": False,
            "verify_input_mode": "name_only",
            "prompt_key": "ec1",
        },
        "ec2_name_definition": {
            "embedding_style": "name_definition",
            "use_parent_constraint": False,
            "verify_input_mode": "definition",
            "prompt_key": "ec2",
        },
        "ec3_definition_parent": {
            "embedding_style": "name_definition",
            "use_parent_constraint": True,
            "verify_input_mode": "definition_parent",
            "prompt_key": "ec3",
        },
    }

    def __init__(
        self,
        schema: Dict,
        embedder,
        verify_model=None,
        verify_tokenizer=None,
        verify_openai_model=None,
        ec_config=None,
    ):
        assert verify_openai_model is not None or (
            verify_model is not None and verify_tokenizer is not None
        )

        self.schema = schema
        self.embedder = embedder

        self.verify_model = verify_model
        self.verify_tokenizer = verify_tokenizer
        self.verify_openai_model = verify_openai_model

        self.ec_config = ec_config or {}

        self.case_name = self.ec_config.get("ablation_case", "ec2_name_definition")
        self.case_cfg = self.EC_ABLATION_CONFIG[self.case_name]

        self.schema_embedding_dict = {}
        self.text_embedding_cache = {}

        self.retrieval_mode = self.ec_config.get("retrieval_mode", "item")
        self._cluster_index = schema_clustering.SchemaClusterIndex(
            threshold=self.ec_config.get("cluster_sim_threshold", 0.85)
        )

        if "entity_types" not in self.schema:
            self.schema["entity_types"] = {}

        self._build_schema_embeddings()

    # =========================
    # EMBEDDING
    # =========================
    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm

    def _build_embedding_text(self, info):
        style = self.case_cfg["embedding_style"]

        name = info.get("entity_type", "")
        d = info.get("definition", "")

        if style == "name":
            return name
        elif style == "name_definition":
            return f"{name}. {d}" if d else name
        return name

    def _encode(self, text):
        key = str(text)
        if key in self.text_embedding_cache:
            return self.text_embedding_cache[key]

        vec = self._normalize(self.embedder.encode(text))
        self.text_embedding_cache[key] = vec
        return vec

    def _build_schema_embeddings(self):
        self.schema_embedding_dict = {}

        for etype, info in self.schema["entity_types"].items():
            text = self._build_embedding_text({
                "entity_type": etype,
                "definition": info.get("definition", ""),
            })
            self.schema_embedding_dict[etype] = self._encode(text)

    def _add_entity_embedding(self, etype, info):
        text = self._build_embedding_text({
            "entity_type": etype,
            "definition": info.get("definition", ""),
        })
        self.schema_embedding_dict[etype] = self._encode(text)

    # =========================
    # RETRIEVE
    # =========================
    def retrieve(self, query_info, top_k=5):

        if not self.schema_embedding_dict:
            return {}, []

        q = self._encode(self._build_embedding_text(query_info))

        names = list(self.schema_embedding_dict.keys())
        embs = np.array(list(self.schema_embedding_dict.values()))

        scores = np.dot(q, embs.T)

        # soft parent (coarse-type) constraint
        if self.case_cfg["use_parent_constraint"]:
            penalty = self.ec_config.get("parent_penalty", 0.2)
            q_parent = query_info.get("parent")
            if q_parent:
                for i, etype in enumerate(names):
                    c = self.schema["entity_types"][etype]
                    c_parent = c.get("parent")
                    if c_parent and c_parent != q_parent:
                        scores[i] -= penalty

        order = np.argsort(-scores)

        # ---------- ITEM-ONLY (default) ----------
        if self.retrieval_mode != "item+cluster":
            idxs = order[:top_k]
            return (
                {names[i]: self.schema["entity_types"][names[i]] for i in idxs},
                [scores[i] for i in idxs]
            )

        # ---------- ITEM + CLUSTER ----------
        self._cluster_index.maybe_rebuild(self.schema_embedding_dict)

        item_idx = list(order[:top_k])
        item_names = [names[i] for i in item_idx]

        extras = schema_clustering.select_cluster_extras(
            names=names,
            scores=scores,
            q_vec=q,
            cluster_index=self._cluster_index,
            exclude=set(item_names),
            top_m=self.ec_config.get("cluster_top_m", 2),
            extra_k=self.ec_config.get("cluster_extra_k", 3),
        )

        name_to_idx = {names[i]: i for i in range(len(names))}
        final_idx = item_idx + [name_to_idx[n] for n in extras]

        return (
            {names[i]: self.schema["entity_types"][names[i]] for i in final_idx},
            [scores[i] for i in final_idx]
        )

    # =========================
    # PROMPT
    # =========================
    def build_prompt(self, text, query_info, candidates, template):

        lines = ["0. NONE"]

        for i, (etype, info) in enumerate(candidates.items(), start=1):

            if self.case_name == "ec1_name_only":
                line = f"{i}. {etype}"

            elif self.case_name == "ec2_name_definition":
                d = info.get("definition", "")
                line = f"{i}. {etype} | {d}" if d else f"{i}. {etype}"

            else:  # ec3_definition_parent
                d = info.get("definition", "")
                p = info.get("parent", "")
                line = f"{i}. {etype} | {d} | parent: {p}" if p else f"{i}. {etype} | {d}"

            lines.append(line)

        return template.format(
            input_text=text,
            query_entity=query_info.get("entity_type", ""),
            query_entity_definition=query_info.get("definition", ""),
            query_entity_parent=query_info.get("parent", ""),
            choices="\n".join(lines),
        )

    def parse_mcq_answer(self, output, n):
        if not output:
            return None

        text = output.strip()

        m = re.search(r"\b([0-9]+)\b", text)
        if m:
            idx = int(m.group(1))
            return idx if idx < n else None

        m = re.search(r"\b([A-Z])\b", text)
        if m:
            idx = ord(m.group(1)) - ord("A")
            return idx if idx < n else None

        return None

    # =========================
    # MAIN
    # =========================
    def canonicalize(self, text, query_info, prompt_template, prompt_key=None, prompt_path=None):
        """
        query_info: {"entity_type": fine_type, "definition": desc, "parent": coarse}
        Returns: (final_type or None, candidate_dict, scores, debug_info)
        final_type is None when the LLM decides this is a NEW entity type.
        """
        top_k = self.ec_config.get("top_k", 5)

        candidate_dict, scores = self.retrieve(query_info, top_k=top_k)
        cand_names = list(candidate_dict.keys())

        llm_choice = None
        decision = None
        raw_output = None
        prompt = None
        final_type = None

        try:
            prompt = self.build_prompt(text, query_info, candidate_dict, prompt_template)
            messages = [{"role": "user", "content": prompt}]

            if self.verify_openai_model:
                raw_output = llm_utils.openai_chat_completion(
                    self.verify_openai_model, None, messages
                )
            else:
                raw_output = llm_utils.generate_completion_transformers(
                    messages, self.verify_model, self.verify_tokenizer
                )

            idx = self.parse_mcq_answer(raw_output, len(cand_names) + 1)
            llm_choice = idx

            if idx is None:
                decision = "fallback_parse_fail"
                final_type = None
            elif idx == 0:
                decision = "llm_new_entity"
                final_type = None
            else:
                final_type = cand_names[idx - 1]
                decision = "llm_reuse"

        except Exception as e:
            print(f"[WARN] EC LLM MCQ failed: {e}")
            if cand_names:
                final_type = cand_names[0]
                decision = "fallback_top1"
            else:
                final_type = None
                decision = "fallback_none"

        debug_info = {
            "top_k": cand_names,
            "scores": scores,
            "prompt_key": prompt_key,
            "prompt_path": prompt_path,
            "prompt": prompt,
            "raw_output": raw_output,
            "llm_choice": llm_choice,
            "decision": decision,
        }

        return final_type, candidate_dict, scores, debug_info
