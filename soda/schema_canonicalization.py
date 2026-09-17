import numpy as np
import copy
import os
import logging
from typing import Dict, List, Tuple, Any
import re

import soda.utils.llm_utils as llm_utils
from soda.pipeline_logger import PipelineLogger
import soda.schema_clustering as schema_clustering

logger = logging.getLogger(__name__)

class SchemaCanonicalizer:
    # ===================================================================
    # SC ablation — v2 9-case design (decided 2026-06-21; see ARCHITECTURE.md
    # §"SC ablation — v2 9-case design"). The old v1 case2 (general def) is split
    # into TWO cases: case3 = EDC-style def (Type A) / case4 = abstract def
    # (Type B). v1→v2 map: case0→case1 · case1→case2 · case2→case4 (+NEW case3) ·
    # case3→case5 · case4→case6 · case5→case7 · case6→case8 · case7→case9.
    # `prompt_key` selects one of 4 MCQ template files (case1..case4), NOT the
    # case identity; the candidate-line format is keyed on case_name in build_prompt.
    # ===================================================================
    SC_ABLATION_CONFIG = {

        # case1: NO LLM. Pure cosine similarity on the name embedding + a
        # threshold to decide reuse-vs-new (top-1 >= threshold -> reuse, else
        # new). Non-LLM baseline isolating how much the LLM MCQ adds. Threshold =
        # sc_config["case0_sim_threshold"] (default 0.85; CLI flag keeps legacy name).
        "case1_embed_threshold": {
            "embedding_style": "name",
            "use_type_constraint": False,
            "verify_input_mode": "name_only",
            "prompt_key": None,
            "no_llm": True,
        },

        "case2_name_only": {
            "embedding_style": "name",
            "use_type_constraint": False,
            "verify_input_mode": "name_only",
            "prompt_key": "case1",
        },

        # case3: EDC-style general def (Type A): "The subject entity [verb] the
        # object entity specified by ..." — the EDC-faithful signal.
        "case3_name_gendef_edc": {
            "embedding_style": "name_gendef_edc",
            "use_type_constraint": False,
            "verify_input_mode": "definition_only",
            "prompt_key": "case2",
        },

        # case4: abstract general def (Type B): "indicates that one entity ...
        # another entity" — SODA's self-generated gloss.
        "case4_name_gendef_abstract": {
            "embedding_style": "name_gendef_abstract",
            "use_type_constraint": False,
            "verify_input_mode": "definition_only",
            "prompt_key": "case2",
        },

        "case5_name_detail": {
            "embedding_style": "name_detail",
            "use_type_constraint": False,
            "verify_input_mode": "query_typed",
            "prompt_key": "case3",
        },

        "case6_name_detail_headtail": {
            "embedding_style": "name_detail_headtail",
            "use_type_constraint": False,
            "verify_input_mode": "query_typed",
            "prompt_key": "case4",
        },

        "case7_detail_typed": {
            "embedding_style": "name_detail",
            "use_type_constraint": True,
            "verify_input_mode": "fully_typed",
            "prompt_key": "case4",
        },

        "case8_concat": {
            "embedding_style": "split_vector_concat",
            "use_type_constraint": False,
            "verify_input_mode": "fully_typed",
            "prompt_key": "case4",
        },

        "case9_weighted": {
            "embedding_style": "split_vector_weighted",
            "use_type_constraint": False,
            "verify_input_mode": "fully_typed",
            "prompt_key": "case4",
        },
    }
    
    def __init__(
        self,
        schema: Dict,
        embedder,
        verify_model=None,
        verify_tokenizer=None,
        verify_openai_model=None,
        sc_config=None,
    ):

        assert verify_openai_model is not None or (
            verify_model is not None and verify_tokenizer is not None
        )

        self.schema = schema
        self.embedder = embedder

        self.verify_model = verify_model
        self.verify_tokenizer = verify_tokenizer
        self.verify_openai_model = verify_openai_model

        self.sc_config = sc_config or {}
        # OFF for now: gates the planned LLM-as-a-Judge self-verify step (template
        # sc_schema_self_verify.txt). Pipeline currently does retrieval -> MCQ only.
        self.sc_config["use_llm_verify"] = False

        self.case_name = self.sc_config.get("ablation_case", "case2_name_only")
        if self.case_name not in self.SC_ABLATION_CONFIG:
            raise ValueError(
                f"Unknown SC ablation_case {self.case_name!r}; valid: "
                f"{list(self.SC_ABLATION_CONFIG)}"
            )
        self.case_cfg = self.SC_ABLATION_CONFIG[self.case_name]

        self.schema_embedding_dict = {}
        self.text_embedding_cache = {}

        # Fail fast on a malformed seed schema: the relation field this case depends on
        # MUST be present + non-empty for every seeded relation (no silent fallback).
        self._validate_schema()

        # ---- cluster-augmented retrieval (optional) ----
        # retrieval_mode == "item"          -> original behavior (default)
        # retrieval_mode == "item+cluster"  -> item top-k + cluster extras
        self.retrieval_mode = self.sc_config.get("retrieval_mode", "item")
        self._cluster_index = schema_clustering.SchemaClusterIndex(
            threshold=self.sc_config.get("cluster_sim_threshold", 0.85)
        )

        self._build_schema_embeddings()

    # Relation field that the active case's embedding style REQUIRES (non-empty).
    # None => the case needs no definition field (name-only / embed-threshold).
    _STYLE_REQUIRED_FIELD = {
        "name": None,
        "name_gendef_edc": "general_definition_edc",
        "name_gendef_abstract": "general_definition_abstract",
        "name_detail": "definition",
        "name_detail_headtail": "definition",
        "split_vector_concat": "definition",
        "split_vector_weighted": "definition",
    }

    def _required_field(self):
        style = self.case_cfg["embedding_style"]
        if style not in self._STYLE_REQUIRED_FIELD:
            raise ValueError(f"Unknown embedding_style {style!r} for case {self.case_name!r}")
        return self._STYLE_REQUIRED_FIELD[style]

    def _validate_schema(self):
        """Raise if any seeded relation lacks the field this case depends on.

        Strict by design: no fallback to a different definition field (that would
        silently test the wrong canon signal and corrupt the ablation). For Mode 1 the
        seed schema is empty here, so this is a no-op until relations are self-generated.
        """
        field = self._required_field()
        if not field:
            return
        bad = [
            rel for rel, info in self.schema.get("relation_types", {}).items()
            if not (info.get(field) or "").strip()
        ]
        if bad:
            raise ValueError(
                f"SC case {self.case_name!r} requires non-empty {field!r} on every "
                f"relation, but {len(bad)} lack it (e.g. {bad[:5]}). Fix the seed schema "
                f"(schemas/*.json) — no fallback is applied."
            )

    # =========================
    # EMBEDDING
    # =========================
    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm

    def _build_embedding_text(self, info):
        style = self.case_cfg["embedding_style"]

        rel = info.get("relation", "")
        # v2: two SEPARATE general-def fields, NO fallback (a fallback would silently
        # substitute the wrong signal and corrupt the case3/case4 ablation).
        g_edc = info.get("general_definition_edc", "")
        g_abs = info.get("general_definition_abstract", "")
        d = info.get("definition", "")
        h = info.get("head_type", "")
        t = info.get("tail_type", "")

        if style == "name":
            return rel
        elif style == "name_gendef_edc":
            return f"{rel}. {g_edc}"
        elif style == "name_gendef_abstract":
            return f"{rel}. {g_abs}"
        elif style == "name_detail":
            return f"{rel}. {d}"
        elif style == "name_detail_headtail":
            return f"{rel}. {d}. Head: {h}. Tail: {t}"
        elif style in ["split_vector_concat", "split_vector_weighted"]:
            return {
                "name": rel,
                "definition": d,
                "head": h,
                "tail": t
            }
        return rel

    def _encode(self, text):
        key = str(text)
        if key in self.text_embedding_cache:
            return self.text_embedding_cache[key]

        if isinstance(text, dict):
            name = self._encode(text["name"])
            d = self._encode(text["definition"])
            h = self._encode(text["head"])
            t = self._encode(text["tail"])

            if self.case_cfg["embedding_style"] == "split_vector_concat":
                vec = np.concatenate([name, d, h, t])
            else:
                w = self.sc_config.get("fusion_weights", {
                    "name": 0.4, "definition": 0.3, "head": 0.15, "tail": 0.15
                })
                vec = w["name"]*name + w["definition"]*d + w["head"]*h + w["tail"]*t

            vec = self._normalize(vec)
        else:
            vec = self._normalize(self.embedder.encode(text))

        self.text_embedding_cache[key] = vec
        return vec

    def _build_schema_embeddings(self):
        self.schema_embedding_dict = {}

        for rel, info in self.schema["relation_types"].items():
            text = self._build_embedding_text({
                "relation": rel,
                "general_definition": info.get("general_definition", ""),
                "general_definition_edc": info.get("general_definition_edc", ""),
                "general_definition_abstract": info.get("general_definition_abstract", ""),
                "definition": info.get("definition", ""),
                "head_type": info.get("head_type", ""),
                "tail_type": info.get("tail_type", ""),
            })
            self.schema_embedding_dict[rel] = self._encode(text)

    # =========================
    # RETRIEVE
    # =========================
    def retrieve(self, query_info, top_k=5):

        if not self.schema_embedding_dict:
            return {}, []

        q = self._encode(self._build_embedding_text(query_info))

        rels = list(self.schema_embedding_dict.keys())
        embs = np.array(list(self.schema_embedding_dict.values()))

        scores = np.dot(q, embs.T)

        if self.case_cfg["use_type_constraint"]:
            penalty = self.sc_config.get("type_penalty", 0.2)
            for i, rel in enumerate(rels):
                c = self.schema["relation_types"][rel]
                if (
                    query_info.get("head_type") != c.get("head_type")
                    or query_info.get("tail_type") != c.get("tail_type")
                ):
                    scores[i] -= penalty

        order = np.argsort(-scores)

        # ---------- ITEM-ONLY (original behavior, default) ----------
        if self.retrieval_mode != "item+cluster":
            idxs = order[:top_k]
            return (
                {rels[i]: self.schema["relation_types"][rels[i]] for i in idxs},
                [scores[i] for i in idxs]
            )

        # ---------- ITEM + CLUSTER ----------
        self._cluster_index.maybe_rebuild(self.schema_embedding_dict)

        item_idx = list(order[:top_k])
        item_names = [rels[i] for i in item_idx]

        extras = schema_clustering.select_cluster_extras(
            names=rels,
            scores=scores,
            q_vec=q,
            cluster_index=self._cluster_index,
            exclude=set(item_names),
            top_m=self.sc_config.get("cluster_top_m", 2),
            extra_k=self.sc_config.get("cluster_extra_k", 3),
        )

        name_to_idx = {rels[i]: i for i in range(len(rels))}
        final_idx = item_idx + [name_to_idx[n] for n in extras]

        return (
            {rels[i]: self.schema["relation_types"][rels[i]] for i in final_idx},
            [scores[i] for i in final_idx]
        )

    # =========================
    # LLM MCP
    # =========================
    def llm_select(self, text, triplet, rel_info, candidates, template):

        prompt = self.build_prompt(text, triplet, rel_info, candidates, template)

        messages = [{"role": "user", "content": prompt}]

        if self.verify_openai_model:
            output = llm_utils.openai_chat_completion(
                self.verify_openai_model, None, messages
            )
        else:
            output = llm_utils.generate_completion_transformers(
                messages, self.verify_model, self.verify_tokenizer
            )

        idx = self.parse_mcq_answer(output, len(candidates) + 1)

        if idx is None or idx == 0:
            return None

        return list(candidates.keys())[idx - 1]

    # =========================
    # PROMPT
    # =========================
    def build_prompt(self, text, triplet, rel_info, candidates, template):

        mode = self.case_cfg["verify_input_mode"]

        rel_def = ""

        # NO fallback: each case reads exactly its own definition field.
        if self.case_name == "case3_name_gendef_edc":
            rel_def = rel_info.get("general_definition_edc", "")
        elif self.case_name == "case4_name_gendef_abstract":
            rel_def = rel_info.get("general_definition_abstract", "")
        elif self.case_name in [
            "case5_name_detail", "case6_name_detail_headtail",
            "case7_detail_typed", "case8_concat", "case9_weighted",
        ]:
            rel_def = rel_info.get("definition", "")

        lines = ["0. NONE"]

        for i, (rel, info) in enumerate(candidates.items(), start=1):

            if self.case_name == "case2_name_only":
                line = f"{i}. {rel}"

            elif self.case_name == "case3_name_gendef_edc":
                g = info.get("general_definition_edc", "")
                line = f"{i}. {rel} | {g}" if g else f"{i}. {rel}"

            elif self.case_name == "case4_name_gendef_abstract":
                g = info.get("general_definition_abstract", "")
                line = f"{i}. {rel} | {g}" if g else f"{i}. {rel}"

            elif self.case_name == "case5_name_detail":
                d = info.get("definition", "")
                line = f"{i}. {rel} | {d}" if d else f"{i}. {rel}"

            else:  # case6/7/8/9 -> rel | detail | head -> tail
                d = info.get("definition", "")
                h = info.get("head_type", "")
                t = info.get("tail_type", "")
                line = f"{i}. {rel} | {d} | {h} -> {t}"

            lines.append(line)
    
        return template.format(
            input_text=text,
            query_triplet=f"({triplet[0]}, {triplet[1]}, {triplet[2]})",
            query_relation=rel_info.get("relation", ""),
            #query_relation_definition=rel_info.get("definition", ""),
            query_relation_definition=rel_def,   # use general or detail
            head_type=rel_info.get("head_type", ""),
            tail_type=rel_info.get("tail_type", ""),
            choices="\n".join(lines)
        )

    def parse_mcq_answer(self, output, n):
        if not output:
            return None
    
        text = output.strip()
    
        # ✅ ưu tiên số
        m = re.search(r"\b([0-9]+)\b", text)
        if m:
            idx = int(m.group(1))
            return idx if idx < n else None
    
        # fallback chữ (A,B,C)
        m = re.search(r"\b([A-Z])\b", text)
        if m:
            idx = ord(m.group(1)) - ord("A")
            return idx if idx < n else None
    
        return None

    def strip_rel_info_by_case(self, rel_info):
        mode = self.case_cfg["verify_input_mode"]

        base = {"relation": rel_info.get("relation", "")}

        # NO fallback: each case carries exactly its own definition field.
        if self.case_name == "case3_name_gendef_edc":
            base["general_definition_edc"] = rel_info.get("general_definition_edc", "")
        elif self.case_name == "case4_name_gendef_abstract":
            base["general_definition_abstract"] = rel_info.get("general_definition_abstract", "")
        elif mode != "name_only":
            base["definition"] = rel_info.get("definition", "")

        if mode in ["query_typed", "fully_typed"]:
            base["head_type"] = rel_info.get("head_type", "")
            base["tail_type"] = rel_info.get("tail_type", "")

        return base

    def _add_relation_embedding(self, rel, info):
        text = self._build_embedding_text({
            "relation": rel,
            "general_definition": info.get("general_definition", ""),
            "general_definition_edc": info.get("general_definition_edc", ""),
            "general_definition_abstract": info.get("general_definition_abstract", ""),
            "definition": info.get("definition", ""),
            "head_type": info.get("head_type", ""),
            "tail_type": info.get("tail_type", ""),
        })
        self.schema_embedding_dict[rel] = self._encode(text)

    # =========================
    # MAIN
    # =========================
    def canonicalize(self, text, triplet, rel_info, prompt_template, prompt_key=None, prompt_path=None):
        h, r, t = triplet
    
        top_k = self.sc_config.get("top_k", 5)
    
        # ===============================
        # 1. RETRIEVE
        # ===============================
        candidate_dict, scores = self.retrieve(rel_info, top_k=top_k)
        rel_names = list(candidate_dict.keys())

        llm_choice = None
        decision = None
        raw_output = None
        prompt = None

        # ===============================
        # case1 embed_threshold: pure embedding + threshold (NO LLM MCQ)
        # top-1 cosine >= threshold -> reuse top-1; otherwise mark new (None).
        # (sc_config key keeps its legacy `case0_sim_threshold` name.)
        # ===============================
        if self.case_cfg.get("no_llm"):
            thr = self.sc_config.get("case0_sim_threshold", 0.85)
            if rel_names:
                top1, top1_score = rel_names[0], float(scores[0])
                if top1_score >= thr:
                    final_rel = top1
                    decision = f"embed_reuse|sim={top1_score:.3f}>=thr={thr}"
                else:
                    final_rel = None
                    decision = f"embed_new|sim={top1_score:.3f}<thr={thr}"
            else:
                final_rel = None
                decision = "embed_new|empty_schema"

            debug_info = {
                "top_k": rel_names,
                "scores": scores,
                "prompt_key": prompt_key,
                "prompt_path": prompt_path,
                "prompt": prompt,
                "raw_output": raw_output,
                "llm_choice": llm_choice,
                "decision": decision,
            }
            return final_rel, candidate_dict, scores, debug_info

        try:
            # ===============================
            # 2. BUILD PROMPT
            # ===============================
            prompt = self.build_prompt(
                text,
                triplet,
                rel_info,
                candidate_dict,
                prompt_template
            )
    
            messages = [{"role": "user", "content": prompt}]
    
            # ===============================
            # 3. LLM MCQ
            # ===============================
            if self.verify_openai_model:
                raw_output = llm_utils.openai_chat_completion(
                    self.verify_openai_model, None, messages
                )
            else:
                raw_output = llm_utils.generate_completion_transformers(
                    messages, self.verify_model, self.verify_tokenizer
                )
    
            idx = self.parse_mcq_answer(raw_output, len(rel_names) + 1)
            llm_choice = idx
    
            # ===============================
            # 4. DECISION
            # ===============================
            if idx is None:
                decision = "fallback_parse_fail"
                final_rel = None
    
            elif idx == 0:
                decision = "llm_new_relation"
                final_rel = None
    
            else:
                final_rel = rel_names[idx - 1]
                decision = "llm_reuse"
    
        except Exception as e:
            print(f"[WARN] LLM MCQ failed: {e}")
    
            if rel_names:
                final_rel = rel_names[0]
                decision = "fallback_top1"
            else:
                final_rel = None
                decision = "fallback_none"
    
        # ===============================
        # DEBUG INFO
        # ===============================
        debug_info = {
            "top_k": rel_names,
            "scores": scores,
            "prompt_key": prompt_key,
            "prompt_path": prompt_path,
            "prompt": prompt,
            "raw_output": raw_output,
            "llm_choice": llm_choice,
            "decision": decision,
        }

        return final_rel, candidate_dict, scores, debug_info

    # ===================================================================
    # BATCHING SPLIT (item-13 speed-up) — prepare() + finalize()
    # ---------------------------------------------------------------
    # canonicalize() above is a single (retrieve → build_prompt → LLM → parse)
    # call. To batch the per-CASE LLM calls of ONE triplet (the only correctness-safe
    # axis — see DECISIONS 2026-06-27c), the LLM step must be hoisted out of the loop.
    # prepare() does everything up to (but excluding) the LLM call; finalize() does
    # everything after. By construction `finalize(prepare(...), <llm(prompt)>)` returns
    # exactly what canonicalize() returns (verified by scripts/test_batch_generation.py).
    # canonicalize() itself is left UNTOUCHED so the default (non-batched) path stays
    # byte-identical. These are used ONLY when --sc_batch_size > 1.
    # ===================================================================
    def prepare(self, text, triplet, rel_info, prompt_template, prompt_key=None, prompt_path=None):
        """Phase 1: retrieve candidates + build the MCQ prompt, NO LLM call.

        Returns a `prepared` dict consumed by finalize(). For no_llm cases (case1) and
        prompt-build failures the decision is already final (needs_llm stays False)."""
        top_k = self.sc_config.get("top_k", 5)
        candidate_dict, scores = self.retrieve(rel_info, top_k=top_k)
        rel_names = list(candidate_dict.keys())

        prepared = {
            "candidate_dict": candidate_dict,
            "scores": scores,
            "rel_names": rel_names,
            "prompt_key": prompt_key,
            "prompt_path": prompt_path,
            "prompt": None,
            "needs_llm": False,
            "final_rel": None,
            "decision": None,
            "llm_choice": None,
        }

        # case1 embed_threshold: pure embedding + threshold (NO LLM) — identical to
        # the `no_llm` branch in canonicalize().
        if self.case_cfg.get("no_llm"):
            thr = self.sc_config.get("case0_sim_threshold", 0.85)
            if rel_names:
                top1, top1_score = rel_names[0], float(scores[0])
                if top1_score >= thr:
                    prepared["final_rel"] = top1
                    prepared["decision"] = f"embed_reuse|sim={top1_score:.3f}>=thr={thr}"
                else:
                    prepared["final_rel"] = None
                    prepared["decision"] = f"embed_new|sim={top1_score:.3f}<thr={thr}"
            else:
                prepared["final_rel"] = None
                prepared["decision"] = "embed_new|empty_schema"
            return prepared

        # LLM cases: build the prompt now; the (batched) LLM call happens outside.
        # canonicalize() wraps build+llm+parse in ONE try whose except is fallback_top1;
        # a build failure here replays that except branch.
        try:
            prepared["prompt"] = self.build_prompt(
                text, triplet, rel_info, candidate_dict, prompt_template
            )
            prepared["needs_llm"] = True
        except Exception as e:
            print(f"[WARN] LLM MCQ failed (prompt build): {e}")
            if rel_names:
                prepared["final_rel"] = rel_names[0]
                prepared["decision"] = "fallback_top1"
            else:
                prepared["final_rel"] = None
                prepared["decision"] = "fallback_none"
        return prepared

    def finalize(self, prepared, raw_output, llm_ok=True):
        """Phase 3: parse the (batched) raw_output → decision; return canonicalize()'s
        4-tuple. raw_output is ignored when the decision was already made in prepare()
        (no_llm / build-fail). llm_ok=False replays canonicalize()'s except branch
        (fallback_top1) for a hard batch/LLM failure."""
        candidate_dict = prepared["candidate_dict"]
        scores = prepared["scores"]
        rel_names = prepared["rel_names"]

        final_rel = prepared["final_rel"]
        decision = prepared["decision"]
        llm_choice = prepared["llm_choice"]

        if prepared["needs_llm"] and decision is None:
            if not llm_ok:
                # hard failure (batch call raised) → canonicalize() except: fallback_top1
                if rel_names:
                    final_rel, decision = rel_names[0], "fallback_top1"
                else:
                    final_rel, decision = None, "fallback_none"
            else:
                try:
                    idx = self.parse_mcq_answer(raw_output, len(rel_names) + 1)
                    llm_choice = idx
                    if idx is None:
                        decision, final_rel = "fallback_parse_fail", None
                    elif idx == 0:
                        decision, final_rel = "llm_new_relation", None
                    else:
                        final_rel, decision = rel_names[idx - 1], "llm_reuse"
                except Exception as e:
                    print(f"[WARN] LLM MCQ failed (parse): {e}")
                    if rel_names:
                        final_rel, decision = rel_names[0], "fallback_top1"
                    else:
                        final_rel, decision = None, "fallback_none"

        debug_info = {
            "top_k": rel_names,
            "scores": scores,
            "prompt_key": prepared["prompt_key"],
            "prompt_path": prepared["prompt_path"],
            "prompt": prepared["prompt"],
            "raw_output": raw_output if prepared["needs_llm"] else None,
            "llm_choice": llm_choice,
            "decision": decision,
        }
        return final_rel, candidate_dict, scores, debug_info
