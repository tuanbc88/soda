"""SODA pipeline: Extract, Define, Canonicalize, Assess.

Adapted from the EDC framework of Zhang and Soh (EMNLP 2024),
https://github.com/clear-nus/edc, which is MIT licensed. Substantially modified
and extended; EDC's copyright and permission notice are reproduced in NOTICE.
"""
import logging
from typing import List, Dict, Tuple
import numpy as np
from typing import List
from tqdm import tqdm
import os
import time
import csv
import pathlib
from functools import partial
import copy
import logging
from importlib import reload
import random
import json
import pandas as pd
import ast
import re

from soda.utils.schema_utils import merge_entity_schema, merge_relation_schema, load_schema
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from soda.evaluate.evaluation_redundancy_relation import compute_relation_redundancy_semantic
from soda.evaluate.compute_sd_sc_metrics import compute_sd_sc_metrics
from soda.extract import Extractor
from soda.schema_definition import SchemaDefiner
from soda.schema_canonicalization import SchemaCanonicalizer
from soda.entity_canonicalization import EntityCanonicalizer
from soda import entity_instance_canonicalization as eic
from soda.entity_extraction import EntityExtractor
import soda.utils.llm_utils as llm_utils
import soda.utils.log_utils as log_utils
import soda.utils.schema_utils as schema_utils
import soda.utils.debug_logger as debug_logger
from soda.utils.e5_mistral_utils import MistralForSequenceEmbedding
from soda.schema_retriever import SchemaRetriever
from soda.pipeline_logger import PipelineLogger 

reload(logging)
logger = logging.getLogger(__name__)

import gc


def _free_caches():
    """Release Python garbage + CUDA cache to curb memory creep over long runs."""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"_free_caches failed: {e}")


def _checkpoint_json(obj, path):
    """Crash-safe incremental dump: write to a tmp file then atomically replace,
    so a kill mid-write can't corrupt the checkpoint."""
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"_checkpoint_json({path}) failed: {e}")


def _sc_checkpoint(path, done, n_items, cases, results, schema_per_case):
    """SC mid-stage resume checkpoint (RESUME_UPGRADE_PLAN.md Phase 2).

    SC is schema-stateful, so we persist BOTH the per-case results and the evolved
    per-case schema; on resume the SchemaCanonicalizer rebuilds its embedding index
    from the restored schema in __init__, so a resumed run is identical (greedy) to
    an uninterrupted one. `debug` is stripped from the stored results (it is not read
    downstream — extract_pred_triplets/save_sc_results only use input_triplet/
    pred_relation/top_k; the per-item sc_mcp_logs JSONL keeps the full trace), keeping
    the checkpoint small. Written atomically via _checkpoint_json."""
    slim = {}
    for case, samples in results.items():
        slim[case] = [
            {"text": s["text"],
             "triplets": [{"input_triplet": t["input_triplet"],
                           "pred_relation": t["pred_relation"],
                           "top_k": t.get("top_k", [])}
                          for t in s["triplets"]]}
            for s in samples
        ]
    _checkpoint_json(
        {"done": done, "n_items": n_items, "cases": list(cases),
         "results": slim, "schema_per_case": schema_per_case},
        path,
    )


def _ec_checkpoint(path, done, n_items, cases, results, schema_per_case,
                   entity_type_map_per_case):
    """EC mid-stage resume checkpoint (RESUME_UPGRADE_PLAN.md Phase 3).

    Like SC, entity canonicalization is schema-stateful, so a resumable checkpoint must
    carry EVERY piece of state the pass accumulates:
      * results               -> dumped verbatim to entity_canon_{ec_case}.json, so unlike
                                 the SC checkpoint nothing may be stripped (each entity dict
                                 is already slim: input_fine_type/pred_entity_type/members/
                                 top_k/is_new/decision);
      * schema_per_case       -> the grown entity_types (+aliases); restored BEFORE the
                                 EntityCanonicalizer objects are built so each rebuilds its
                                 embedding index from it, making a resumed run identical
                                 (greedy) to an uninterrupted one;
      * entity_type_map_per_case -> surface-name -> canonical-type, consumed by
                                 dump_canon_kg_with_entity_types; it accumulates across
                                 items and cannot be recomputed from the other two.
    Written atomically via _checkpoint_json."""
    _checkpoint_json(
        {"done": done, "n_items": n_items, "cases": list(cases),
         "results": results, "schema_per_case": schema_per_case,
         "entity_type_map_per_case": entity_type_map_per_case},
        path,
    )

# NOTE (2026-06-22): this per-model GEN_CONFIGS dict was NEVER wired — every stage calls
# `generate_completion_transformers(...)` WITHOUT a temperature, so decoding defaulted to
# temperature=0.0 → do_sample=False → GREEDY everywhere (OIE, SD, SC). All v1 results were
# greedy. We removed the misleading dict. Decoding policy is now explicit:
#   - OIE  : greedy (faithful extraction — no sampling, no fabricated triples)
#   - SC   : greedy (deterministic MCQ)
#   - SD   : greedy by default; set `--sd_temperature > 0` (env `EDC_SD_TEMPERATURE`) to SAMPLE
#            the definition/typing stage — the controlled variance source for the ≥3-seed study
#            (seed reproducibility via `--seed`). See DECISIONS 2026-06-22 / RUN_GUIDE §2b.




class EDC:
    def __init__(self, **edc_configuration) -> None:
        
        # OIE - Open Infomation Extraction module settings
        self.oie_llm_name = edc_configuration["oie_llm"]
        self.oie_prompt_template_file_path = edc_configuration["oie_prompt_template_file_path"]
        self.oie_few_shot_example_file_path = edc_configuration["oie_few_shot_example_file_path"]

        # SD decoding temperature. 0.0 = greedy (default, reproduces v1). >0 = SAMPLE the SD
        # definition/typing stage → the controlled variance source for the ≥3-seed study (OIE
        # stays greedy regardless). Reproducible per --seed. Env `EDC_SD_TEMPERATURE` also honored.
        self.sd_temperature = float(
            edc_configuration.get("sd_temperature", None)
            if edc_configuration.get("sd_temperature", None) is not None
            else os.environ.get("EDC_SD_TEMPERATURE", 0.0)
        )

        # COREF - optional LLM coreference REWRITE stage BEFORE OIE (approach A).
        # OFF by default. Reuses the OIE LLM unless --coref_llm is given. Intended
        # for GraphRAG-scope datasets only — it rewrites entity surface forms, so it
        # breaks the span-based strict/exact/partial metrics on gold-triple sets.
        self.enable_coref = edc_configuration.get("enable_coref", False)
        # backend: "llm" (Qwen3 rewrite, both langs) | "maverick" (ACL'24, English-only)
        self.coref_method = (edc_configuration.get("coref_method") or "llm").lower()
        self.coref_llm_name = edc_configuration.get("coref_llm") or self.oie_llm_name
        self.coref_prompt_template_file_path = edc_configuration.get(
            "coref_prompt_template_file_path",
            "./soda/prompt_templates/eng/coref_template.txt",
        )
        self.coref_few_shot_example_file_path = edc_configuration.get(
            "coref_few_shot_example_file_path", None
        )

        # SD - Schema Definition module settings
        self.sd_llm_name = edc_configuration["sd_llm"]


        self.sd1_entity_template_file_path = edc_configuration["sd1_entity_prompt_template_file_path"]  # define entity type
        self.sd1_entity_few_shot_example_file_path = edc_configuration["sd1_entity_few_shot_example_file_path"]  

        self.sd2a_generalrelation_template_file_path = edc_configuration["sd2a_generalrelation_prompt_template_file_path"]  # general define relation
        self.sd2a_generalrelation_few_shot_example_file_path = edc_configuration["sd2a_generalrelation_few_shot_example_file_path"]
        # v2: EDC-style (Type A) general-def pass (optional; its OWN template + few-shot)
        self.sd2a_edc_generalrelation_template_file_path = edc_configuration.get("sd2a_edc_generalrelation_prompt_template_file_path")
        self.sd2a_edc_generalrelation_few_shot_example_file_path = edc_configuration.get("sd2a_edc_generalrelation_few_shot_example_file_path")
        
        self.sd2b_relation_template_file_path = edc_configuration["sd2_relation_prompt_template_file_path"]  # detail define relation
        self.sd2b_relation_few_shot_example_file_path = edc_configuration["sd2_relation_few_shot_example_file_path"]  

        self.sd3_clasy_template_file_path = edc_configuration["sd3_clasy_prompt_template_file_path"]  # define role
        self.sd3_clasy_few_shot_example_file_path = edc_configuration["sd3_clasy_few_shot_example_file_path"]  

        
        # SC - Schema Canonicalization module settings
        self.sc_llm_name = edc_configuration["sc_llm"]
        self.sc_embedder_name = edc_configuration["sc_embedder"]
        
        # self.sc_template_file_path = edc_configuration["sc_prompt_template_file_path"]    # original - not use
        self.sc_template_option1_file_path  = edc_configuration["sc_prompt_template_option1_file_path"]   # name only
        self.sc_template_option2_file_path  = edc_configuration["sc_prompt_template_option2_file_path"]   # general define relation
        self.sc_template_option3_file_path  = edc_configuration["sc_prompt_template_option3_file_path"]   # detail define relation
        self.sc_template_option4_file_path  = edc_configuration["sc_prompt_template_option4_file_path"]   # detail define relation + head/tail   
        self.sc_self_verify_template_file_path  = edc_configuration["sc_prompt_self_verify_template_file_path"]  # RESERVED: planned LLM-as-a-Judge self-verify step (sc_schema_self_verify.txt) — not wired yet; keep the file (use_llm_verify flag is off for now)

        
        self.sc_config = edc_configuration.get("sc_config", {
            "embedding_mode": "structured",   # definition | concat | structured | triplet_aware
            "verify_mode": "mcq",   # multi choice question
            "verify_input_mode": "fully_typed",  # definition_only (option1) | query_typed (option2) | fully_typed ((option3))
            "top_k": 5,
            "use_llm_verify": True,    #  Binary Self-Verify   True/False
            "use_type_constraint": True,
            "use_joint_scoring": True,
            "alpha": 0.7,   # embedding weight
            "beta": 0.3,    # type constraint weight

            # ---- cluster-augmented retrieval (NEW) ----
            # "item" = original behavior (default, reproducible);
            # "item+cluster" = item top-k + cluster extras
            "retrieval_mode": "item",
            "cluster_sim_threshold": 0.85,
            "cluster_top_m": 2,
            "cluster_extra_k": 3,

            # ---- entity-type canonicalization (NEW) ----
            # off by default so relation outputs/files stay reproducible
            "enable_entity_canon": False,
            "parent_penalty": 0.2,

            # ---- entity-INSTANCE canonicalization (RQ §8.5; DECISIONS 2026-07-16b) ----
            # Merges coreferent entity NODES (the GraphRAG-recall lever), unlike enable_entity_canon
            # which only assigns TYPES. OFF by default: it rewrites entity surface forms, so it is
            # GraphRAG-scope only (eduhcmut / hotpot) — on gold-triple sets it would change the
            # entity basis vs EDC (§8.2). Writes NEW artifacts; never touches canon_kg_{case}.txt.
            "enable_instance_canon": False,
            # ic2/ic3 need EC's canonical types; this picks which EC case supplies them
            # (None -> the first EC case, mirroring the canon_kg_with_entity_* dump).
            "instance_canon_ec_case": None,

            # ---- case0: non-LLM embedding+threshold baseline (NEW) ----
            # top-1 cosine >= this threshold -> reuse, else new.
            "case0_sim_threshold": 0.85,
        })

        # CLI overrides (from runa.py flags). Only keys explicitly present are
        # applied, so the defaults above still hold otherwise.
        for _k in [
            "retrieval_mode",
            "cluster_sim_threshold",
            "cluster_top_m",
            "cluster_extra_k",
            "enable_entity_canon",
            "enable_instance_canon",
            "instance_canon_ec_case",
            "case0_sim_threshold",
            "top_k",
            "sc_batch_size",
        ]:
            if _k in edc_configuration and edc_configuration[_k] is not None:
                self.sc_config[_k] = edc_configuration[_k]

        # Entity-canon prompt templates (only opened when enable_entity_canon=True).
        # Defaults keep runa.py working even if it doesn't set them explicitly.
        self.ec_template_option1_file_path = edc_configuration.get(
            "ec_prompt_template_option1_file_path",
            "./soda/prompt_templates/eng/ec_template_option1_name_only.txt",
        )
        self.ec_template_option2_file_path = edc_configuration.get(
            "ec_prompt_template_option2_file_path",
            "./soda/prompt_templates/eng/ec_template_option2_definition.txt",
        )
        self.ec_template_option3_file_path = edc_configuration.get(
            "ec_prompt_template_option3_file_path",
            "./soda/prompt_templates/eng/ec_template_option3_definition_parent.txt",
        )
        
        # Refinement settings (for Iterative Schema Refinement)
        self.sr_adapter_path = edc_configuration["sr_adapter_path"]

        self.sr_embedder_name = edc_configuration["sr_embedder"]
        self.oie_r_prompt_template_file_path = edc_configuration["oie_refine_prompt_template_file_path"]
        self.oie_r_few_shot_example_file_path = edc_configuration["oie_refine_few_shot_example_file_path"]

        self.ee_llm_name = edc_configuration["ee_llm"]
        self.ee_template_file_path = edc_configuration["ee_prompt_template_file_path"]
        self.ee_few_shot_example_file_path = edc_configuration["ee_few_shot_example_file_path"]
        # PER-ITEM immediate OIE refine (re-extract each item right after OIE pass-1 with a
        # local EE-merged entity hint). Independent of end-to-end refine
        # (refinement_iterations, set on extract_kg). Default OFF -> no behavior change.
        self.refine_per_item = edc_configuration.get("refine_per_item", False)

        self.em_template_file_path = edc_configuration["em_prompt_template_file_path"]

        self.initial_schema_path = edc_configuration["target_schema_path"]
        self.enrich_schema = edc_configuration["enrich_schema"]

        # ---- run mode (NEW) -------------------------------------------------
        # Mode 1: no schema  -> schema empty, must allow new types
        # Mode 2: schema + enrich -> allow new types
        # Mode 3: schema + no enrich -> frozen schema, never add new types
        # allow_new_schema_base is the per-run base; extract_kg() may further
        # disable it for late iterations via freeze_iter.
        self.allow_new_schema_base = (
            self.initial_schema_path is None or bool(self.enrich_schema)
        )

        self.eval_embedder_name = edc_configuration["eval_embedder"]

        if self.initial_schema_path is not None:
            self.schema = load_schema(self.initial_schema_path)
        else:
            self.schema = {
                "relation_types": {},
                "entity_types": {}
            }

        # Mode 1 safety: an empty schema must allow new types regardless of flags.
        if not self.schema.get("relation_types") and not self.schema.get("entity_types"):
            self.allow_new_schema_base = True

        # Load the needed models and tokenizers
        self.needed_model_set = set(
            [self.oie_llm_name, self.sd_llm_name, self.sc_llm_name, self.sc_embedder_name, self.ee_llm_name, self.eval_embedder_name]
        )

        self.loaded_model_dict = {}

        logging.basicConfig(level=edc_configuration["loglevel"])

        logger.info(f"Model used: {self.needed_model_set}")
        

    def load_model(self, model_name, model_type, role=None):
        """
        Load (or reuse cached) a model.

        role: "oie" | "sd" | "sc" | "ee" | None
            When set, per-role ENV vars are checked first, falling back to the
            global EDC_LOAD_IN_4BIT / EDC_LOAD_IN_8BIT.
            Per-role ENV pattern: EDC_<ROLE>_LOAD_IN_4BIT / EDC_<ROLE>_LOAD_IN_8BIT
            Example:  EDC_SC_LOAD_IN_4BIT=1  (quantize only the SC role)
        """
        assert model_type in ["sts", "hf"]

        # vLLM backend (item-13 step 2): when EDC_USE_VLLM=1, every HF-model load returns a
        # shared vLLM engine instead of an HF model (one engine per distinct model; OIE=SD=SC=EE
        # are usually the same → one engine serves all). Recognised downstream by __soda_vllm__.
        # OFF by default → the HF path below is untouched. See VLLM_PLAN.md.
        from soda.utils import vllm_backend
        if model_type == "hf" and vllm_backend.is_vllm_enabled():
            cache_key = f"{model_name}::vllm"
            if cache_key not in self.loaded_model_dict:
                logger.info(f"Loading {model_name} via vLLM (role={role})")
                engine = vllm_backend.get_engine(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.loaded_model_dict[cache_key] = (engine, tokenizer)
            else:
                logger.info(f"vLLM engine {model_name} already loaded, reusing.")
            return self.loaded_model_dict[cache_key]

        # Resolve quantization flags for this role.
        def _quant_flag(suffix):
            if role:
                v = os.environ.get(f"EDC_{role.upper()}_{suffix}")
                if v is not None:
                    return v == "1"
            return os.environ.get(f"EDC_{suffix}", "0") == "1"

        load_in_4bit = _quant_flag("LOAD_IN_4BIT")
        load_in_8bit = _quant_flag("LOAD_IN_8BIT")
        quant_key = "4bit" if load_in_4bit else ("8bit" if load_in_8bit else "fp")

        # Cache key includes quant so that same model at different precisions
        # can coexist (e.g. 7B-4bit for OIE/SD, 7B-fp16 for SC is unlikely but safe).
        cache_key = f"{model_name}::{quant_key}"

        if cache_key in self.loaded_model_dict:
            logger.info(f"Model {model_name} ({quant_key}) already loaded, reusing.")
        else:
            logger.info(f"Loading model {model_name} (role={role}, quant={quant_key})")
            if model_type == "hf":
                if torch.cuda.is_available():
                    use_bf16 = torch.cuda.is_bf16_supported()
                    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
                else:
                    compute_dtype = torch.float32

                quant_config = None
                if load_in_4bit or load_in_8bit:
                    from transformers import BitsAndBytesConfig
                    if load_in_4bit:
                        quant_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=compute_dtype,
                            bnb_4bit_use_double_quant=True,
                        )
                    else:
                        quant_config = BitsAndBytesConfig(load_in_8bit=True)

                if quant_config is not None:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        quantization_config=quant_config,
                        low_cpu_mem_usage=True,
                    )
                    logger.info(
                        f"Loaded {model_name} quantized "
                        f"({'4bit-nf4' if load_in_4bit else '8bit'}, "
                        f"compute={compute_dtype})"
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=compute_dtype,
                        low_cpu_mem_usage=True,
                    )
                    logger.info(f"Loaded {model_name} in {compute_dtype}")

                tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.loaded_model_dict[cache_key] = (model, tokenizer)
            elif model_type == "sts":
                model = SentenceTransformer(model_name, trust_remote_code=True)
                self.loaded_model_dict[cache_key] = model

        return self.loaded_model_dict[cache_key]

    def _evict_model_cache(self, model_name: str):
        # Keep any ::vllm engine resident — one engine serves every stage for the whole run,
        # so evicting it mid-run would force an expensive re-init (and is usually wrong).
        keys = [k for k in self.loaded_model_dict
                if k.startswith(f"{model_name}::") and not k.endswith("::vllm")]
        for k in keys:
            del self.loaded_model_dict[k]

    def coref(self, input_text_list: List[str], pipeline_logger=None):
        """LLM coreference REWRITE (approach A) — a separate stage BEFORE OIE.

        Each item is rewritten so pronouns / anaphora / definite-descriptions are
        replaced by explicit canonical mentions; OIE/SD/SC then run on the rewritten
        text. OFF by default (``enable_coref``). Reuses the OIE model (cached → no
        extra VRAM) unless ``coref_llm`` differs. The ORIGINAL text is kept in
        ``coref_total.json`` (idx / original_text / coref_text) for inspection and
        for downstream chunk linkage. Resumable: if that file exists it is reused.

        ⚠️ It changes entity surface forms → it BREAKS span-based strict/exact/partial
        on the gold-triple sets (webnlg/rebel/wiki-nre). Intended for GraphRAG-scope
        datasets only. See RESEARCH_QUESTIONS.md §8.4.

        Backend (``coref_method``): "llm" (Qwen3 rewrite, EN+VI) or "maverick"
        (Maverick, ACL 2024 — efficient dedicated coref, ENGLISH-only).
        """
        path = os.path.join(self.iter_dir, "coref_total.json") if self.iter_dir else None
        if path and os.path.exists(path):
            recs = load_total_file(path)
            print(f"🔥 Loading coref from total file ({len(recs)} items)")
            return [r.get("coref_text") or input_text_list[i]
                    for i, r in enumerate(recs)]

        method = getattr(self, "coref_method", "llm")
        if method == "maverick":
            out_list, recs = self._coref_maverick(input_text_list)
        else:
            out_list, recs = self._coref_llm(input_text_list)

        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(recs, f, indent=2, ensure_ascii=False)
            print(f"[Coref] saved {len(recs)} items ({method}) → {path}")
        return out_list

    def _coref_llm(self, input_text_list):
        """Coref backend A: prompt the OIE LLM (Qwen3) to rewrite the item (EN+VI)."""
        if not llm_utils.is_model_openai(self.coref_llm_name):
            model, tok = self.load_model(self.coref_llm_name, "hf", role="coref")
        else:
            model, tok = None, None

        tmpl = open(self.coref_prompt_template_file_path, encoding="utf-8").read()
        fs = ""
        if self.coref_few_shot_example_file_path and os.path.exists(self.coref_few_shot_example_file_path):
            fs = open(self.coref_few_shot_example_file_path, encoding="utf-8").read()

        logger.info("Running Coref (LLM rewrite, pre-OIE)...")
        out_list, recs = [], []
        for idx, text in enumerate(input_text_list):
            prompt = tmpl.format_map({"few_shot_examples": fs, "input_text": text})
            messages = [{"role": "user", "content": prompt}]
            try:
                if model is None:
                    raw = llm_utils.openai_chat_completion(self.coref_llm_name, None, messages)
                else:
                    # rewrite ≈ input length → give a generous output budget
                    raw = llm_utils.generate_completion_transformers(
                        messages, model, tok, dynamic_ratio=1.6, max_new_tokens_cap=2048)
            except Exception as e:
                logger.warning(f"[Coref] idx={idx} failed: {e}")
                raw = None

            coref_text = _clean_coref_output(raw)
            # Safety: fall back to the ORIGINAL if the rewrite is empty/degenerate
            # (too short vs input ⇒ the LLM probably dropped content).
            if not coref_text or len(coref_text) < max(8, int(0.4 * len(text.strip()))):
                coref_text = text.strip()

            out_list.append(coref_text)
            recs.append({"idx": idx, "method": "llm",
                         "original_text": text.strip(), "coref_text": coref_text})
            if (idx + 1) % 50 == 0:
                print(f"[Coref-llm] {idx + 1}/{len(input_text_list)}")
        return out_list, recs

    def _coref_maverick(self, input_text_list):
        """Coref backend B: Maverick (ACL 2024) clusters → substitute each mention
        with its cluster's representative. ENGLISH-only. `pip install maverick-coref`."""
        model = self._load_maverick()
        logger.info("Running Coref (Maverick rewrite, pre-OIE)...")
        out_list, recs = [], []
        for idx, text in enumerate(input_text_list):
            t = text.strip()
            try:
                pred = model.predict(t)
                spans = pred.get("clusters_char_offsets") or []
                mentions = pred.get("clusters_text_mentions") or []
                coref_text = _maverick_rewrite(t, spans, mentions)
                clusters_log = mentions
            except Exception as e:
                logger.warning(f"[Coref-maverick] idx={idx} failed: {e}")
                coref_text, clusters_log = t, []

            if not coref_text or len(coref_text) < max(8, int(0.4 * len(t))):
                coref_text = t

            out_list.append(coref_text)
            recs.append({"idx": idx, "method": "maverick", "original_text": t,
                         "coref_text": coref_text, "clusters": clusters_log})
            if (idx + 1) % 50 == 0:
                print(f"[Coref-maverick] {idx + 1}/{len(input_text_list)}")
        return out_list, recs

    def _load_maverick(self):
        if getattr(self, "_maverick_model", None) is not None:
            return self._maverick_model
        try:
            from maverick import Maverick
        except ImportError as e:
            raise ImportError(
                "coref_method='maverick' needs the maverick-coref package "
                "(English-only). Install: pip install maverick-coref"
            ) from e
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._maverick_model = Maverick(device=device)
        logger.info(f"[Coref] Maverick loaded on {device}.")
        return self._maverick_model

    def oie(
        self, input_text_list: List[str], previous_extracted_triplets_list: List[List[str]] = None, free_model=False
        , pipeline_logger=None
    ):
        if not llm_utils.is_model_openai(self.oie_llm_name):
            # Load the HF model for OIE
            oie_model, oie_tokenizer = self.load_model(self.oie_llm_name, "hf", role="oie")
            #extractor = Extractor(oie_model, oie_tokenizer)
            extractor = Extractor(oie_model,oie_tokenizer, pipeline_logger=pipeline_logger,iter_dir=self.iter_dir)
        else:
            extractor = Extractor(openai_model=self.oie_llm_name)

        oie_triples_list = []
        entity_hint_list = None
        relation_hint_list = None
        # LLM-call counter (cost comparison none/per-item/end-to-end), reset per OIE call
        self._refine_calls = {"oie_pass1": 0, "oie_pass2": 0, "ee": 0, "merge": 0}

        if previous_extracted_triplets_list is not None:
            # Refined OIE
            logger.info("Running Refined OIE...")
            oie_refinement_prompt_template_str = open(self.oie_r_prompt_template_file_path).read()
            oie_refinement_few_shot_examples_str = open(self.oie_r_few_shot_example_file_path).read()

            logger.info("Putting together the refinement hint...")
            entity_hint_list, relation_hint_list = self.construct_refinement_hint(
                input_text_list, previous_extracted_triplets_list, free_model=free_model
            )

            assert len(previous_extracted_triplets_list) == len(input_text_list)
            for idx, input_text in enumerate(tqdm(input_text_list)):
                input_text = input_text_list[idx]
                entity_hint_str = entity_hint_list[idx]
                relation_hint_str = relation_hint_list[idx]

                print(f"[OIE-refine] Processing idx={idx}")
                refined_oie_triplets = extractor.extract(
                    input_text,
                    oie_refinement_few_shot_examples_str,
                    oie_refinement_prompt_template_str,
                    entity_hint_str,
                    relation_hint_str,
                    index=idx,
                )
                self._refine_calls["oie_pass2"] += 1
                print(f"[OIE-refine] Done idx={idx} | triplets={len(refined_oie_triplets)}")
                # fall back to the previous-iteration triplets if the refine pass parsed empty
                if not refined_oie_triplets:
                    refined_oie_triplets = previous_extracted_triplets_list[idx]
                oie_triples_list.append(refined_oie_triplets)
        else:
            # Normal OIE — optionally with PER-ITEM IMMEDIATE refine: right after OIE
            # pass-1 of an item, re-extract THAT item with a local EE-merged entity hint
            # (and a relation hint if a schema already exists). Differs from end-to-end
            # refine only in granularity (per item vs per full pass) and timing (the
            # relation hint sees only the schema available NOW — empty in Mode 1 at OIE).
            logger.info("Running OIE...")
            oie_few_shot_examples_str = open(self.oie_few_shot_example_file_path).read()
            oie_few_shot_prompt_template_str = open(self.oie_prompt_template_file_path).read()

            # KATE — dynamic per-input few-shot (DECISIONS 2026-07-03b/c). OFF unless
            # OIE_KATE_POOL is set (default byte-identical). An explicitly-labeled
            # "dynamic-primed" DIAGNOSIS/ablation arm: only the demo CONTENT varies
            # per input; count/format/template stay fixed.
            _kate = None
            _kate_pool = os.environ.get("OIE_KATE_POOL", "").strip()
            if _kate_pool:
                from soda.utils.kate_selector import KateSelector
                _kate = KateSelector(_kate_pool, k=int(os.environ.get("OIE_KATE_K", "6")))
                logger.info(f"[OIE] KATE ON (dynamic-primed ablation arm): "
                            f"pool={_kate_pool} k={_kate.k}")

            # TODO(refine, RESEARCH_QUESTIONS.md §8.3): NOT yet validated on the server.
            # The refine OIE path was broken (JSON template vs list parser) until 2026-06-19;
            # smoke-test on webnlg_mini_20 (parse OK? EE/merge return entities? hints sane?)
            # before trusting any refine number.
            per_item_refine = getattr(self, "refine_per_item", False)
            refine_helpers = oie_refine_fs = oie_refine_tmpl = None
            if per_item_refine:
                logger.info("[OIE] PER-ITEM refine ON (2 OIE passes/item + EE+merge hint).")
                oie_refine_fs = open(self.oie_r_few_shot_example_file_path).read()
                oie_refine_tmpl = open(self.oie_r_prompt_template_file_path).read()
                refine_helpers = self._load_refine_helpers()

            # Mid-stage resume (opt-in EDC_RESUME_INPLACE=1; default OFF -> byte-identical).
            # OIE is per-item independent + greedy-deterministic, so reloading a partial
            # oie_total.json and skipping done items yields the identical result. RESUME_UPGRADE_PLAN.md.
            _start_idx = 0
            if os.environ.get("EDC_RESUME_INPLACE") == "1" and self.iter_dir:
                _rp = os.path.join(self.iter_dir, "oie_total.json")
                if os.path.exists(_rp):
                    try:
                        _prev = json.load(open(_rp, encoding="utf-8"))
                        if isinstance(_prev, list) and len(_prev) <= len(input_text_list):
                            oie_triples_list = list(_prev)
                            _start_idx = len(_prev)
                            print(f"[OIE] resume-inplace: {_start_idx}/{len(input_text_list)} items already done, continuing")
                    except Exception as _e:
                        logger.warning(f"[OIE] resume-inplace load failed ({_e}); starting fresh")

            for idx in range(_start_idx, len(input_text_list)):
                input_text = input_text_list[idx]
                print(f"[OIE] Processing idx={idx}")
                _fs = _kate.render(input_text) if _kate else oie_few_shot_examples_str
                oie_triples = extractor.extract(input_text, _fs,
                                                oie_few_shot_prompt_template_str, index=idx)
                self._refine_calls["oie_pass1"] += 1

                if per_item_refine:
                    ent_hint, rel_hint = self._build_item_hint(
                        input_text, oie_triples, refine_helpers, relation_example_dict={})
                    refined = extractor.extract(input_text, oie_refine_fs, oie_refine_tmpl,
                                                ent_hint, rel_hint, index=idx)
                    self._refine_calls["oie_pass2"] += 1
                    if refined:   # keep the recovery pass; fall back to pass-1 if it parsed empty
                        oie_triples = refined

                print(f"[OIE] Done idx={idx} | triplets={len(oie_triples)}")
                oie_triples_list.append(oie_triples)

                # periodic checkpoint + VRAM/RAM cleanup so a crash doesn't lose
                # everything and memory doesn't creep up over a long dataset
                if self.iter_dir and (idx + 1) % 100 == 0:
                    _checkpoint_json(oie_triples_list,
                                     os.path.join(self.iter_dir, "oie_total.json"))
                    _free_caches()

            if per_item_refine and refine_helpers is not None:
                self._free_refine_helpers(refine_helpers)

        logger.info("OIE finished.")
        
        # SAVE TOTAL
        if self.iter_dir:
            path = os.path.join(self.iter_dir, "oie_total.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(oie_triples_list, f, indent=2, ensure_ascii=False)
            # LLM-call cost of this OIE pass (for none/per-item/end-to-end comparison)
            calls = getattr(self, "_refine_calls", {})
            calls["total_llm_calls"] = sum(v for k, v in calls.items() if k != "total_llm_calls")
            calls["n_items"] = len(input_text_list)
            with open(os.path.join(self.iter_dir, "oie_refine_cost.json"), "w", encoding="utf-8") as f:
                json.dump(calls, f, indent=2)
            print(f"[OIE] LLM-call cost: {calls}")

            # Diagnosis mode (KATE / OIE-only ablations): stop cleanly after the OIE
            # artifact is saved — SD/SC/EC are not needed for the extraction-level
            # verdict (oie_miss_diagnosis reads oie_total.json). OFF by default.
            if os.environ.get("EDC_STOP_AFTER_OIE", "0") == "1":
                print("[OIE] EDC_STOP_AFTER_OIE=1 -> stopping after oie_total.json "
                      "(diagnosis mode; SD/SC/EC skipped).")
                raise SystemExit(0)

        # 
        if free_model:
            logger.info(f"Freeing model {self.oie_llm_name} as it is no longer needed")
            llm_utils.free_model(oie_model, oie_tokenizer)
            self._evict_model_cache(self.oie_llm_name)

        return oie_triples_list, entity_hint_list, relation_hint_list

    
    def schema_definition(self, input_text_list, oie_results_list, use_sd_general=True, free_model=False , pipeline_logger=None):
        assert len(input_text_list) == len(oie_results_list)
    
        # LOAD MODEL
        if not llm_utils.is_model_openai(self.sd_llm_name):
            sd_model, sd_tokenizer = self.load_model(self.sd_llm_name, "hf", role="sd")
            #schema_definer = SchemaDefiner(model=sd_model, tokenizer=sd_tokenizer)
            schema_definer = SchemaDefiner(model=sd_model, tokenizer=sd_tokenizer, use_sd_general=use_sd_general
                                           , pipeline_logger=pipeline_logger,iter_dir=self.iter_dir,
                                           sd_temperature=self.sd_temperature)
        else:
            sd_model, sd_tokenizer = None, None
            schema_definer = SchemaDefiner(openai_model=self.sd_llm_name, sd_temperature=self.sd_temperature)
        if self.sd_temperature > 0:
            logger.info(f"[SD] sampling ON (temperature={self.sd_temperature}) — variance source; OIE stays greedy")
    
        # LOAD PROMPTS
        sd1_prompt = open(self.sd1_entity_template_file_path).read()
        sd1_fs = open(self.sd1_entity_few_shot_example_file_path).read()

        sd2a_general_prompt= open(self.sd2a_generalrelation_template_file_path).read()
        sd2a_general_fs= open(self.sd2a_generalrelation_few_shot_example_file_path).read()

        # v2: EDC-style (Type A) general-def pass (→ general_definition_edc, case3).
        # REQUIRED whenever the general-def step runs (use_sd_general): both its OWN Type-A
        # template AND few-shot must exist. NO fallback — reusing the abstract template would
        # silently produce abstract-styled edc defs and corrupt the case3/case4 ablation.
        sd2a_general_edc_fs = None
        sd2a_general_edc_prompt = None
        if use_sd_general:
            _edc_tpl_path = self.sd2a_edc_generalrelation_template_file_path
            _edc_fs_path = self.sd2a_edc_generalrelation_few_shot_example_file_path
            if not _edc_tpl_path or not os.path.exists(_edc_tpl_path):
                raise FileNotFoundError(
                    f"SD2a-EDC Type-A template missing ({_edc_tpl_path}); required for "
                    f"case3 (general_definition_edc). Create it — no fallback is applied."
                )
            if not _edc_fs_path or not os.path.exists(_edc_fs_path):
                raise FileNotFoundError(
                    f"SD2a-EDC few-shot missing ({_edc_fs_path}); required for case3 "
                    f"(general_definition_edc). Create it — no fallback is applied."
                )
            sd2a_general_edc_prompt = open(_edc_tpl_path).read()
            sd2a_general_edc_fs = open(_edc_fs_path).read()

        sd2_prompt = open(self.sd2b_relation_template_file_path).read()
        sd2_fs = open(self.sd2b_relation_few_shot_example_file_path).read()
        
        sd3_prompt = open(self.sd3_clasy_template_file_path).read()
        sd3_fs = open(self.sd3_clasy_few_shot_example_file_path).read()

        results = []

        # Mid-stage resume (opt-in EDC_RESUME_INPLACE=1; default OFF -> byte-identical).
        # SD is per-item independent (define_schema_full uses only this item's text+triples;
        # cross-item schema growth is in SC, not SD) + greedy -> skipping done items is safe.
        _sd_start = 0
        if os.environ.get("EDC_RESUME_INPLACE") == "1" and self.iter_dir:
            _rp = os.path.join(self.iter_dir, "sd_total.json")
            if os.path.exists(_rp):
                try:
                    _prev = json.load(open(_rp, encoding="utf-8"))
                    if isinstance(_prev, list) and len(_prev) <= len(oie_results_list):
                        results = list(_prev)
                        _sd_start = len(_prev)
                        print(f"[SD] resume-inplace: {_sd_start}/{len(oie_results_list)} items already done, continuing")
                except Exception as _e:
                    logger.warning(f"[SD] resume-inplace load failed ({_e}); starting fresh")

        for idx in range(_sd_start, len(oie_results_list)):
            triples = oie_results_list[idx]
            logger.info("Running Refined SD...")
            
            print(f"[SD] Processing idx={idx}")
            text = input_text_list[idx]

            if not triples:
                results.append({
                    "entities": [],
                    "relations": []
                })
                continue
    
            try:
                sd = schema_definer.define_schema_full(
                    text, triples,
                    sd1_prompt, sd1_fs,
                    sd2_prompt, sd2_fs,
                    sd3_prompt=sd3_prompt, sd3_fs=sd3_fs,
                    sd_general_prompt=sd2a_general_prompt,
                    sd_general_fs=sd2a_general_fs,
                    sd_general_edc_prompt=sd2a_general_edc_prompt,
                    sd_general_edc_fs=sd2a_general_edc_fs,
                    index=idx
                )
            except Exception as e:
                logger.error(f"[Item {idx}] SD crashed: {e}")
                sd = {
                    "entities": [],
                    "relations": []
                }

            entities = sd.get("entities", [])
            relations = sd.get("relations", [])
            print(f"[SD] Done idx={idx} | triplets={len(triples)} | entities={len(entities)} | relations={len(relations)}")
            
            if not entities or not relations:
                logger.warning(f"[Item {idx}] Empty SD output: {sd}")
            
            results.append(sd)

            # periodic checkpoint + VRAM/RAM cleanup (SD does ~4 LLM calls/item,
            # so it is the heaviest stage — was OOM-killed mid-loop before)
            if self.iter_dir and (idx + 1) % 50 == 0:
                _checkpoint_json(results,
                                 os.path.join(self.iter_dir, "sd_total.json"))
                _free_caches()

        logger.info("SD finished.")
        # ================= SAVE TOTAL =================
        if self.iter_dir:
            path = os.path.join(self.iter_dir, "sd_total.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        if free_model and sd_model is not None:
            llm_utils.free_model(sd_model, sd_tokenizer)
    
        return results

    
    def schema_canonicalization(
        self,
        input_text_list,
        oie_triplets_list,
        sd_dict_list,
        cases=None,
    ):
    
        if cases is None:
            cases = list(SchemaCanonicalizer.SC_ABLATION_CONFIG.keys())
    
        sc_embedder = self.load_model(self.sc_embedder_name, "sts")
    
        # clone schema per case
        schema_per_case = {
            case: copy.deepcopy(self.schema)
            for case in cases
        }

        results = {case: [] for case in cases}

        # ---- mid-stage resume (opt-in EDC_RESUME_INPLACE=1; default OFF -> byte-identical) ----
        # SC is schema-stateful, so the checkpoint (sc_resume.json) stores BOTH the per-case
        # results AND the evolved per-case schema. Restoring the schema HERE, before the
        # SchemaCanonicalizer objects are constructed, means __init__ rebuilds the embedding
        # index from the restored schema, so a resumed run is identical to an uninterrupted one
        # (greedy). Up to 49 items just before a crash may redo (checkpoint modulo); their KG
        # output is rebuilt fresh so there is no duplication. RESUME_UPGRADE_PLAN.md Phase 2.
        _sc_start_idx = 0
        _sc_resume_on = os.environ.get("EDC_RESUME_INPLACE") == "1" and bool(self.iter_dir)
        _sc_ckpt_path = os.path.join(self.iter_dir, "sc_resume.json") if self.iter_dir else None
        if _sc_resume_on and _sc_ckpt_path and os.path.exists(_sc_ckpt_path):
            try:
                _prev = json.load(open(_sc_ckpt_path, encoding="utf-8"))
                if (isinstance(_prev, dict)
                        and _prev.get("n_items") == len(input_text_list)
                        and list(_prev.get("cases", [])) == list(cases)
                        and 0 <= _prev.get("done", 0) <= len(input_text_list)):
                    results = {case: list(_prev["results"][case]) for case in cases}
                    for case in cases:
                        schema_per_case[case] = _prev["schema_per_case"][case]
                    _sc_start_idx = _prev["done"]
                    print(f"[SC] resume-inplace: {_sc_start_idx}/{len(input_text_list)} "
                          f"items already done, continuing")
                else:
                    print("[SC] resume-inplace: checkpoint mismatch "
                          "(n_items/cases changed) -> starting fresh")
            except Exception as _e:
                logger.warning(f"[SC] resume-inplace load failed ({_e}); starting fresh")

        # init SC objects
        sc_objects = {}
        for case in cases:
            self.sc_config["ablation_case"] = case
    
            if llm_utils.is_model_openai(self.sc_llm_name):
                sc_objects[case] = SchemaCanonicalizer(
                    schema_per_case[case],
                    sc_embedder,
                    verify_openai_model=self.sc_llm_name,
                    sc_config=self.sc_config,
                )
            else:
                model, tok = self.load_model(self.sc_llm_name, "hf", role="sc")
                sc_objects[case] = SchemaCanonicalizer(
                    schema_per_case[case],
                    sc_embedder,
                    verify_model=model,
                    verify_tokenizer=tok,
                    sc_config=self.sc_config,
                )
    
        # load prompt templates + path
        prompt_templates = {
            "case1": (open(self.sc_template_option1_file_path).read(), self.sc_template_option1_file_path),
            "case2": (open(self.sc_template_option2_file_path).read(), self.sc_template_option2_file_path),
            "case3": (open(self.sc_template_option3_file_path).read(), self.sc_template_option3_file_path),
            "case4": (open(self.sc_template_option4_file_path).read(), self.sc_template_option4_file_path),
        }

        # ---- optional per-CASE batching of the LLM MCQ (item-13 speed-up) ----
        # Default OFF (sc_batch_size<=1) -> the per-case sc.canonicalize() path below is
        # used unchanged (byte-identical). When ON, the LLM call of all LLM-needing cases
        # of ONE triplet is batched (chunked by sc_batch_size). Cases have independent
        # schema clones, so this is exactly equivalent (DECISIONS 2026-06-27c). HF only;
        # the OpenAI SC path is never batched here. NOT batched across triplets — within
        # an item the schema grows per triplet, so that would change retrieval candidates.
        sc_batch_size = int(self.sc_config.get("sc_batch_size", 1) or 1)
        sc_batch_on = (not llm_utils.is_model_openai(self.sc_llm_name)) and sc_batch_size > 1
        sc_model = sc_tok = None
        if sc_batch_on:
            sc_model, sc_tok = self.load_model(self.sc_llm_name, "hf", role="sc")
            logger.info(f"[SC] per-case LLM batching ON (sc_batch_size={sc_batch_size})")

        # =========================
        # LOOP SAMPLE  (range-based so mid-stage resume can skip already-done items)
        # =========================
        for i in range(_sc_start_idx, len(input_text_list)):
            text = input_text_list[i]
            print(f"[SC] Processing idx={i}")
            
            triplets = oie_triplets_list[i]
            rel_map = {r["relation"]: r for r in sd_dict_list[i].get("relations", [])}
    
            sample_outputs_by_case = {
                case: {"text": text, "triplets": []}
                for case in cases
            }
    
            triplet_blocks = []
            triplet_blocks_vertical = []
    
            # =========================
            # LOOP TRIPLET (OUTER)
            # =========================
            for triplet in triplets:
    
                if len(triplet) < 3:
                    continue
    
                h, r, t = triplet
                case_data_map = {}

                # When batching is ON, prepare every case + run the LLM MCQ for all
                # LLM-needing cases of THIS triplet in batched generate() calls. Returns
                # prepared_by_case + raw_by_case[case]=(raw_output, llm_ok). None when OFF.
                prepared_by_case, raw_by_case = (
                    self._sc_batch_triplet_cases(
                        text, triplet, r, cases, sc_objects, prompt_templates,
                        rel_map, sc_model, sc_tok, sc_batch_size)
                    if sc_batch_on else (None, None)
                )

                # =====================
                # LOOP CASE
                # =====================
                for case in cases:

                    sc = sc_objects[case]
                    schema_case = schema_per_case[case]

                    # `raw` (the SD relation record, or an empty default) is used by BOTH
                    # the canonicalize call below and the schema-mutation block further
                    # down, so compute it once here for both the batched and default paths.
                    raw = rel_map.get(r, {
                        "relation": r,
                        "definition": "",
                        "head_type": "",
                        "tail_type": ""
                    })

                    if prepared_by_case is not None:
                        # batched path: parse the pre-computed (batched) LLM output
                        prepared = prepared_by_case[case]
                        raw_output, llm_ok = raw_by_case.get(case, (None, True))
                        final_rel, cand, scores, debug = sc.finalize(
                            prepared, raw_output, llm_ok=llm_ok)
                    else:
                        # default path (unchanged): one sequential canonicalize() per case
                        cfg = SchemaCanonicalizer.SC_ABLATION_CONFIG[case]
                        prompt_key = cfg["prompt_key"]

                        if prompt_key is None:
                            # case1 (no-LLM embedding+threshold) needs no prompt
                            prompt_template, prompt_path = None, None
                        elif prompt_key not in prompt_templates:
                            raise ValueError(f"Missing prompt for {prompt_key}")
                        else:
                            prompt_template, prompt_path = prompt_templates[prompt_key]

                        rel_info = sc.strip_rel_info_by_case(raw)

                        final_rel, cand, scores, debug = sc.canonicalize(
                            text,
                            triplet,
                            rel_info,
                            prompt_template,
                            prompt_key=prompt_key,
                            prompt_path=prompt_path
                        )

                    is_new = False
                    allow_new = self.sc_config.get("allow_new_schema", True)

                    if final_rel is None:   # LLM says NEW / parse-fail
                        if allow_new:
                            final_rel = r
                            is_new = True

                            if final_rel not in schema_case["relation_types"]:
                                # init aliases field
                                raw_copy = copy.deepcopy(raw)
                                if "aliases" not in raw_copy:
                                    raw_copy["aliases"] = []

                                schema_case["relation_types"][final_rel] = raw
                                sc._add_relation_embedding(final_rel, raw)
                        else:
                            # FROZEN SCHEMA (mode 3): do not add. Map to the
                            # closest existing candidate so the triplet stays
                            # grounded in the fixed schema. None if schema empty.
                            cand_keys = list(cand.keys())
                            if cand_keys:
                                final_rel = cand_keys[0]
                                is_new = False
                                schema_utils.add_relation_alias(
                                    schema_case=schema_case,
                                    canonical_rel=final_rel,
                                    raw_rel=r,
                                )
                                debug["decision"] = f"{debug.get('decision')}|frozen_top1"
                            else:
                                final_rel = None
                                is_new = False
                                debug["decision"] = f"{debug.get('decision')}|frozen_unmatched"

                    else:   # REUSE RELATION
                        is_new = False
                        schema_utils.add_relation_alias(
                            schema_case=schema_case,
                            canonical_rel=final_rel,
                            raw_rel=r
                        )
                        
    
                    # save result
                    sample_outputs_by_case[case]["triplets"].append({
                        "input_triplet": triplet,
                        "pred_relation": final_rel,
                        "top_k": list(cand.keys()),
                        "debug": debug,
                    })
    
                    case_data_map[case] = {
                        "pred": final_rel,
                        "top_k": debug["top_k"],
                        "scores": debug["scores"],
                        "is_new_relation": is_new,
                        "decision": debug["decision"],
                        "llm_choice": debug["llm_choice"],
                        "prompt_key": debug["prompt_key"],
                        "prompt_path": debug["prompt_path"],
                    }
    
                # =====================
                # BUILD LOG
                # =====================
                block_v = log_utils.build_triplet_table_vertical(
                    triplet,
                    case_data_map,
                    cases,
                    top_k=5
                )
    
                triplet_blocks_vertical.append(block_v)
    
            # =====================
            # SC DEBUG TRACE (JSONL, one record per idx+case; gated by EDC_DEBUG_LOGS)
            # Replaces the old per-item sc_mcp_logs/*.txt + sc_logs2/*.txt dumps.
            # =====================
            for case in cases:
                trips = []
                for t_idx, t_data in enumerate(sample_outputs_by_case[case]["triplets"]):
                    dbg = t_data["debug"]
                    trips.append({
                        "t_idx": t_idx,
                        "input_triplet": t_data["input_triplet"],
                        "pred_relation": t_data["pred_relation"],
                        "top_k": dbg.get("top_k"),
                        "scores": [round(float(s), 4) for s in (dbg.get("scores") or [])],
                        "llm_choice": dbg.get("llm_choice"),
                        "decision": dbg.get("decision"),
                        "prompt": dbg.get("prompt"),
                        "raw_output": dbg.get("raw_output"),
                    })
                debug_logger.jsonl_log(self.iter_dir, "sc_mcp_logs", {
                    "idx": i, "case": case, "text": text, "triplets": trips,
                })

            for case in cases:
                results[case].append(sample_outputs_by_case[case])

            schema_str = ", ".join(
                [f"{case}={len(schema_per_case[case]['relation_types'])}" for case in cases]
            )
            print(f"[SC] Done idx={i} | {schema_str}")

            # mid-stage resume checkpoint (opt-in; RESUME_UPGRADE_PLAN.md Phase 2).
            # Every 50 items + the final item, persist results + evolved schema.
            if _sc_resume_on and _sc_ckpt_path and (
                (i + 1) % 50 == 0 or (i + 1) == len(input_text_list)
            ):
                _sc_checkpoint(_sc_ckpt_path, i + 1, len(input_text_list),
                               cases, results, schema_per_case)

        return results, schema_per_case

    def _sc_batch_triplet_cases(self, text, triplet, r, cases, sc_objects,
                                prompt_templates, rel_map, sc_model, sc_tok, batch_size):
        """item-13: prepare every case for ONE triplet, then run the LLM MCQ for all
        LLM-needing cases in batched generate() calls (chunked by batch_size).

        Returns (prepared_by_case, raw_by_case) where raw_by_case[case] = (raw_output,
        llm_ok). Across cases the schema clones are independent, so batching here is
        exactly equivalent to the per-case sequential path (DECISIONS 2026-06-27c). The
        per-case prompt/rel_info resolution mirrors the default loop body verbatim."""
        prepared_by_case = {}
        for case in cases:
            sc = sc_objects[case]
            cfg = SchemaCanonicalizer.SC_ABLATION_CONFIG[case]
            prompt_key = cfg["prompt_key"]
            if prompt_key is None:
                prompt_template, prompt_path = None, None
            elif prompt_key not in prompt_templates:
                raise ValueError(f"Missing prompt for {prompt_key}")
            else:
                prompt_template, prompt_path = prompt_templates[prompt_key]
            raw = rel_map.get(r, {
                "relation": r, "definition": "", "head_type": "", "tail_type": "",
            })
            rel_info = sc.strip_rel_info_by_case(raw)
            prepared_by_case[case] = sc.prepare(
                text, triplet, rel_info, prompt_template,
                prompt_key=prompt_key, prompt_path=prompt_path)

        raw_by_case = {}
        llm_cases = [c for c in cases if prepared_by_case[c]["needs_llm"]]
        for s in range(0, len(llm_cases), batch_size):
            chunk = llm_cases[s:s + batch_size]
            prompts = [
                [{"role": "user", "content": prepared_by_case[c]["prompt"]}]
                for c in chunk
            ]
            try:
                outs = llm_utils.generate_completion_transformers_batch(
                    prompts, sc_model, sc_tok)
                for c, o in zip(chunk, outs):
                    raw_by_case[c] = (o, True)
            except Exception as e:
                logger.warning(f"[SC-batch] batched generate failed ({e}); fallback_top1")
                for c in chunk:
                    raw_by_case[c] = (None, False)
        return prepared_by_case, raw_by_case


    def entity_canonicalization(
        self,
        input_text_list,
        sd_dict_list,
        ec_cases=None,
    ):
        """
        Entity-type canonicalization pass (twin of schema_canonicalization for
        relations). For each sample, the unique fine-types produced by SD1 are
        mapped onto global entity types (or marked new). Runs independently of
        the 7 relation cases and only when enable_entity_canon=True.

        Returns:
            ec_results        : {ec_case: [ {text, entities:[...]}, ... ]}
            entity_schema_per_case : {ec_case: schema-with-grown-entity_types}
            entity_type_map_per_case : {ec_case: {entity_surface_name: canon_type}}
        """
        if ec_cases is None:
            ec_cases = list(EntityCanonicalizer.EC_ABLATION_CONFIG.keys())

        sc_embedder = self.load_model(self.sc_embedder_name, "sts")

        schema_per_case = {
            case: copy.deepcopy(self.schema)
            for case in ec_cases
        }

        ec_results = {case: [] for case in ec_cases}
        entity_type_map_per_case = {case: {} for case in ec_cases}

        # ---- mid-stage resume (opt-in EDC_RESUME_INPLACE=1; default OFF -> byte-identical) ----
        # EC is schema-stateful in the same way SC is, so the checkpoint (ec_resume.json) stores
        # the per-case results, the evolved entity schema AND the accumulated surface->type map.
        # Restoring the schema HERE, before the EntityCanonicalizer objects are constructed, means
        # __init__ rebuilds the embedding index from the restored schema, so a resumed run is
        # identical to an uninterrupted one (greedy). Up to 49 items just before a crash may redo
        # (checkpoint modulo); their output is rebuilt fresh so there is no duplication.
        # RESUME_UPGRADE_PLAN.md Phase 3.
        _ec_start_idx = 0
        _ec_resume_on = os.environ.get("EDC_RESUME_INPLACE") == "1" and bool(self.iter_dir)
        _ec_ckpt_path = os.path.join(self.iter_dir, "ec_resume.json") if self.iter_dir else None
        if _ec_resume_on and _ec_ckpt_path and os.path.exists(_ec_ckpt_path):
            try:
                _prev = json.load(open(_ec_ckpt_path, encoding="utf-8"))
                if (isinstance(_prev, dict)
                        and _prev.get("n_items") == len(input_text_list)
                        and list(_prev.get("cases", [])) == list(ec_cases)
                        and 0 <= _prev.get("done", 0) <= len(input_text_list)):
                    ec_results = {case: list(_prev["results"][case]) for case in ec_cases}
                    for case in ec_cases:
                        schema_per_case[case] = _prev["schema_per_case"][case]
                        entity_type_map_per_case[case] = dict(
                            _prev.get("entity_type_map_per_case", {}).get(case, {}))
                    _ec_start_idx = _prev["done"]
                    print(f"[EC] resume-inplace: {_ec_start_idx}/{len(input_text_list)} "
                          f"items already done, continuing")
                else:
                    print("[EC] resume-inplace: checkpoint mismatch "
                          "(n_items/cases changed) -> starting fresh")
            except Exception as _e:
                logger.warning(f"[EC] resume-inplace load failed ({_e}); starting fresh")

        # base ec_config inherits cluster settings from sc_config
        base_ec_config = {
            "top_k": self.sc_config.get("top_k", 5),
            "retrieval_mode": self.sc_config.get("retrieval_mode", "item"),
            "cluster_sim_threshold": self.sc_config.get("cluster_sim_threshold", 0.85),
            "cluster_top_m": self.sc_config.get("cluster_top_m", 2),
            "cluster_extra_k": self.sc_config.get("cluster_extra_k", 3),
            "parent_penalty": self.sc_config.get("parent_penalty", 0.2),
        }

        ec_objects = {}
        for case in ec_cases:
            ec_cfg = dict(base_ec_config)
            ec_cfg["ablation_case"] = case

            if llm_utils.is_model_openai(self.sc_llm_name):
                ec_objects[case] = EntityCanonicalizer(
                    schema_per_case[case],
                    sc_embedder,
                    verify_openai_model=self.sc_llm_name,
                    ec_config=ec_cfg,
                )
            else:
                model, tok = self.load_model(self.sc_llm_name, "hf", role="sc")
                ec_objects[case] = EntityCanonicalizer(
                    schema_per_case[case],
                    sc_embedder,
                    verify_model=model,
                    verify_tokenizer=tok,
                    ec_config=ec_cfg,
                )

        # prompt templates
        ec_prompt_templates = {
            "ec1": (open(self.ec_template_option1_file_path).read(), self.ec_template_option1_file_path),
            "ec2": (open(self.ec_template_option2_file_path).read(), self.ec_template_option2_file_path),
            "ec3": (open(self.ec_template_option3_file_path).read(), self.ec_template_option3_file_path),
        }

        for i in range(_ec_start_idx, len(input_text_list)):
            text = input_text_list[i]
            print(f"[EC] Processing idx={i}")

            ent_defs = sd_dict_list[i].get("entities", []) if i < len(sd_dict_list) else []

            # group by fine-type (canonicalize per TYPE, not per instance)
            # fine_type -> {"definition": desc, "parent": coarse, "members": [entity_name,...]}
            type_groups = {}
            for item in ent_defs:
                if not (isinstance(item, (list, tuple)) and len(item) == 4):
                    continue
                ent_name, coarse, fine, desc = item
                g = type_groups.setdefault(
                    fine, {"definition": desc, "parent": coarse, "members": []}
                )
                g["members"].append(ent_name)

            sample_outputs_by_case = {
                case: {"text": text, "entities": []}
                for case in ec_cases
            }

            for case in ec_cases:
                ec = ec_objects[case]
                schema_case = schema_per_case[case]
                cfg = EntityCanonicalizer.EC_ABLATION_CONFIG[case]
                prompt_template, prompt_path = ec_prompt_templates[cfg["prompt_key"]]

                for fine, g in type_groups.items():
                    query_info = {
                        "entity_type": fine,
                        "definition": g["definition"],
                        "parent": g["parent"],
                    }

                    final_type, cand, scores, debug = ec.canonicalize(
                        text,
                        query_info,
                        prompt_template,
                        prompt_key=cfg["prompt_key"],
                        prompt_path=prompt_path,
                    )

                    is_new = False
                    allow_new = self.sc_config.get("allow_new_schema", True)

                    if final_type is None:   # LLM says NEW / parse-fail
                        if allow_new:
                            final_type = fine
                            is_new = True
                            if final_type not in schema_case["entity_types"]:
                                new_info = {
                                    "definition": g["definition"],
                                    "parent": g["parent"],
                                    "attributes": {},
                                    "aliases": [],
                                }
                                schema_case["entity_types"][final_type] = new_info
                                ec._add_entity_embedding(final_type, new_info)
                        else:
                            # FROZEN SCHEMA (mode 3): map to closest existing type
                            cand_keys = list(cand.keys())
                            if cand_keys:
                                final_type = cand_keys[0]
                                is_new = False
                                schema_utils.add_entity_type_alias(
                                    schema_case=schema_case,
                                    canonical_type=final_type,
                                    raw_type=fine,
                                )
                                debug["decision"] = f"{debug.get('decision')}|frozen_top1"
                            else:
                                final_type = None
                                is_new = False
                                debug["decision"] = f"{debug.get('decision')}|frozen_unmatched"
                    else:                    # REUSE
                        schema_utils.add_entity_type_alias(
                            schema_case=schema_case,
                            canonical_type=final_type,
                            raw_type=fine,
                        )

                    # record surface-name -> canonical type
                    for m in g["members"]:
                        if final_type is not None:
                            entity_type_map_per_case[case][m] = final_type

                    sample_outputs_by_case[case]["entities"].append({
                        "input_fine_type": fine,
                        "pred_entity_type": final_type,
                        "members": g["members"],
                        "top_k": list(cand.keys()),
                        "is_new": is_new,
                        "decision": debug["decision"],
                    })

            # debug dump per case (JSONL, one record per idx+case; gated by EDC_DEBUG_LOGS)
            for case in ec_cases:
                debug_logger.jsonl_log(self.iter_dir, "ec_logs", {
                    "idx": i, "ec_case": case, "text": text,
                    "entities": sample_outputs_by_case[case]["entities"],
                })

            for case in ec_cases:
                ec_results[case].append(sample_outputs_by_case[case])

            schema_str = ", ".join(
                [f"{case}={len(schema_per_case[case]['entity_types'])}" for case in ec_cases]
            )
            print(f"[EC] Done idx={i} | {schema_str}")

            # mid-stage resume checkpoint (opt-in; RESUME_UPGRADE_PLAN.md Phase 3).
            # Every 50 items + the final item, persist results + evolved schema + type map.
            if _ec_resume_on and _ec_ckpt_path and (
                (i + 1) % 50 == 0 or (i + 1) == len(input_text_list)
            ):
                _ec_checkpoint(_ec_ckpt_path, i + 1, len(input_text_list),
                               ec_cases, ec_results, schema_per_case,
                               entity_type_map_per_case)

        return ec_results, schema_per_case, entity_type_map_per_case


    # ------------------------------------------------------------------
    # Refinement hint construction — refactored so BOTH end-to-end (batch)
    # and per-item-immediate refine share the same hint logic.
    #   entity hint = merge( previous-pass entities , a FRESH dedicated
    #                        entity-extraction (EE) pass )   <- recovers misses
    #   relation hint = previous relations + schema-retrieved relations
    #                   (only when a schema exists; Mode-1 OIE-time schema is empty)
    # ------------------------------------------------------------------
    def _load_refine_helpers(self):
        """Load EE model + schema retriever + EE/EM templates ONCE (shared)."""
        ee_fs = open(self.ee_few_shot_example_file_path).read()
        ee_tmpl = open(self.ee_template_file_path).read()
        em_tmpl = open(self.em_template_file_path).read()
        if not llm_utils.is_model_openai(self.ee_llm_name):
            ee_model, ee_tokenizer = self.load_model(self.ee_llm_name, "hf", role="ee")
            entity_extractor = EntityExtractor(model=ee_model, tokenizer=ee_tokenizer)
        else:
            ee_model = ee_tokenizer = None
            entity_extractor = EntityExtractor(openai_model=self.ee_llm_name)
        # Schema retriever only if the schema has relations. In Mode 1 the schema is
        # still EMPTY at OIE time (SC has not run yet) -> no relation hint, entity-only.
        schema_retriever = None
        sr_embedding_model = None
        if (self.schema or {}).get("relation_types"):
            sr_embedding_model = self.load_model(self.sr_embedder_name, "sts")
            schema_retriever = SchemaRetriever(self.schema, sr_embedding_model, None,
                                               finetuned_e5mistral=False)
        return {"ee_fs": ee_fs, "ee_tmpl": ee_tmpl, "em_tmpl": em_tmpl,
                "entity_extractor": entity_extractor, "schema_retriever": schema_retriever,
                "ee_model": ee_model, "ee_tokenizer": ee_tokenizer,
                "sr_embedding_model": sr_embedding_model}

    def _free_refine_helpers(self, helpers):
        try:
            if helpers.get("sr_embedding_model") is not None:
                llm_utils.free_model(helpers["sr_embedding_model"])
                self._evict_model_cache(self.sr_embedder_name)
            if helpers.get("ee_model") is not None:
                llm_utils.free_model(helpers["ee_model"], helpers["ee_tokenizer"])
                self._evict_model_cache(self.ee_llm_name)
        except Exception as e:
            logger.warning(f"_free_refine_helpers: {e}")

    def _build_item_hint(self, input_text_str, extracted_triplets, helpers,
                         relation_example_dict=None, include_relation_example="self"):
        """Build (entity_hint_str, relation_hint_str) for ONE item."""
        if not hasattr(self, "_refine_calls"):
            self._refine_calls = {"oie_pass1": 0, "oie_pass2": 0, "ee": 0, "merge": 0}
        relation_example_dict = relation_example_dict or {}
        entity_extractor = helpers["entity_extractor"]
        schema_retriever = helpers["schema_retriever"]

        previous_entities, previous_relations = set(), set()
        for t in extracted_triplets:
            if len(t) >= 3:
                previous_entities.add(t[0]); previous_entities.add(t[2]); previous_relations.add(t[1])
        previous_entities, previous_relations = list(previous_entities), list(previous_relations)

        # entity hint = merge(previous entities, a fresh dedicated EE pass)
        extracted_entities = entity_extractor.extract_entities(
            input_text_str, helpers["ee_fs"], helpers["ee_tmpl"])
        self._refine_calls["ee"] += 1
        merged_entities = entity_extractor.merge_entities(
            input_text_str, previous_entities, extracted_entities, helpers["em_tmpl"])
        self._refine_calls["merge"] += 1
        entity_hint_str = str(merged_entities)

        # relation hint = previous relations + schema-retrieved (when a schema exists)
        hint_relations = list(previous_relations)
        if schema_retriever is not None:
            for relation in schema_retriever.retrieve_relevant_relations(input_text_str):
                if relation not in hint_relations:
                    hint_relations.append(relation)
        rt = (self.schema or {}).get("relation_types", {})
        candidate_relation_str = ""
        for relation_idx, relation in enumerate(hint_relations):
            if relation not in rt:
                continue
            relation_definition = rt[relation].get("definition", "")
            candidate_relation_str += f"{relation_idx + 1}. {relation}: {relation_definition}\n"
            exs = relation_example_dict.get(relation) if include_relation_example == "self" else None
            if exs:
                ex = random.choice(exs)
                candidate_relation_str += (
                    f"""For example, {ex['triplet']} can be extracted from "{ex['text']}"\n""")
        return entity_hint_str, candidate_relation_str

    def construct_refinement_hint(
        self,
        input_text_list: List[str],
        extracted_triplets_list: List[List[List[str]]],
        include_relation_example="self",
        relation_top_k=10,
        free_model=False,
    ):
        """End-to-end (batch) hint for ALL items. Reuses _build_item_hint."""
        helpers = self._load_refine_helpers()
        relation_example_dict = {}
        if include_relation_example == "self":
            for idx in range(len(input_text_list)):
                for triplet in extracted_triplets_list[idx]:
                    if len(triplet) >= 3:
                        relation_example_dict.setdefault(triplet[1], []).append(
                            {"text": input_text_list[idx], "triplet": triplet})
        entity_hint_list, relation_hint_list = [], []
        for idx in tqdm(range(len(input_text_list))):
            eh, rh = self._build_item_hint(
                input_text_list[idx], extracted_triplets_list[idx], helpers,
                relation_example_dict, include_relation_example)
            entity_hint_list.append(eh)
            relation_hint_list.append(rh)
        if free_model:
            self._free_refine_helpers(helpers)
        return entity_hint_list, relation_hint_list

    # resume_from = None   # chạy full
    # resume_from = "sd"   # skip OIE
    # resume_from = "sc"   # skip OIE + SD
    def _collect_runtime_meta(self):
        """Server config + model/pipeline params + RAM/GPU usage, for the timing log.
        Fully defensive: any missing dependency is skipped, never breaks the run."""
        meta = {}
        try:
            import platform, socket
            meta["host"] = socket.gethostname()
            meta["platform"] = platform.platform()
            meta["python"] = platform.python_version()
            meta["cpu_count"] = os.cpu_count()
        except Exception:
            pass
        # System + process RAM
        try:
            import psutil
            vm = psutil.virtual_memory()
            meta["ram_total_gb"] = round(vm.total / 1e9, 1)
            meta["ram_used_gb"] = round((vm.total - vm.available) / 1e9, 1)
            meta["proc_rss_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
        except Exception:
            try:  # Linux fallback: peak RSS (ru_maxrss is kB on Linux)
                import resource
                meta["proc_peak_rss_gb"] = round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
            except Exception:
                pass
        # torch / CUDA / GPU memory
        try:
            import torch
            meta["torch"] = torch.__version__
            meta["cuda"] = torch.version.cuda
            if torch.cuda.is_available():
                i = torch.cuda.current_device()
                p = torch.cuda.get_device_properties(i)
                meta["gpu"] = p.name
                meta["gpu_total_gb"] = round(p.total_memory / 1e9, 1)
                meta["gpu_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated(i) / 1e9, 2)
                meta["gpu_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(i) / 1e9, 2)
        except Exception:
            pass
        # Model + pipeline config
        meta["models"] = {
            "oie": self.oie_llm_name, "sd": self.sd_llm_name, "sc": self.sc_llm_name,
            "sc_embedder": getattr(self, "sc_embedder_name", None),
        }
        meta["dtype"] = ("4bit" if os.environ.get("EDC_LOAD_IN_4BIT") == "1"
                         else "8bit" if os.environ.get("EDC_LOAD_IN_8BIT") == "1"
                         else "bf16/auto")
        meta["pipeline"] = {
            "retrieval_mode": self.sc_config.get("retrieval_mode"),
            "sd_temperature": getattr(self, "sd_temperature", 0.0),
            "allow_new_schema": self.sc_config.get("allow_new_schema"),
            "enable_entity_canon": self.sc_config.get("enable_entity_canon"),
            "enable_instance_canon": self.sc_config.get("enable_instance_canon"),
            "instance_canon_ec_case": self.sc_config.get("instance_canon_ec_case"),
            "enable_coref": getattr(self, "enable_coref", False),
        }
        return meta

    def entity_canonicalization_only(self, input_text_list, output_dir, iteration=0):
        """Run ONLY the entity-canonicalization (EC) pass + its dumps on a run whose SC is
        already finished on disk, then update stage_timing.json.

        WHY THIS EXISTS (2026-07-25): `RESUME_FROM=sc` cannot be used to "just finish EC" on a
        Mode-1 run, because the SC path constructs the SchemaCanonicalizer objects, and their
        constructor VALIDATES the schema it is handed. A fresh run passes that check trivially
        (the discovered schema starts empty), but a *restored* schema contains relations the LLM
        minted during the run, some with an empty definition — so the validator correctly refuses
        it (`SC case 'case3...' requires non-empty 'general_definition_edc'`). Filling those
        definitions in would corrupt a RESULT artifact (the discovered schema), so instead we skip
        SC construction entirely: EC only needs the SD output + the saved per-case SC results.

        Reads from `<output_dir>/iter{iteration}`: `sd_total.json` and `case*.json`
        (as written by save_sc_results) plus `canon_schema_{case}.json` for the KG re-dump.
        Writes the same artifacts the in-pipeline EC step writes: `entity_canon_{ec_case}.json`,
        `canon_entity_schema_{ec_case}.json`, `canon_kg_with_entity_{case}.json`.
        """
        import glob
        iter_dir = f"{output_dir}/iter{iteration}"
        self.iter_dir = iter_dir
        if not os.path.isdir(iter_dir):
            raise SystemExit(f"[ec-only] no iter dir: {iter_dir}")

        sd_dict_list = load_total_file(os.path.join(iter_dir, "sd_total.json"))
        print(f"[ec-only] loaded SD for {len(sd_dict_list)} items")

        # per-case SC results, in the canonical case order (case1..case9 by leading index)
        case_files = sorted(glob.glob(os.path.join(iter_dir, "case[0-9]*.json")),
                            key=lambda p: int(os.path.basename(p).split("_")[0].replace("case", "")))
        if not case_files:
            raise SystemExit(f"[ec-only] no case*.json (finished SC) in {iter_dir}")
        sc_results, schema_per_case, cases = {}, {}, []
        for cf in case_files:
            case = os.path.basename(cf)[:-5]
            cases.append(case)
            data = json.load(open(cf, encoding="utf-8"))
            # save_sc_results writes input/pred/top_k; the extractors expect input_triplet/pred_relation
            sc_results[case] = [
                {"text": s.get("text", ""),
                 "triplets": [{"input_triplet": t.get("input_triplet", t.get("input")),
                               "pred_relation": t.get("pred_relation", t.get("pred")),
                               "top_k": t.get("top_k", [])}
                              for t in s.get("triplets", [])]}
                for s in data
            ]
            scf = os.path.join(iter_dir, f"canon_schema_{case}.json")
            schema_per_case[case] = json.load(open(scf, encoding="utf-8")) if os.path.exists(scf) else {}
        print(f"[ec-only] loaded SC results for {len(cases)} cases: {cases}")

        _t0 = time.perf_counter()
        _tok0 = llm_utils.get_token_usage()
        ec_results, entity_schema_per_case, entity_type_map_per_case = \
            self.entity_canonicalization(input_text_list, sd_dict_list)
        ec_sec = round(time.perf_counter() - _t0, 1)
        ec_tokens = llm_utils.token_usage_delta(_tok0)

        for ec_case, samples in ec_results.items():
            with open(f"{iter_dir}/entity_canon_{ec_case}.json", "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            with open(f"{iter_dir}/canon_entity_schema_{ec_case}.json", "w", encoding="utf-8") as f:
                json.dump(entity_schema_per_case[ec_case], f, indent=2, ensure_ascii=False)

        default_ec = list(ec_results.keys())[0]
        entity_type_map = entity_type_map_per_case[default_ec]
        for case in cases:
            schema_utils.dump_canon_kg_with_entity_types(
                extract_pred_triplets(sc_results, case),
                schema_per_case[case],
                entity_type_map,
                f"{iter_dir}/canon_kg_with_entity_{case}.json",
            )

        # stage_timing: keep whatever a previous partial run recorded, add/overwrite the EC block
        stpath = os.path.join(iter_dir, "stage_timing.json")
        timing = {}
        if os.path.exists(stpath):
            try:
                timing = json.load(open(stpath, encoding="utf-8"))
            except Exception:
                timing = {}
        ni = max(len(input_text_list), 1)
        timing.update({"iteration": iteration, "n_items": len(input_text_list),
                       "ec_sec": ec_sec, "ec_tokens": ec_tokens,
                       "ec_only_rerun": True})
        timing.setdefault("per_item_sec", {})["ec_sec"] = round(ec_sec / ni, 3)
        timing["runtime"] = self._collect_runtime_meta()
        with open(stpath, "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2, ensure_ascii=False)
        print(f"[ec-only] DONE: EC {ec_sec}s ({ec_sec/ni:.2f} s/item) -> {iter_dir}")
        return ec_results

    def extract_kg(
        self,
        input_text_list,
        gold_triplets_list,
        output_dir=None,
        refinement_iterations=0,
        freeze_iter=None,
        resume_from=None,
        reuse_stage_dir=None,
    ):

        if output_dir:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("SODA starts running...")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        # Reset token accounting so stage_timing token totals are per-run.
        try:
            llm_utils.reset_token_usage()
        except Exception:
            pass
    
        cases = [
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
        # Optional SC-case subset (comma-separated), e.g.
        #   EDC_SC_CASES=case4_name_gendef_abstract
        # for the case4 seed-def regen re-run (re-runs ONE case on a reused OIE+SD
        # base at ~1/9 SC cost). Unset (default) = all 9 -> byte-identical.
        # Downstream eval skips missing case files, so a subset dir evals cleanly.
        _cases_env = os.environ.get("EDC_SC_CASES", "").strip()
        if _cases_env:
            _sel = [c.strip() for c in _cases_env.split(",") if c.strip()]
            _bad = [c for c in _sel if c not in SchemaCanonicalizer.SC_ABLATION_CONFIG]
            if _bad:
                raise ValueError(
                    f"EDC_SC_CASES contains unknown case(s) {_bad}; "
                    f"valid: {list(SchemaCanonicalizer.SC_ABLATION_CONFIG)}"
                )
            cases = _sel
            print(f"[SC] EDC_SC_CASES override -> running only {cases}")
    
        triplets_from_last_iteration = None
    
        for iteration in range(refinement_iterations + 1):
    
            logger.info(f"Iteration {iteration}")
    
            iter_dir = f"{output_dir}/iter{iteration}"
            pathlib.Path(iter_dir).mkdir(parents=True, exist_ok=True)
            self.iter_dir = iter_dir
    
            pipeline_logger = PipelineLogger(
                save_path=f"{iter_dir}/pipeline_log.json"
            )
            # ================= RESUME MODE =================
            if resume_from:
                print(f"🚀 RESUME MODE from {resume_from}")

            # Optionally pull cached OIE/SD totals from ANOTHER run's iter dir
            # (OIE+SD are independent of RUN_MODE / retrieval / case, so they can
            # be computed once and reused across the 16 SC variants). Copy them
            # into THIS run's iter_dir so the run stays self-contained (eval reads
            # oie_total.json from its own dir).
            if reuse_stage_dir and resume_from in ["sd", "sc"]:
                import shutil
                src_iter = os.path.join(reuse_stage_dir, f"iter{iteration}")
                src = src_iter if os.path.isdir(src_iter) else reuse_stage_dir

                needed = ["oie_total.json"]
                if resume_from == "sc":
                    needed.append("sd_total.json")
                missing = [fn for fn in needed
                           if not os.path.exists(os.path.join(src, fn))]
                if missing:
                    raise FileNotFoundError(
                        f"reuse_stage_dir is incomplete: missing {missing} in {src}. "
                        f"The base run probably did not finish that stage "
                        f"(e.g. SD was OOM-killed). Re-run the base to completion "
                        f"first (you can use --resume_from sd to skip OIE)."
                    )
                for fn in needed:
                    s, d = os.path.join(src, fn), os.path.join(iter_dir, fn)
                    # IN-PLACE reuse: FULL_RESUME=sd/sc points REUSE_STAGE_DIR at the cell's OWN dir
                    # (run_track3_scale_reuse.sh: "REUSE_STAGE_DIR defaults to the full cell dir (in
                    # place)"), so src == iter_dir and the copy is a no-op that shutil.copy REFUSES
                    # with SameFileError. Skip it: the file is already where it needs to be.
                    # (Found 2026-07-17 on the H100 — FULL_RESUME=sd had never actually been run.)
                    if os.path.exists(d) and os.path.samefile(s, d):
                        print(f"♻️  {fn} already in place ({iter_dir}) — in-place resume, no copy")
                        continue
                    shutil.copy(s, d)
                    print(f"♻️  Reused {fn} from {src}")

            # ================= COREF (optional, pre-OIE) =================
            # Rewrite each item so OIE/SD/SC see resolved mentions. The ORIGINAL
            # input_text_list is untouched (chunk linkage / eval use it). When
            # resuming, reuse the base run's coref_total.json so SD/SC see the
            # same text the OIE triples were extracted from.
            if getattr(self, "enable_coref", False):
                cpath = os.path.join(iter_dir, "coref_total.json")
                if resume_from in ["sd", "sc"] and os.path.exists(cpath):
                    working_text_list = [
                        r.get("coref_text") or input_text_list[i]
                        for i, r in enumerate(load_total_file(cpath))
                    ]
                    print(f"🔥 Loading coref from total file ({len(working_text_list)} items)")
                else:
                    working_text_list = self.coref(input_text_list, pipeline_logger=pipeline_logger)
            else:
                working_text_list = input_text_list

            # Per-iteration wall-clock timing (→ iter_dir/stage_timing.json). Lets us
            # measure throughput on a new GPU and extrapolate run time for big corpora.
            _timing = {"iteration": iteration, "n_items": len(working_text_list)}

            # ================= OIE =================
            if resume_from in ["sd", "sc"]:
                print("🔥 Loading OIE from total file...")
                oie_triplets_list = load_total_file(
                    os.path.join(iter_dir, "oie_total.json")
                )
                _timing["oie_sec"] = "reused"
            else:
                _t0 = time.perf_counter()
                _tok0 = llm_utils.get_token_usage()
                oie_triplets_list, _, _ = self.oie(
                    working_text_list,
                    previous_extracted_triplets_list=triplets_from_last_iteration,
                    pipeline_logger=pipeline_logger
                )
                _timing["oie_sec"] = round(time.perf_counter() - _t0, 1)
                _timing["oie_tokens"] = llm_utils.token_usage_delta(_tok0)
            _timing["n_triplets"] = sum(len(t) for t in oie_triplets_list)

            # ================= SD =================
            if resume_from == "sc":
                print("🔥 Loading SD from total file...")
                sd_dict_list = load_total_file(
                    os.path.join(iter_dir, "sd_total.json")
                )
                _timing["sd_sec"] = "reused"
            else:
                _t0 = time.perf_counter()
                _tok0 = llm_utils.get_token_usage()
                sd_dict_list = self.schema_definition(
                    working_text_list,
                    oie_triplets_list,
                    use_sd_general=True,
                    pipeline_logger=pipeline_logger
                )
                _timing["sd_sec"] = round(time.perf_counter() - _t0, 1)
                _timing["sd_tokens"] = llm_utils.token_usage_delta(_tok0)
    
            # ================= SC =================
            freeze_this_iter = (
                freeze_iter is not None and iteration >= freeze_iter
            )
            self.sc_config["allow_new_schema"] = (
                self.allow_new_schema_base and not freeze_this_iter
            )
            print(
                f"[MODE] allow_new_schema={self.sc_config['allow_new_schema']} "
                f"(base={self.allow_new_schema_base}, frozen_iter={freeze_this_iter})"
            )
    
            _t0 = time.perf_counter()
            _tok0 = llm_utils.get_token_usage()
            sc_results, schema_per_case = self.schema_canonicalization(
                working_text_list,
                oie_triplets_list,
                sd_dict_list,
                cases=cases
            )
            _timing["sc_sec"] = round(time.perf_counter() - _t0, 1)
            _timing["sc_tokens"] = llm_utils.token_usage_delta(_tok0)
            _timing["n_sc_cases"] = len(cases)

            # ================= SAVE RAW =================
            save_sc_results(sc_results, iter_dir)
    
            # ================= SAVE KG PER CASE (JSON and TXT) =================
            triplets_by_case = {}
    
            for case in cases:
    
                triplets_case = extract_pred_triplets(sc_results, case)
                triplets_by_case[case] = triplets_case
    
                schema_case = schema_per_case[case]   
    
                schema_utils.dump_canon_kg(
                    triplets_case,
                    schema_case,   # 🔥 per-case schema
                    f"{iter_dir}/canon_kg_{case}.json"
                )

                # ================= TXT (per sample) =================
                triplets_per_sample = extract_pred_triplets_per_sample(sc_results, case)
            
                schema_utils.dump_canon_kg_txt(
                    triplets_per_sample,
                    f"{iter_dir}/canon_kg_{case}.txt"
                )

            # ================= SAVE SCHEMA PER CASE =================
            for case in cases:
            
                schema_case = schema_per_case.get(case, {})
            
                with open(f"{iter_dir}/canon_schema_{case}.json", "w", encoding="utf-8") as f:
                    json.dump(schema_case, f, indent=2, ensure_ascii=False)    
    
            # ================= ENTITY CANONICALIZATION (optional) =================
            entity_type_map_per_case = {}
            if self.sc_config.get("enable_entity_canon", False):
                print("🧩 Running entity-type canonicalization...")

                _t0 = time.perf_counter()
                _tok0 = llm_utils.get_token_usage()
                ec_results, entity_schema_per_case, entity_type_map_per_case = \
                    self.entity_canonicalization(working_text_list, sd_dict_list)
                _timing["ec_sec"] = round(time.perf_counter() - _t0, 1)
                _timing["ec_tokens"] = llm_utils.token_usage_delta(_tok0)

                for ec_case, samples in ec_results.items():
                    with open(f"{iter_dir}/entity_canon_{ec_case}.json", "w", encoding="utf-8") as f:
                        json.dump(samples, f, indent=2, ensure_ascii=False)

                    with open(f"{iter_dir}/canon_entity_schema_{ec_case}.json", "w", encoding="utf-8") as f:
                        json.dump(entity_schema_per_case[ec_case], f, indent=2, ensure_ascii=False)

                # Re-dump relation KGs with canonical entity types attached.
                # Uses the first EC case by default; does NOT overwrite the
                # original canon_kg_{case}.json files.
                default_ec = list(ec_results.keys())[0]
                entity_type_map = entity_type_map_per_case[default_ec]

                for case in cases:
                    triplets_case = triplets_by_case[case]
                    schema_utils.dump_canon_kg_with_entity_types(
                        triplets_case,
                        schema_per_case[case],
                        entity_type_map,
                        f"{iter_dir}/canon_kg_with_entity_{case}.json",
                    )

            # ================= ENTITY-INSTANCE CANONICALIZATION (optional; RQ §8.5) ==========
            # Merges coreferent entity NODES so chunks that only exact-string matching would leave
            # apart share a node — the GraphRAG-recall lever. Runs AFTER EC because ic2/ic3 use EC's
            # canonical types as the corroborating signal. Writes NEW artifacts only.
            if self.sc_config.get("enable_instance_canon", False):
                _t0 = time.perf_counter()

                # Surfaces are relation-case-invariant (h/t come from `input_triplet`), so the
                # cluster map is computed ONCE per IC case over the union and applied to all
                # relation cases — not re-embedded per case.
                surfaces_freq = eic.collect_surface_freq(triplets_by_case)
                desc_map = eic.collect_entity_descriptions(sd_dict_list)

                ec_case_used = self.sc_config.get("instance_canon_ec_case")
                if entity_type_map_per_case:
                    if ec_case_used not in entity_type_map_per_case:
                        ec_case_used = list(entity_type_map_per_case.keys())[0]
                    type_map = entity_type_map_per_case[ec_case_used]
                else:
                    ec_case_used = None
                    type_map = {}
                    print("⚠️  [IC] enable_entity_canon is OFF -> no canonical entity types are "
                          "available, so ic2/ic3 have no type signal and DEGRADE TO ic1. Their "
                          "numbers are NOT a valid test of the §8.5 type-conditioning hypothesis "
                          "(DECISIONS 2026-07-16b). Enable entity canon for a real ic2/ic3 run.")

                print(f"🔗 Running entity-INSTANCE canonicalization "
                      f"({len(surfaces_freq)} unique surfaces, types from ec_case={ec_case_used})...")

                sc_embedder = self.load_model(self.sc_embedder_name, "sts")
                ic = eic.EntityInstanceCanonicalizer(sc_embedder)

                for ic_case in eic.IC_ABLATION_CONFIG:
                    canon_map, ic_stats = ic.canonicalize(
                        surfaces_freq, type_map, desc_map, ic_case
                    )
                    ic_stats["ec_case_used"] = ec_case_used

                    with open(f"{iter_dir}/instance_canon_{ic_case}.json", "w",
                              encoding="utf-8") as f:
                        json.dump(canon_map, f, indent=2, ensure_ascii=False)
                    eic.save_stats(ic_stats, f"{iter_dir}/instance_canon_stats_{ic_case}.json")

                    # Rewrite every relation case's KG under this IC case. NEVER overwrites
                    # canon_kg_{case}.txt — the headline-scored artifact keeps raw surfaces (§8.2).
                    for case in cases:
                        triplets_per_sample = extract_pred_triplets_per_sample(sc_results, case)
                        rewritten = eic.rewrite_triplets_per_sample(triplets_per_sample, canon_map)
                        schema_utils.dump_canon_kg_txt(
                            rewritten,
                            f"{iter_dir}/canon_kg_instance_{ic_case}_{case}.txt",
                        )

                _timing["ic_sec"] = round(time.perf_counter() - _t0, 1)

            # ================= STAGE TIMING =================
            # Wall-clock per stage + derived per-item/per-1k rates → extrapolate big runs.
            try:
                ni = max(_timing.get("n_items") or 0, 1)
                def _per1k(v):
                    return round(v / ni * 1000, 1) if isinstance(v, (int, float)) else v
                _timing["per_item_sec"] = {
                    k: (round(_timing[k] / ni, 3) if isinstance(_timing.get(k), (int, float)) else _timing.get(k))
                    for k in ("oie_sec", "sd_sec", "sc_sec", "ec_sec", "ic_sec")
                }
                _timing["proj_per_1k_sec"] = {
                    k: _per1k(_timing.get(k)) for k in ("oie_sec", "sd_sec", "sc_sec", "ec_sec", "ic_sec")
                }
                _stage_secs = [_timing.get(k) for k in ("oie_sec", "sd_sec", "sc_sec", "ec_sec", "ic_sec")
                               if isinstance(_timing.get(k), (int, float))]
                _timing["pipeline_sec_measured"] = round(sum(_stage_secs), 1)
                # Token totals across measured stages + per-item projection.
                _tok_keys = ("oie_tokens", "sd_tokens", "sc_tokens", "ec_tokens")
                _tok_total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                for k in _tok_keys:
                    d = _timing.get(k)
                    if isinstance(d, dict):
                        for f in _tok_total:
                            _tok_total[f] += d.get(f, 0)
                _timing["tokens_total"] = _tok_total
                _timing["tokens_per_item"] = {f: round(_tok_total[f] / ni, 1) for f in _tok_total}
                _timing["runtime"] = self._collect_runtime_meta()  # server/model/pipeline + RAM/GPU
                with open(os.path.join(iter_dir, "stage_timing.json"), "w", encoding="utf-8") as f:
                    json.dump(_timing, f, indent=2, ensure_ascii=False)
                print(f"[TIMING] iter{iteration}: {_timing.get('oie_sec')}s OIE | "
                      f"{_timing.get('sd_sec')}s SD | {_timing.get('sc_sec')}s SC "
                      f"({_timing.get('n_sc_cases')} cases) | {_timing.get('ec_sec','-')}s EC "
                      f"| {ni} items, {_timing.get('n_triplets')} triplets")
                print(f"[TOKENS] iter{iteration}: {_tok_total['total_tokens']} total "
                      f"({_tok_total['prompt_tokens']} prompt + {_tok_total['completion_tokens']} completion) "
                      f"over {_tok_total['calls']} LLM calls")
            except Exception as _e:
                logger.warning(f"stage_timing dump failed: {_e}")

            # ================= EVAL =================
            # Evaluation is now a SEPARATE step (the KG is built without a gold
            # KG in hand). Run after the pipeline:
            #   - KGC strict/exact/partial : soda/evaluate/run_full_evaluation.py
            #   - Mode-1 (name-drift robust): soda/evaluate/mode1_metric.py
            #   - OIE stage                 : soda/evaluate/oie_metric.py
            # The old inline evaluate_*/plot_* stack was dead (runa passes
            # gold=None) and overlapped soda/evaluate/ — removed.

            # ================= NEXT ITER =================
            # Feed the RAW OIE triplets of this iteration into the next iteration's
            # refined OIE (was hardcoded to the weak case2_name_general [v1; v2 case4] canon output).
            # The entity hint — the recall driver — only needs the entities, which the
            # OIE triplets carry directly and case-independently; relation hints come
            # from the schema retriever over the (now-populated) induced schema.
            triplets_from_last_iteration = oie_triplets_list

        return sc_results

#======================================================================
#====== ================
#======================================================================
def _clean_coref_output(raw):
    """Extract the rewritten text from a coref LLM response (approach A).

    The LLM is asked to output ONLY the rewritten text; this strips the few
    common wrappers it still adds (code fences, a leading 'Rewritten text:' /
    'Output:' / Vietnamese 'Văn bản ...:' label) defensively.
    """
    if not raw:
        return ""
    t = raw.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    t = re.sub(
        r"^(rewritten text|resolved text|output|kết quả|văn bản[^\n:]*)\s*:\s*",
        "", t, flags=re.IGNORECASE,
    ).strip()
    return t


_COREF_PRONOUNS = {
    "he", "she", "it", "they", "him", "her", "them", "his", "hers", "its",
    "their", "theirs", "this", "that", "these", "those", "who", "which",
    "we", "us", "our", "ours", "you", "your", "yours", "i", "me", "my", "mine",
}
_COREF_ARTICLE_PREFIX = (
    "the ", "a ", "an ", "this ", "that ", "these ", "those ",
    "its ", "his ", "her ", "their ", "our ",
)


def _coref_representative(mentions):
    """Pick the canonical mention of a cluster: most proper-noun-like, not a
    pronoun/definite-description (so 'Rome' beats 'the city', 'its')."""
    def score(m):
        ml = m.strip()
        low = ml.lower()
        s = 0.0
        if low in _COREF_PRONOUNS:
            s -= 100
        if low.startswith(_COREF_ARTICLE_PREFIX):
            s -= 10
        s += sum(1 for w in ml.split() if w[:1].isupper())   # proper-noun signal
        s += 0.01 * len(ml)                                  # tie-break: longer
        return s
    return max(mentions, key=score)


def _maverick_rewrite(text, clusters_char_offsets, clusters_text_mentions):
    """Rewrite text using Maverick clusters (approach A). char offsets are INCLUSIVE
    on both ends. Replace every non-representative mention with its cluster's
    representative; apply right-to-left so original offsets stay valid; skip
    overlapping spans."""
    repls = []  # (start, end_inclusive, replacement)
    for spans, mentions in zip(clusters_char_offsets, clusters_text_mentions):
        if not spans or not mentions:
            continue
        rep = _coref_representative(mentions).strip()
        for (s, e), m in zip(spans, mentions):
            if m.strip() == rep:
                continue
            repls.append((s, e, rep))

    if not repls:
        return text

    repls.sort(key=lambda x: x[0], reverse=True)
    applied = []
    out = text
    for s, e, rep in repls:
        if any(not (e < a_s or s > a_e) for a_s, a_e in applied):
            continue  # overlaps an already-applied span
        out = out[:s] + rep + out[e + 1:]
        applied.append((s, e))
    return out


def load_total_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
        

def save_sc_results(results, output_dir):
    import os, json
    os.makedirs(output_dir, exist_ok=True)

    for case, samples in results.items():
        path = f"{output_dir}/{case}.json"

        simplified = []

        for s in samples:
            item = {
                "text": s["text"],
                "triplets": []
            }

            for t in s["triplets"]:
                item["triplets"].append({
                    "input": t["input_triplet"],
                    "pred": t["pred_relation"],
                    "top_k": t["top_k"]
                })

            simplified.append(item)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(simplified, f, indent=2, ensure_ascii=False)

    
def extract_pred_triplets(sc_results, case):

    if case not in sc_results:
        return []

    results = []

    for sample in sc_results[case]:

        sample_triplets = []

        for t in sample.get("triplets", []):
            pred = t.get("pred_relation")

            if pred is None:
                continue

            h, _, tail = t["input_triplet"]

            sample_triplets.append((h, pred, tail))

        results.append(sample_triplets)

    return results


def extract_pred_triplets_per_sample(sc_results, case):
    """
    Output:
    [
        [ (h, r, t), (h, r, t) ],   # sample 0
        [ (h, r, t) ],              # sample 1
        ...
    ]
    """

    if case not in sc_results:
        return []

    results = []

    for sample in sc_results[case]:
        sample_triplets = []

        for t in sample.get("triplets", []):
            pred = t.get("pred_relation")

            if pred is None:
                continue

            h, _, tail = t["input_triplet"]
            sample_triplets.append([h, pred, tail])  # list format

        results.append(sample_triplets)

    return results

