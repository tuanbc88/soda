from argparse import ArgumentParser
from soda.edc_framework import EDC
import os
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================================================
# DEFAULTS
# =========================================================

DEFAULT_DATASET_NAME = "webnlg"
DEFAULT_PROMPT_LANG = "eng"

# =========================================================
# HELPERS
# =========================================================

def p(base_dir, filename):
    return os.path.join(base_dir, filename)


def build_paths(dataset_name, prompt_lang):

    prompt_template_dir = f"./soda/prompt_templates/{prompt_lang}"

    few_shot_dir = f"./soda/few_shot_examples/{dataset_name}"

    dataset_path = f"./datasets/{dataset_name}.txt"

    schema_path = f"./schemas/{dataset_name}_schema.json"

    output_dir = f"./output/{dataset_name}"

    return {
        "prompt_template_dir": prompt_template_dir,
        "few_shot_dir": few_shot_dir,
        "dataset_path": dataset_path,
        "schema_path": schema_path,
        "output_dir": output_dir,
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = ArgumentParser()

    # =====================================================
    # DATASET / LANGUAGE
    # =====================================================

    parser.add_argument(
        "--dataset_name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset profile name",
    )

    parser.add_argument(
        "--prompt_lang",
        default=DEFAULT_PROMPT_LANG,
        help="Prompt language profile: eng | vni",
    )

    # =====================================================
    # MODELS
    # =====================================================

    parser.add_argument(
        "--oie_llm",
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )

    parser.add_argument(
        "--sd_llm",
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )

    parser.add_argument(
        "--sc_llm",
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )

    parser.add_argument(
        "--ee_llm",
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )

    parser.add_argument(
        "--sc_embedder",
        default="intfloat/e5-mistral-7b-instruct",
    )

    parser.add_argument(
        "--sr_embedder",
        default="intfloat/e5-mistral-7b-instruct",
    )

    parser.add_argument(
        "--eval_embedder",
        default="intfloat/e5-mistral-7b-instruct",
    )

    parser.add_argument(
        "--sr_adapter_path",
        default=None,
    )

    # =====================================================
    # PIPELINE
    # =====================================================

    # END-TO-END refine: re-run the whole pipeline N extra times, each re-extracting OIE
    # with hints built from the previous iteration (EDC+R style).
    parser.add_argument(
        "--refinement_iterations",
        default=0,
        type=int,
    )

    # PER-ITEM IMMEDIATE refine: within the first OIE pass, re-extract each item right
    # after its pass-1 with a local EE-merged entity hint. Independent of (and combinable
    # with) --refinement_iterations. OFF by default.
    parser.add_argument(
        "--refine_per_item",
        action="store_true",
    )

    parser.add_argument(
        "--enrich_schema",
        action="store_true",
    )

    parser.add_argument(
        "--freeze_iter",
        type=int,
        default=None,
    )

    # Variance seed. OIE/SD sample (do_sample=True) → stochastic; SC is greedy
    # (deterministic given OIE+SD). Setting this seeds torch/numpy/random ONCE at start
    # so the OIE+SD draws are reproducible. A "seed" = one full BASE run; the ≥3-seed
    # variance study = 3 BASE runs with different --seed (then mean±std over the matrices).
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    # SD decoding temperature. 0.0 = greedy (default, deterministic, reproduces v1).
    # >0 SAMPLES the SD definition/typing stage → the controlled variance source for the
    # ≥3-seed study (OIE + SC stay greedy). Pair with --seed for reproducibility.
    parser.add_argument(
        "--sd_temperature",
        type=float,
        default=0.0,
    )

    # Resume a run by reusing cached stage outputs (oie_total.json / sd_total.json)
    # in the iter dir, instead of recomputing:
    #   none -> full run (default)
    #   sd   -> skip OIE  (load oie_total.json)
    #   sc   -> skip OIE+SD (load oie_total.json + sd_total.json)
    parser.add_argument(
        "--resume_from",
        choices=["none", "sd", "sc"],
        default="none",
    )

    # Run ONLY the entity-canonicalization pass on a run whose SC already finished on disk
    # (reads sd_total.json + case*.json from iter0, writes entity_canon_*/canon_entity_schema_*/
    # canon_kg_with_entity_*). Needed because --resume_from sc re-builds the SchemaCanonicalizer,
    # whose validator rejects an already-grown discovered schema containing an LLM-minted relation
    # with an empty definition. See EDC.entity_canonicalization_only (2026-07-25).
    parser.add_argument("--ec_only", action="store_true")

    # Reuse cached OIE/SD totals from ANOTHER run's output dir (its iter dir).
    # OIE+SD are independent of mode/retrieval/case, so compute once then point
    # later runs here with --resume_from sc to only re-run SC. The totals are
    # copied into this run's iter dir so it stays self-contained.
    parser.add_argument(
        "--reuse_stage_dir",
        default=None,
    )

    # =====================================================
    # RUN MODE / RETRIEVAL / ENTITY-CANON  (NEW)
    # =====================================================

    # Mode 1: empty schema (overrides --target_schema_path -> None)
    parser.add_argument(
        "--no_schema",
        action="store_true",
    )

    # Retrieval: "item" (original) or "item+cluster" (cluster-augmented)
    parser.add_argument(
        "--retrieval_mode",
        default="item",
        choices=["item", "item+cluster"],
    )
    parser.add_argument("--cluster_sim_threshold", type=float, default=0.85)
    parser.add_argument("--cluster_top_m", type=int, default=2)
    parser.add_argument("--cluster_extra_k", type=int, default=3)

    # case0 (non-LLM embedding+threshold baseline): top-1 cosine >= threshold -> reuse
    parser.add_argument("--case0_sim_threshold", type=float, default=0.85)

    # SC candidate-retrieval depth (#candidates shown to the MCQ + logged in the
    # SC trace). Default None -> keep the built-in 5. Bump it (e.g. 10/15) for ONE
    # run, then retrieval_recall_metric.py gives recall@1..@k offline (k-sweep).
    parser.add_argument("--sc_top_k", dest="top_k", type=int, default=None)

    # SC speed-up (item 13): batch the per-CASE LLM MCQ calls of one triplet into a
    # single generate(). Default 1 = OFF (sequential per-case path, byte-identical).
    # >1 = batch up to N LLM-cases/call (e.g. 8 → all LLM-cases of a triplet at once).
    # HF SC only; must reproduce greedy numbers — validate batched-on vs -off on server.
    parser.add_argument("--sc_batch_size", type=int, default=1)

    # Entity-type canonicalization pass (off by default)
    parser.add_argument(
        "--enable_entity_canon",
        action="store_true",
    )

    # Entity-INSTANCE canonicalization pass (RQ §8.5; DECISIONS 2026-07-16b). OFF by default.
    # Merges coreferent entity NODES (the GraphRAG-recall lever) — unlike --enable_entity_canon,
    # which only assigns entity TYPES and leaves nodes as raw surfaces. Rewrites surface forms in
    # NEW artifacts (canon_kg_instance_{ic_case}_{case}.txt) → GraphRAG-scope datasets only.
    parser.add_argument(
        "--enable_instance_canon",
        action="store_true",
    )
    # Which EC case supplies the canonical types for ic2/ic3 (default None → the first EC case).
    # Requires --enable_entity_canon; without it ic2/ic3 have no type signal and degrade to ic1.
    parser.add_argument(
        "--instance_canon_ec_case",
        default=None,
    )

    # Coreference REWRITE stage BEFORE OIE (LLM, approach A). OFF by default.
    # Rewrites entity surface forms → use ONLY for GraphRAG-scope datasets; it
    # breaks span-based strict/exact/partial on gold-triple sets. See RQ §8.4.
    parser.add_argument(
        "--enable_coref",
        action="store_true",
    )
    # LLM for the coref stage; default None → reuse --oie_llm (cached, no extra VRAM).
    parser.add_argument(
        "--coref_llm",
        default=None,
    )
    # Coref backend: "llm" (Qwen3 rewrite, EN+VI) | "maverick" (ACL'24 dedicated coref,
    # ENGLISH-only, needs `pip install maverick-coref`).
    parser.add_argument(
        "--coref_method",
        choices=["llm", "maverick"],
        default="llm",
    )

    # =====================================================
    # OPTIONAL OVERRIDE
    # =====================================================

    parser.add_argument(
        "--input_text_file_path",
        default=None,
    )

    parser.add_argument(
        "--target_schema_path",
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        default=None,
    )

    # =====================================================
    # LOGGING
    # =====================================================

    parser.add_argument(
        "--logging_verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    parser.add_argument(
        "--logging_debug",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
    )

    # =====================================================
    # FIRST PARSE
    # =====================================================

    args = parser.parse_args()

    dataset_name = args.dataset_name
    prompt_lang = args.prompt_lang

    paths = build_paths(dataset_name, prompt_lang)

    PROMPT_TEMPLATE_DIR = paths["prompt_template_dir"]
    FEW_SHOT_DIR = paths["few_shot_dir"]

    # =====================================================
    # AUTO DEFAULTS
    # =====================================================

    parser.set_defaults(

        # =================================================
        # OIE
        # =================================================

        oie_prompt_template_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "oie_template.txt"
        ),

        oie_few_shot_example_file_path=p(
            FEW_SHOT_DIR,
            "oie_few_shot_examples.txt"
        ),

        # =================================================
        # COREF (optional, pre-OIE rewrite)
        # =================================================

        coref_prompt_template_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "coref_template.txt"
        ),

        coref_few_shot_example_file_path=p(
            FEW_SHOT_DIR,
            "coref_few_shot_examples.txt"
        ),

        # =================================================
        # SD1 ENTITY
        # =================================================

        sd1_entity_prompt_template_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "sd1_entity_def_template.txt"
        ),

        sd1_entity_few_shot_example_file_path=p(
            FEW_SHOT_DIR,
            "sd1_entity_def_fewshot.txt"
        ),

        # =================================================
        # SD2A
        # =================================================

        sd2a_generalrelation_prompt_template_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "sd2_rel_general_abstract_template.txt"
        ),

        sd2a_generalrelation_few_shot_example_file_path=p(
            FEW_SHOT_DIR,
            "sd2_rel_general_abstract_fewshot.txt"
        ),

        # v2: EDC-style (Type A) general-def pass (→ general_definition_edc), case3.
        # Its OWN template (Type-A subject-object style) + few-shot. Optional: if either
        # file is absent the SD2a-EDC pass is skipped (case3 then falls back to the legacy
        # general_definition). The abstract SD2a pass above is unchanged (→ case4).
        sd2a_edc_generalrelation_prompt_template_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "sd2_rel_general_edc_template.txt"
        ),

        sd2a_edc_generalrelation_few_shot_example_file_path=p(
            FEW_SHOT_DIR,
            "sd2_rel_general_edc_fewshot.txt"
        ),

        # =================================================
        # SD2B
        # =================================================

        sd2_relation_prompt_template_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "sd2_rel_detail_template.txt"
        ),

        sd2_relation_few_shot_example_file_path=p(
            FEW_SHOT_DIR,
            "sd2_rel_detail_fewshot.txt"
        ),

        # =================================================
        # SD3
        # =================================================

        sd3_clasy_prompt_template_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "sd3_rel_classify_template.txt"
        ),

        sd3_clasy_few_shot_example_file_path=p(
            FEW_SHOT_DIR,
            "sd3_rel_classify_fewshot.txt"
        ),

        # =================================================
        # SC
        # =================================================

        sc_prompt_template_option1_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "sc_template_option1_name_only.txt"
        ),

        sc_prompt_template_option2_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "sc_template_option2_general_definition.txt"
        ),

        sc_prompt_template_option3_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "sc_template_option3_definition.txt"
        ),

        sc_prompt_template_option4_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "sc_template_option4_definition_headtail.txt"
        ),

        sc_prompt_self_verify_template_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "sc_schema_self_verify.txt"
        ),

        # =================================================
        # EC (ENTITY CANONICALIZATION)
        # =================================================

        ec_prompt_template_option1_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "ec_template_option1_name_only.txt"
        ),

        ec_prompt_template_option2_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "ec_template_option2_definition.txt"
        ),

        ec_prompt_template_option3_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "ec_template_option3_definition_parent.txt"
        ),

        # =================================================
        # REFINEMENT
        # =================================================

        oie_refine_prompt_template_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "oie_r_template.txt"
        ),

        oie_refine_few_shot_example_file_path=p(
            FEW_SHOT_DIR,
            "oie_few_shot_refine_examples.txt"
        ),

        ee_prompt_template_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "ee_template.txt"
        ),

        ee_few_shot_example_file_path=p(
            FEW_SHOT_DIR,
            "ee_few_shot_examples.txt"
        ),

        em_prompt_template_file_path=p(
            PROMPT_TEMPLATE_DIR,
            "em_template.txt"
        ),

        # =================================================
        # INPUT / OUTPUT
        # =================================================

        input_text_file_path=paths["dataset_path"],

        target_schema_path=paths["schema_path"],

        output_dir=paths["output_dir"],
    )

    # =====================================================
    # FINAL PARSE
    # =====================================================

    args = parser.parse_args()
    args = vars(args)

    # Variance seed (not an EDC kwarg) — pop + seed torch/numpy/random ONCE here so the
    # stochastic OIE/SD stages are reproducible for this BASE run.
    _ec_only = args.pop("ec_only", False)
    _seed = args.pop("seed", None)
    if _seed is not None:
        import transformers
        transformers.set_seed(int(_seed))

    # Mode 1: empty schema -> ignore the default schema path
    if args.get("no_schema"):
        args["target_schema_path"] = None

    # =====================================================
    # PRINT CONFIG
    # =====================================================

    print("===================================================")
    print(f"DATASET_NAME : {dataset_name}")
    print(f"PROMPT_LANG  : {prompt_lang}")
    print(f"PROMPT_DIR   : {PROMPT_TEMPLATE_DIR}")
    print(f"FEWSHOT_DIR  : {FEW_SHOT_DIR}")
    print(f"INPUT_FILE   : {args['input_text_file_path']}")
    print(f"SCHEMA_FILE  : {args['target_schema_path']}")
    print(f"OUTPUT_DIR   : {args['output_dir']}")
    print(f"SEED         : {_seed if _seed is not None else '<none>'}")
    print(f"SD_TEMP      : {args.get('sd_temperature', 0.0)}  (0=greedy; >0 samples SD → variance source; OIE/SC always greedy)")
    print(f"ENABLE_COREF : {args.get('enable_coref')} (method={args.get('coref_method')})")
    print(f"INSTANCE_CANON: {args.get('enable_instance_canon')} "
          f"(ec_case={args.get('instance_canon_ec_case') or '<first>'})")
    print("===================================================")

    # Maverick is English-only (trained on OntoNotes) — warn if used on a vni run.
    if args.get("enable_coref") and args.get("coref_method") == "maverick" and prompt_lang == "vni":
        print("⚠️  coref_method='maverick' is ENGLISH-only; use --coref_method llm for Vietnamese (vni).")

    # Guard: coref rewrites entity surface forms → span-based strict/exact/partial
    # become meaningless on the gold-triple sets. Warn loudly (do NOT abort).
    _gold_triple_sets = (dataset_name in ("rebel", "wiki-nre")
                         or dataset_name.startswith("webnlg"))
    if args.get("enable_coref") and _gold_triple_sets:
        print("⚠️ " * 12)
        print(f"⚠️  --enable_coref is ON for a GOLD-TRIPLE dataset ({dataset_name}).")
        print("⚠️  Coref rewrites entity surface forms → strict/exact/partial will be")
        print("⚠️  WRONG (entity spans won't match gold). Coref is for GraphRAG-scope")
        print("⚠️  datasets only (eduhcmut/hotpot/trivia). See RESEARCH_QUESTIONS.md §8.4.")
        print("⚠️ " * 12)

    # Guard: instance canon merges entity NODES → same span problem as coref, and it would change
    # the entity basis vs EDC (§8.2). The headline canon_kg_{case}.txt is NOT touched (IC writes
    # canon_kg_instance_*), so this is a warning about what you may compare, not a broken run.
    if args.get("enable_instance_canon") and _gold_triple_sets:
        print("⚠️ " * 12)
        print(f"⚠️  --enable_instance_canon is ON for a GOLD-TRIPLE dataset ({dataset_name}).")
        print("⚠️  Instance canon merges entity NODES → the canon_kg_instance_* KGs are NOT")
        print("⚠️  comparable to gold (entity spans won't match) and change the entity basis")
        print("⚠️  vs EDC. canon_kg_{case}.txt (the headline artifact) is untouched, so the")
        print("⚠️  headline numbers stay valid. §8.5 is GraphRAG-scope only (eduhcmut/hotpot).")
        print("⚠️  See RESEARCH_QUESTIONS.md §8.5.")
        print("⚠️ " * 12)

    if args.get("enable_instance_canon") and not args.get("enable_entity_canon"):
        print("⚠️  --enable_instance_canon without --enable_entity_canon: ic2/ic3 have no")
        print("⚠️  canonical entity types and DEGRADE TO ic1 — the 3-case comparison is void.")

    # =====================================================
    # BUILD EDC
    # =====================================================

    edc = EDC(**args)

    # =====================================================
    # LOAD INPUT
    # =====================================================

    with open(
        args["input_text_file_path"],
        "r",
        encoding="utf-8"
    ) as f:
        input_text_list = f.readlines()

    # =====================================================
    # RUN
    # =====================================================

    # "none" -> None (full run); "sd"/"sc" -> resume from cached stage outputs
    resume_from = None if args["resume_from"] == "none" else args["resume_from"]

    if _ec_only:
        # Finish ONLY the EC pass of a run whose SC is already on disk. Cannot be done with
        # --resume_from sc: that path constructs the SchemaCanonicalizer, whose validator
        # rejects a RESTORED (already-grown) discovered schema when the LLM minted a relation
        # with an empty definition. See EDC.entity_canonicalization_only.
        output_kg = edc.entity_canonicalization_only(
            input_text_list=input_text_list,
            output_dir=args["output_dir"],
        )
    else:
        output_kg = edc.extract_kg(
            input_text_list=input_text_list,
            gold_triplets_list=None,
            output_dir=args["output_dir"],
            refinement_iterations=args["refinement_iterations"],
            freeze_iter=args["freeze_iter"],
            resume_from=resume_from,
            reuse_stage_dir=args["reuse_stage_dir"],
        )
