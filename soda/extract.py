from typing import List
import os
from pathlib import Path
import re
import json
import logging
import ast
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

import soda.utils.llm_utils as llm_utils
import soda.utils.log_utils as log_utils
import soda.utils.debug_logger as debug_logger
from soda.pipeline_logger import PipelineLogger
#from soda.relation_refinement import RelationRefiner

logger = logging.getLogger(__name__)


class Extractor:
    # The class to handle the first stage: Open Information Extraction
    def __init__(self, model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None, openai_model=None, pipeline_logger: PipelineLogger = None,iter_dir=None) -> None:
        assert openai_model is not None or (model is not None and tokenizer is not None)
        self.model = model
        self.tokenizer = tokenizer
        self.openai_model = openai_model
        self.pipeline_logger = pipeline_logger
        self.iter_dir = iter_dir

    def extract(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        prompt_template_str: str,
        entities_hint: str = None,
        relations_hint: str = None,
        index: int = None,
    ) -> List[List[str]]:
        assert (entities_hint is None and relations_hint is None) or (
            entities_hint is not None and relations_hint is not None
        )
        
        input_text_str = input_text_str.strip()
        # print('-------------OIE: input data ----------------')
        # print(f" + input_text_str: {input_text_str}")
        # print(f" + entities_hint: {entities_hint}")
        # print(f" + relations_hint: {relations_hint}")
        # print('-------------------------------------------------')
        filled_prompt = prompt_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
                "input_text": input_text_str,
                "entities_hint": entities_hint,
                "relations_hint": relations_hint,
            }
        )
        #print('-------------OIE: filled_prompt ----------------')
        #print(filled_prompt)
        #print('-------------------------------------------------')
        messages = [{"role": "user", "content": filled_prompt}]

        if self.openai_model is None:
            completion = llm_utils.generate_completion_transformers(
                messages, self.model, self.tokenizer, answer_prepend="Triplets: "
            )
        else:
            completion = llm_utils.openai_chat_completion(self.openai_model, None, messages)
        
        # ======================================================
        # DUMP RAW LLM OUTPUT (JSONL, one line per item; gated by EDC_DEBUG_LOGS)
        debug_logger.jsonl_log(self.iter_dir, "oie_raw_llm_output", {
            "idx": index,
            "input_text": input_text_str,
            "entities_hint": entities_hint,
            "relations_hint": relations_hint,
            "prompt": filled_prompt,
            "raw_output": completion if completion else "[EMPTY OUTPUT]",
        })
        # =============================================================

        extracted_triplets_list  = parse_triplets_from_text(completion)

        if not extracted_triplets_list:
            print("[WARN] parse failed → retry again - 1")
            # retry 1 lần với budget output thực sự lớn hơn: parse fail của item dài
            # thường do output bị cắt ở trần dynamic (≤512), nên retry phải vượt 512
            # mới có tác dụng (512 cũ = no-op vì dynamic đã chạm 512).
            if self.openai_model is None:
                completion = llm_utils.generate_completion_transformers(
                    messages,
                    self.model,
                    self.tokenizer,
                    max_new_tokens=1024
                )
            else:
                completion = llm_utils.openai_chat_completion(
                    self.openai_model, None, messages, max_tokens=1024
                )
            debug_logger.jsonl_log(self.iter_dir, "oie_raw_llm_output", {
                "idx": index, "retry": True,
                "input_text": input_text_str, "prompt": filled_prompt,
                "raw_output": completion if completion else "[EMPTY OUTPUT]",
            })

            extracted_triplets_list = parse_triplets_from_text(completion)

        debug_logger.jsonl_log(self.iter_dir, "oie_logs", {
            "idx": index, "type": "OIE output", "data": extracted_triplets_list,
        })

        
        if self.pipeline_logger:
            self.pipeline_logger.log(
                item_id=index,
                stage="oie",
                data={
                    "raw_output": completion,
                    "parsed": extracted_triplets_list,
                    "entities_hint": entities_hint,
                    "relations_hint": relations_hint
                }
            )
        
        return extracted_triplets_list


def parse_triplets_from_text(raw_text: str) -> List[Tuple[str, str, str]]:
    """
    Parse LLM output dạng:
    [['A','r','B'], ['C','r','D']]
    """

    if not raw_text:
        return []

    # ---- step 1: extract phần chứa list ----
    match = re.search(r"\[\s*\[.*\]\s*\]", raw_text, re.DOTALL)
    if not match:
        return []

    list_str = match.group(0)

    # list_str = llm_utils._escape_inner_quotes(list_str)

    # ---- step 2: safe eval ----
    parsed = llm_utils.safe_literal_eval_list(list_str)
    if not parsed:
        logger.warning(f"[OIE PARSE FAIL] raw_list_str={list_str}")
        return []

    # ---- step 3: validate ----
    triplets = []
    for item in parsed:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 3
            and all(isinstance(x, str) for x in item)
        ):
            h, r, t = item

            # clean nhẹ
            h, r, t = h.strip(), r.strip(), t.strip()

            if h and r and t:
                triplets.append((h, r, t))

    return triplets

    
    
def parse_raw_triplets(raw_triplets: str):
    # Look for enclosing brackets
    unmatched_left_bracket_indices = []
    matched_bracket_pairs = []

    collected_triples = []
    for c_idx, c in enumerate(raw_triplets):
        if c == "[":
            unmatched_left_bracket_indices.append(c_idx)
        if c == "]":
            if len(unmatched_left_bracket_indices) == 0:
                continue
            # Found a right bracket, match to the last found left bracket
            matched_left_bracket_idx = unmatched_left_bracket_indices.pop()
            matched_bracket_pairs.append((matched_left_bracket_idx, c_idx))
    for l, r in matched_bracket_pairs:
        bracketed_str = raw_triplets[l : r + 1]
        try:
            parsed_triple = ast.literal_eval(bracketed_str)
            if len(parsed_triple) == 3 and all([isinstance(t, str) for t in parsed_triple]):
                if all([e != "" and e != "_" for e in parsed_triple]):
                    collected_triples.append(parsed_triple)
            elif not all([type(x) == type(parsed_triple[0]) for x in parsed_triple]):
                for e_idx, e in enumerate(parsed_triple):
                    if isinstance(e, list):
                        parsed_triple[e_idx] = ", ".join(e)
                collected_triples.append(parsed_triple)
        except Exception as e:
            pass
    logger.debug(f"Triplets {raw_triplets} parsed as {collected_triples}")
    return collected_triples


def parse_oie_output_v2(llm_output: str):
    """
    Robust parser for OIE LLM output.

    Handles cases where the LLM returns:
    - explanations before JSON
    - markdown code blocks
    - extra text
    """

    if not llm_output or not isinstance(llm_output, str):
        raise ValueError("LLM output is empty or not a string")

    text = llm_output.strip()

    # --------------------------------------------------
    # Remove markdown code blocks if present
    # --------------------------------------------------
    text = text.replace("```json", "").replace("```", "")

    # --------------------------------------------------
    # Extract JSON object
    # --------------------------------------------------
    match = re.search(r"\{[\s\S]*\}", text)

    if not match:
        raise ValueError(f"No JSON object found in OIE output:\n{text}")

    json_str = match.group(0)

    # --------------------------------------------------
    # Parse JSON
    # --------------------------------------------------
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"OIE output contains invalid JSON:\n{json_str}"
        ) from e

    # --------------------------------------------------
    # Validate schema
    # --------------------------------------------------
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not a dictionary")

    data.setdefault("entity_relations", [])
    data.setdefault("attributes", [])

    if not isinstance(data["entity_relations"], list):
        raise ValueError("entity_relations must be a list")

    if not isinstance(data["attributes"], list):
        raise ValueError("attributes must be a list")

    return data