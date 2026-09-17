import json
import os
import openai
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import ast
from sentence_transformers import SentenceTransformer
from typing import List
import gc
import torch
import logging
import re

logger = logging.getLogger(__name__)


# =====================================================================
# Token-usage accounting (process-global; default-on, zero behavior change)
# ---------------------------------------------------------------------
# Every transformers generation call records its prompt/completion token
# counts here. The framework snapshots this around each stage (OIE/SD/SC/EC)
# to attribute tokens per stage in stage_timing.json. The OpenAI path can
# add to this later via _record_token_usage() from its response usage.
# =====================================================================
_TOKEN_USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def _record_token_usage(prompt_tokens: int, completion_tokens: int):
    """Add one LLM call's token counts to the process-global accumulator."""
    _TOKEN_USAGE["calls"] += 1
    _TOKEN_USAGE["prompt_tokens"] += int(prompt_tokens)
    _TOKEN_USAGE["completion_tokens"] += int(completion_tokens)


def get_token_usage():
    """Snapshot of cumulative token usage (with derived total)."""
    u = dict(_TOKEN_USAGE)
    u["total_tokens"] = u["prompt_tokens"] + u["completion_tokens"]
    return u


def reset_token_usage():
    """Reset the cumulative counters (e.g. at the start of a run)."""
    _TOKEN_USAGE.update(calls=0, prompt_tokens=0, completion_tokens=0)


def token_usage_delta(before: dict):
    """Per-stage usage = current snapshot minus a `before` snapshot."""
    now = get_token_usage()
    return {
        "calls": now["calls"] - before.get("calls", 0),
        "prompt_tokens": now["prompt_tokens"] - before.get("prompt_tokens", 0),
        "completion_tokens": now["completion_tokens"] - before.get("completion_tokens", 0),
        "total_tokens": now["total_tokens"] - before.get("total_tokens", 0),
    }


def free_model(model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None):
    # A vLLM engine is shared across all roles for the whole run — never free/recreate it.
    if getattr(model, "__soda_vllm__", False):
        return
    try:
        model.cpu()
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(e)


def get_embedding_e5mistral(model, tokenizer, sentence, task=None):
    model.eval()
    device = model.device

    if task != None:
        # It's a query to be embed
        sentence = get_detailed_instruct(task, sentence)

    sentence = [sentence]

    max_length = 4096
    # Tokenize the input texts
    batch_dict = tokenizer(
        sentence, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True
    )
    # append eos_token_id to every input_ids
    batch_dict["input_ids"] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")

    batch_dict.to(device)

    embeddings = model(**batch_dict).detach().cpu()

    assert len(embeddings) == 1

    return embeddings[0]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def get_embedding_sts(model: SentenceTransformer, text: str, prompt_name=None, prompt=None):
    embedding = model.encode(text, prompt_name=prompt_name, prompt=prompt)
    return embedding


# def parse_raw_entities(raw_entities: str):
#     parsed_entities = []
#     left_bracket_idx = raw_entities.index("[")
#     right_bracket_idx = raw_entities.index("]")
#     try:
#         parsed_entities = ast.literal_eval(raw_entities[left_bracket_idx : right_bracket_idx + 1])
#     except Exception as e:
#         pass
#     logging.debug(f"Entities {raw_entities} parsed as {parsed_entities}")
#     return parsed_entities

def parse_raw_entities(raw_entities: str):
    if not raw_entities:
        return []

    raw = raw_entities.strip()

    # ---------- Case 1: Python list ----------
    if "[" in raw and "]" in raw:
        try:
            left = raw.find("[")
            right = raw.find("]", left)
            parsed = ast.literal_eval(raw[left:right + 1])
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except Exception:
            pass

    # ---------- Case 2: comma-separated text ----------
    # e.g. "A, B, C"
    if "," in raw:
        parts = [p.strip(" :\"'") for p in raw.split(",")]
        parts = [p for p in parts if len(p) > 0]
        if parts:
            logger.warning(
                "[parse_raw_entities] Fallback comma-split used"
            )
            return parts

    # ---------- Case 3: give up ----------
    logger.warning(
        f"[parse_raw_entities] Cannot parse entities, return empty. Raw: {raw_entities}"
    )
    return []


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


def parse_relation_definition(raw_definitions: str):
    descriptions = raw_definitions.split("\n")
    relation_definition_dict = {}

    for description in descriptions:
        if ":" not in description:
            continue
        index_of_colon = description.index(":")
        relation = description[:index_of_colon].strip()

        relation_description = description[index_of_colon + 1 :].strip()

        if relation == "Answer":
            continue

        relation_definition_dict[relation] = relation_description
    logger.debug(f"Relation Definitions {raw_definitions} parsed as {relation_definition_dict}")
    return relation_definition_dict


def is_model_openai(model_name):
    # Route every LLM role through the OpenAI-compatible HTTP path when an external LLM
    # server is configured (item-13 step-2 "Plan B": a SEPARATE-env vLLM `vllm serve`, so
    # SODA's Python-3.9 env never imports vllm). Otherwise only real GPT models.
    if os.environ.get("EDC_LLM_BASE_URL"):
        return True
    return "gpt" in model_name


# def generate_completion_transformers(
#     input: list,
#     model: AutoModelForCausalLM,
#     tokenizer: AutoTokenizer,
#     max_new_token=800,  # TuanBC change 256 to 800
#     temperature=0,   # TuanBC add more config
#     answer_prepend="",
# ):
#     device = model.device
#     tokenizer.pad_token = tokenizer.eos_token

#     messages = tokenizer.apply_chat_template(input, add_generation_prompt=True, tokenize=False) + answer_prepend

#     model_inputs = tokenizer(messages, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

#     generation_config = GenerationConfig(
#         do_sample=False,
#         max_new_tokens=max_new_token,
#         pad_token_id=tokenizer.eos_token_id,
#         return_dict_in_generate=True,
#     )

#     generation = model.generate(**model_inputs, generation_config=generation_config)
#     sequences = generation["sequences"]
#     generated_ids = sequences[:, model_inputs["input_ids"].shape[1] :]
#     generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

#     logging.debug(f"Prompt:\n {messages}\n Result: {generated_texts}")
#     return generated_texts

from transformers import GenerationConfig
import torch
import logging


def _clean_generated(generated_text: str) -> str:
    """Post-process a raw decoded completion. Shared by the single-prompt and the
    batched generators so both clean IDENTICALLY (factored out, no behavior change):
      1. strip a Qwen3-style <think>...</think> reasoning trace (safety net even with
         enable_thinking=False) — keep only text after the last </think>;
      2. truncate any trailing junk after the first closing ``]]`` (important for OIE).
    """
    # --- strip thinking trace ---
    if "</think>" in generated_text:
        generated_text = generated_text.split("</think>")[-1]
    generated_text = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.DOTALL)
    generated_text = generated_text.replace("<think>", "").replace("</think>", "").strip()

    # --- post-clean: remove trailing junk after the first ]] ---
    if "]]" in generated_text:
        generated_text = generated_text.split("]]")[0] + "]]"

    return generated_text


def generate_completion_transformers(
    input: list,
    model,
    tokenizer,
    max_new_tokens=None,        # 🔥 dynamic nếu None
    temperature=0.0,
    answer_prepend="",
    max_context_tokens=4096,    # 🔥 limit context (tuỳ model)
    dynamic_ratio=0.25,         # 🔥 % output so với input
    max_new_tokens_cap=512,     # 🔥 trần output cho nhánh dynamic; truyền vào để nâng
):
    """
    Improved generation:
    - dynamic max_new_tokens
    - truncate long input
    - avoid hanging on long samples
    """

    # vLLM dispatch: when load_model returned a vLLM engine (EDC_USE_VLLM=1) the "model" is a
    # VLLMEngine, route there. Default (HF model) path below is unchanged. See VLLM_PLAN.md.
    if getattr(model, "__soda_vllm__", False):
        from soda.utils import vllm_backend
        return vllm_backend.generate(
            model, tokenizer, input, max_new_tokens=max_new_tokens, temperature=temperature,
            answer_prepend=answer_prepend, max_context_tokens=max_context_tokens,
            dynamic_ratio=dynamic_ratio, max_new_tokens_cap=max_new_tokens_cap)

    device = model.device
    tokenizer.pad_token = tokenizer.eos_token

    # =========================================
    # BUILD PROMPT
    # =========================================
    # Disable "thinking mode" when the chat template supports it (Qwen3, etc.)
    # so structured/list output is not wrapped in <think>...</think> or routed
    # into a reasoning channel. Templates that don't accept the kwarg (Qwen2.5,
    # Mistral, Llama) fall back to the plain call.
    try:
        messages = tokenizer.apply_chat_template(
            input,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        ) + answer_prepend
    except TypeError:
        messages = tokenizer.apply_chat_template(
            input,
            add_generation_prompt=True,
            tokenize=False
        ) + answer_prepend

    # =========================================
    # TOKENIZE (NO TRUNCATE YET)
    # =========================================
    inputs = tokenizer(messages, return_tensors="pt", add_special_tokens=False)

    input_len = inputs["input_ids"].shape[1]

    # =========================================
    # 🔥 TRUNCATE IF TOO LONG
    # =========================================
    if input_len > max_context_tokens:
        logging.warning(f"[GEN] Input too long ({input_len}) → truncating")

        # keep tail (important for instruction-following)
        inputs = tokenizer(
            messages,
            return_tensors="pt",
            truncation=True,
            max_length=max_context_tokens,
            add_special_tokens=False
        )
        input_len = inputs["input_ids"].shape[1]

    inputs = inputs.to(device)

    # =========================================
    # 🔥 DYNAMIC max_new_tokens
    # =========================================
    if max_new_tokens is None:
        # heuristic: output ≈ 20–30% input length
        dynamic_tokens = int(input_len * dynamic_ratio)

        # clamp range (trần mặc định 512; nâng qua max_new_tokens_cap khi cần)
        max_new_tokens = max(64, min(dynamic_tokens, max_new_tokens_cap))

    logging.debug(f"[GEN] input_len={input_len}, max_new_tokens={max_new_tokens}")

    # =========================================
    # GENERATION CONFIG
    # =========================================
    generation_config = GenerationConfig(
        do_sample=(temperature > 0),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    # =========================================
    # GENERATE
    # =========================================
    with torch.no_grad():
        generation = model.generate(
            **inputs,
            generation_config=generation_config
        )

    sequences = generation["sequences"]

    generated_ids = sequences[:, input_len:]

    # Token accounting (prompt = real fed length after any truncation;
    # completion = newly generated tokens). Defensive: never break generation.
    try:
        _record_token_usage(int(input_len), int(generated_ids.shape[1]))
    except Exception:
        pass

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    ).strip()

    # Strip thinking trace + OIE ]] post-clean (shared with the batched path).
    generated_text = _clean_generated(generated_text)

    logging.debug(f"[GEN OUTPUT]: {generated_text}")

    return generated_text


def generate_completion_transformers_batch(
    inputs: list,                 # list of `messages` (each: [{"role","content"}, ...])
    model,
    tokenizer,
    max_new_tokens=None,
    temperature=0.0,
    answer_prepend="",
    max_context_tokens=4096,
    dynamic_ratio=0.25,
    max_new_tokens_cap=512,
):
    """Batched twin of ``generate_completion_transformers`` (item 13 speed-up).

    Left-pads all prompts into one batch, runs a single ``model.generate``, then
    truncates each row to ITS OWN per-prompt ``max_new_tokens`` budget (computed with
    the SAME clamp formula as the single-prompt path) before decoding. Under GREEDY
    decoding the decoded string for each prompt therefore equals what the single-prompt
    path would produce, **modulo floating-point batch-matmul nondeterminism** (batched
    reductions can, rarely, flip an argmax at a near-tie). That residual is why batched
    output MUST be validated against the single path on the real model/server before any
    number is trusted; the local CPU unit test only proves the padding/truncation logic.

    OFF by default in the pipeline (only used when ``--sc_batch_size > 1``). Returns a
    ``list[str]`` aligned with ``inputs``; ``[]`` in -> ``[]`` out.
    """
    if not inputs:
        return []

    # vLLM dispatch (EDC_USE_VLLM=1): continuous-batched generation. HF path below unchanged.
    if getattr(model, "__soda_vllm__", False):
        from soda.utils import vllm_backend
        return vllm_backend.generate_batch(
            model, tokenizer, inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            answer_prepend=answer_prepend, max_context_tokens=max_context_tokens,
            dynamic_ratio=dynamic_ratio, max_new_tokens_cap=max_new_tokens_cap)

    device = model.device
    tokenizer.pad_token = tokenizer.eos_token
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"   # decoder-only batched generation needs left-pad
    try:
        # ---- chat-templated strings (mirror single path: thinking off if supported) ----
        texts = []
        for messages in inputs:
            try:
                t = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                    enable_thinking=False,
                ) + answer_prepend
            except TypeError:
                t = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                ) + answer_prepend
            texts.append(t)

        # ---- true per-prompt input length (no padding) → per-prompt budget ----
        # Mirrors the single path: budget = clamp(input_len * ratio) unless an explicit
        # max_new_tokens is given. Long inputs are truncated to max_context_tokens first.
        per_lens, per_budget = [], []
        for t in texts:
            L = tokenizer(t, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
            if L > max_context_tokens:
                L = max_context_tokens
            if max_new_tokens is not None:
                b = max_new_tokens
            else:
                b = max(64, min(int(L * dynamic_ratio), max_new_tokens_cap))
            per_lens.append(L)
            per_budget.append(b)
        batch_max = max(per_budget)

        # ---- batch tokenize (left-pad; right-truncate tail, matching single path) ----
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_context_tokens, add_special_tokens=False,
        ).to(device)

        generation_config = GenerationConfig(
            do_sample=(temperature > 0),
            temperature=temperature,
            max_new_tokens=batch_max,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

        with torch.no_grad():
            generation = model.generate(**enc, generation_config=generation_config)

        sequences = generation["sequences"]
        prompt_width = enc["input_ids"].shape[1]
        gen_only = sequences[:, prompt_width:]   # (B, batch_max)

        outputs = []
        for i in range(len(texts)):
            # Truncate to THIS prompt's own budget so the decoded text matches the
            # single-prompt path (which generated exactly per_budget[i] tokens). After
            # an early <eos> the row is pad-filled; skip_special_tokens drops it.
            row = gen_only[i, : per_budget[i]]
            try:
                # NB: completion count here is an upper bound — an <eos> early-stop is
                # not subtracted (token totals feed the appendix only, not results).
                _record_token_usage(int(per_lens[i]), int(row.shape[0]))
            except Exception:
                pass
            text_out = tokenizer.decode(row, skip_special_tokens=True).strip()
            outputs.append(_clean_generated(text_out))
        return outputs
    finally:
        tokenizer.padding_side = prev_side


_OPENAI_CLIENT = None


def _get_openai_client():
    """Cached OpenAI SDK client. When EDC_LLM_BASE_URL is set it points at a local
    OpenAI-compatible server (e.g. `vllm serve` in its own env) — item-13 Plan B."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    from openai import OpenAI
    base_url = os.environ.get("EDC_LLM_BASE_URL")  # e.g. http://localhost:8000/v1
    api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    _OPENAI_CLIENT = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def openai_chat_completion(
    model, system_prompt, history, temperature=0, max_tokens=512,
    max_retries=8, base_delay=2.0, max_delay=120.0,
):
    import random as _random
    client = _get_openai_client()
    server_mode = bool(os.environ.get("EDC_LLM_BASE_URL"))

    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = history

    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    # On a vLLM server, disable the Qwen3 "thinking" trace (match the HF path's
    # enable_thinking=False) unless explicitly re-enabled — otherwise reasoning eats the
    # max_tokens budget before the answer. Set EDC_LLM_ENABLE_THINKING=1 for non-Qwen models.
    if server_mode and os.environ.get("EDC_LLM_ENABLE_THINKING", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    last_exc = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            try:
                u = response.usage
                if u is not None:
                    _record_token_usage(u.prompt_tokens, u.completion_tokens)
            except Exception:
                pass
            content = response.choices[0].message.content
            # Match the HF path's post-clean (think-strip + ]] truncate) in server mode.
            if server_mode and content:
                content = _clean_generated(content)
            logging.debug(f"Model: {model}\nPrompt:\n {messages}\n Result: {content}")
            return content
        except Exception as e:
            last_exc = e
            # extra_body chat_template_kwargs can error on a model whose template lacks
            # enable_thinking → drop it and retry without (don't burn all retries on it).
            if "extra_body" in kwargs:
                logger.warning(f"[openai] dropping enable_thinking extra_body after error: {e}")
                kwargs.pop("extra_body", None)
                continue
            delay = min(base_delay * (2 ** attempt) + _random.uniform(0, 1), max_delay)
            logger.warning(f"[openai] attempt {attempt+1}/{max_retries} failed: {e}. Retry in {delay:.1f}s")
            time.sleep(delay)

    raise RuntimeError(f"openai_chat_completion failed after {max_retries} retries") from last_exc

# TuanBC 2026.03.10 22h00
def parse_schema_json(text):

    import json
    import re

    try:
        return json.loads(text)
    except Exception:

        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            return json.loads(match.group())

    return {"relations": {}, "entities": {}}

# TuanBC 2026.03.11 16h00
def parse_llm_json(text):

    # remove code block
    text = re.sub(r"```json|```", "", text).strip()

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1:
        text = text[start:end+1]

    return json.loads(text)



def safe_literal_eval_list(text):
    try:
        return ast.literal_eval(text)
    except:
        pass

    # fix dấu ' nằm giữa word (O'Brien, Martyrs'_Memorial)
    text = re.sub(r"(?<=\w)'(?=\w)", "\\'", text)

    try:
        return ast.literal_eval(text)
    except:
        return []

# --- escape single quote inside tokens (robust)
# def _escape_inner_quotes(s: str) -> str:
#     """
#     Escape ' inside tokens but keep list syntax intact.
#     Example:
#         Baku_Turkish_Martyrs'_Memorial
#     ->  Baku_Turkish_Martyrs\'_Memorial
#     """

#     result = []
#     in_string = False

#     for i, ch in enumerate(s):
#         if ch == "'":
#             # check if this is a boundary quote or inner quote
#             prev_char = s[i - 1] if i > 0 else ""
#             next_char = s[i + 1] if i < len(s) - 1 else ""

#             # boundary if near comma/bracket
#             if prev_char in ["[", ",", " "] or next_char in [",", "]", " "]:
#                 result.append(ch)
#                 in_string = not in_string
#             else:
#                 # inner quote → escape
#                 result.append("\\'")
#         else:
#             result.append(ch)

#     return "".join(result)
    