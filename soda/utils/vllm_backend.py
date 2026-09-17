"""vLLM generation backend (item-13 speed-up, step 2). OFF by default.

Activated only when ``EDC_USE_VLLM=1``: then ``EDC.load_model(..., "hf", role)`` returns a
``VLLMEngine`` (instead of an HF model) and ``llm_utils.generate_completion_transformers[_batch]``
dispatch here. The default (flag unset) path never imports vllm and is byte-identical.

This module lazy-imports ``vllm`` INSIDE its functions, so importing it where vLLM is not
installed (e.g. the local dev box) is safe.

Fidelity: mirrors the HF path exactly where it is free — same chat template (enable_thinking
off), same per-prompt max_new_tokens clamp, greedy via temperature=0, shared post-clean and
token metering. It is NOT byte-identical to HF (different kernels flip near-tie argmaxes), so
validate via metric-impact <= seed variance, not exact reproduce. See VLLM_PLAN.md.
"""
import os
import logging

logger = logging.getLogger(__name__)


class VLLMEngine:
    """Thin wrapper around a vLLM ``LLM`` so generation dispatch can recognise it
    (via the ``__soda_vllm__`` marker) without importing vllm just for an isinstance check."""
    __soda_vllm__ = True

    def __init__(self, engine, model_name):
        self.engine = engine
        self.model_name = model_name


_ENGINES = {}


def is_vllm_enabled():
    return os.environ.get("EDC_USE_VLLM", "0").strip().lower() in {"1", "true", "yes", "on"}


def get_engine(model_name):
    """Construct (or reuse) a vLLM engine for ``model_name``. One engine per distinct model:
    since OIE=SD=SC=EE are usually the same model, a single engine serves all roles."""
    if model_name in _ENGINES:
        return _ENGINES[model_name]

    from vllm import LLM  # lazy: only when vLLM is actually used

    quant = os.environ.get("EDC_VLLM_QUANT") or None      # bitsandbytes|awq|gptq|None(bf16)
    dtype = os.environ.get("EDC_VLLM_DTYPE", "auto")        # "bfloat16" for the A100 headline
    gpu_util = float(os.environ.get("EDC_VLLM_GPU_UTIL", "0.90"))
    max_len = int(os.environ.get("EDC_VLLM_MAX_LEN", "4096"))

    kwargs = dict(
        model=model_name,
        dtype=dtype,
        enable_prefix_caching=True,    # THE win: the shared template+few-shot prefix is cached once
        gpu_memory_utilization=gpu_util,
        max_model_len=max_len,
    )
    if quant:
        kwargs["quantization"] = quant

    logger.info(f"[vLLM] constructing engine: {model_name} "
                f"(dtype={dtype} quant={quant} gpu_util={gpu_util} max_len={max_len})")
    eng = VLLMEngine(LLM(**kwargs), model_name)
    _ENGINES[model_name] = eng
    return eng


def _build_prompt(tokenizer, messages, answer_prepend):
    """Same chat-templating as the HF path (thinking disabled where supported)."""
    try:
        s = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        s = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
    return s + answer_prepend


def _budget(tokenizer, prompt, max_new_tokens, max_context_tokens, dynamic_ratio, max_new_tokens_cap):
    """Per-prompt max_new_tokens, identical formula to generate_completion_transformers."""
    input_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if input_len > max_context_tokens:
        input_len = max_context_tokens
    if max_new_tokens is not None:
        return int(max_new_tokens)
    return max(64, min(int(input_len * dynamic_ratio), max_new_tokens_cap))


def _sampling_params(temperature, budget, max_context_tokens):
    from vllm import SamplingParams
    kw = dict(temperature=float(temperature), top_p=1.0, max_tokens=int(budget))
    # Defensive prompt truncation for the rare >max_model_len prompt (HF right-truncates;
    # vLLM keeps the tail — minor edge-case difference, documented in VLLM_PLAN.md §9).
    try:
        return SamplingParams(truncate_prompt_tokens=int(max_context_tokens), **kw)
    except TypeError:
        return SamplingParams(**kw)


def _finish(out, llm_utils):
    o0 = out.outputs[0]
    try:
        llm_utils._record_token_usage(len(out.prompt_token_ids), len(o0.token_ids))
    except Exception:
        pass
    return llm_utils._clean_generated((o0.text or "").strip())


def generate(engine_wrapper, tokenizer, messages, max_new_tokens=None, temperature=0.0,
             answer_prepend="", max_context_tokens=4096, dynamic_ratio=0.25,
             max_new_tokens_cap=512):
    """Single-prompt vLLM generation; drop-in for generate_completion_transformers."""
    import soda.utils.llm_utils as llm_utils
    prompt = _build_prompt(tokenizer, messages, answer_prepend)
    budget = _budget(tokenizer, prompt, max_new_tokens, max_context_tokens,
                     dynamic_ratio, max_new_tokens_cap)
    sp = _sampling_params(temperature, budget, max_context_tokens)
    out = engine_wrapper.engine.generate([prompt], sp, use_tqdm=False)[0]
    return _finish(out, llm_utils)


def generate_batch(engine_wrapper, tokenizer, inputs, max_new_tokens=None, temperature=0.0,
                   answer_prepend="", max_context_tokens=4096, dynamic_ratio=0.25,
                   max_new_tokens_cap=512):
    """Batched vLLM generation (continuous batching); drop-in for
    generate_completion_transformers_batch. Per-prompt SamplingParams so each keeps its own budget."""
    import soda.utils.llm_utils as llm_utils
    if not inputs:
        return []
    prompts, sps = [], []
    for messages in inputs:
        p = _build_prompt(tokenizer, messages, answer_prepend)
        b = _budget(tokenizer, p, max_new_tokens, max_context_tokens, dynamic_ratio, max_new_tokens_cap)
        prompts.append(p)
        sps.append(_sampling_params(temperature, b, max_context_tokens))
    outs = engine_wrapper.engine.generate(prompts, sps, use_tqdm=False)
    return [_finish(out, llm_utils) for out in outs]
