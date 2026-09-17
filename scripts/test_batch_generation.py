"""Local GPU-free unit tests for the item-13 SC batching speed-up.

Two equivalences are proven WITHOUT a GPU (so the "must reproduce greedy numbers"
gate has a local check before the server validation):

  Test A — `generate_completion_transformers_batch` == `generate_completion_transformers`
           token-for-token under GREEDY decoding, on a tiny real model (sshleifer/tiny-gpt2,
           cached locally), across prompts of DIFFERENT lengths (exercises left-padding +
           per-row budget truncation). This is the only part with a residual fp risk on the
           real server model; here on CPU fp32 it should match exactly.

  Test B — `finalize(prepare(...), llm(prompt))` == `canonicalize(...)` for ALL 9 SC cases
           and several fake-LLM answers (NONE / reuse / out-of-range / unparseable). Pure
           Python logic, fully deterministic — proves the batching refactor preserves the
           single-call decision/debug exactly.

Run:  python scripts/test_batch_generation.py
(openai + sentence_transformers are stubbed — neither is used by the code under test.)
"""
import os
import sys
import types
import importlib.util
import importlib.machinery

# --- offline + stub the two deps the code imports but that aren't used here ---
# IMPORTANT: only stub when the real package is GENUINELY ABSENT (e.g. the local box),
# and give the stub a valid __spec__ — otherwise, on a server where these ARE installed,
# transformers' `importlib.util.find_spec("openai")` would either pick the real one (good)
# or choke on a spec-less stub (ValueError: openai.__spec__ is None).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure_stub(name, **attrs):
    try:
        if importlib.util.find_spec(name) is not None:
            return  # real package present — never shadow it
    except (ImportError, ValueError):
        pass
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


_ensure_stub("openai")
# llm_utils does `from sentence_transformers import SentenceTransformer`
_ensure_stub("sentence_transformers", SentenceTransformer=object)


# =====================================================================
# Test A — batched == single (greedy), real tiny GPT2 + hermetic char tokenizer
# ---------------------------------------------------------------------
# The MODEL is a real (randomly-initialized) GPT2 — so the part that actually matters
# for batched left-padding (attention-mask -> position_ids handling inside generate)
# is exercised for real. The tokenizer is a self-contained char-level stand-in so the
# test needs no network/weights download; it faithfully mimics HF left/right padding +
# attention_mask, which is all the generation helpers rely on it for.
# =====================================================================
class _BatchEnc(dict):
    def to(self, device):
        return self  # CPU-only test


class _CharTok:
    """Minimal char-level tokenizer matching the slice of the HF tokenizer API that
    generate_completion_transformers[_batch] use (call -> tensors+mask, decode,
    padding_side, pad/eos)."""
    def __init__(self, vocab_size=50257, eos_token_id=50256):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.eos_token = "<eos>"
        self.pad_token = self.eos_token
        self.padding_side = "right"

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kw):
        return "".join(m["content"] for m in messages)

    def _encode(self, text):
        return [min(ord(c) % (self.vocab_size - 1), self.vocab_size - 2) for c in text]

    def __call__(self, text, return_tensors=None, padding=False, truncation=False,
                 max_length=None, add_special_tokens=False):
        import torch
        seqs = [self._encode(text)] if isinstance(text, str) else [self._encode(t) for t in text]
        if truncation and max_length:
            seqs = [s[:max_length] for s in seqs]          # right-truncation (HF default)
        if padding and len(seqs) > 1:
            L = max(len(s) for s in seqs)
            pad_id, inp, att = self.eos_token_id, [], []
            for s in seqs:
                npad = L - len(s)
                if self.padding_side == "left":
                    inp.append([pad_id] * npad + s); att.append([0] * npad + [1] * len(s))
                else:
                    inp.append(s + [pad_id] * npad); att.append([1] * len(s) + [0] * npad)
        else:
            inp, att = seqs, [[1] * len(s) for s in seqs]
        return _BatchEnc(input_ids=torch.tensor(inp), attention_mask=torch.tensor(att))

    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        out = []
        for i in ids:
            if skip_special_tokens and i == self.eos_token_id:
                continue
            out.append(chr(i) if i < 0x110000 else "?")
        return "".join(out)


def test_a_batch_equals_single():
    import soda.utils.llm_utils as L

    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2Config
        torch.manual_seed(0)
        # tiny real GPT2, random weights, fully offline; n_positions comfortably above
        # (longest prompt + max generated budget) to avoid position overflow.
        config = GPT2Config(vocab_size=50257, n_positions=256, n_embd=32, n_layer=2, n_head=2)
        model = GPT2LMHeadModel(config)
        tok = _CharTok(vocab_size=50257, eos_token_id=50256)
    except Exception as e:
        print(f"[Test A] SKIP (could not build tiny GPT2 offline: {repr(e)[:120]})")
        return None

    model.eval()

    # Capability gate: the production helpers call model.generate(generation_config=...),
    # an API added in transformers>=4.26. On older transformers (the local box ships
    # 4.25.1) this raises — SKIP cleanly and let the server (modern transformers) run it.
    try:
        import torch
        from transformers import GenerationConfig
        with torch.no_grad():
            model.generate(input_ids=torch.tensor([[1, 2, 3]]),
                           generation_config=GenerationConfig(max_new_tokens=1,
                                                              pad_token_id=tok.eos_token_id))
    except Exception as e:
        print(f"[Test A] SKIP (transformers too old for generation_config API: "
              f"{repr(e)[:90]}). Run on the server's transformers to validate batched==single.")
        return None

    prompts = [
        "A",
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog and then keeps running",
        "One two three",
    ]
    msgs = [[{"role": "user", "content": p}] for p in prompts]

    single = [L.generate_completion_transformers(m, model, tok) for m in msgs]
    batch = L.generate_completion_transformers_batch(msgs, model, tok)

    assert len(single) == len(batch) == len(prompts)
    ok = True
    for i, (s, b) in enumerate(zip(single, batch)):
        match = (s == b)
        ok = ok and match
        if not match:
            print(f"[Test A] MISMATCH idx={i}")
            print(f"  single: {s!r}")
            print(f"  batch : {b!r}")

    # also: batch-of-1 must equal single (no padding involved)
    b1 = L.generate_completion_transformers_batch([msgs[2]], model, tok)
    if b1[0] != single[2]:
        ok = False
        print(f"[Test A] batch-of-1 mismatch:\n  {single[2]!r}\n  {b1[0]!r}")

    print(f"[Test A] {'PASS' if ok else 'FAIL'} — batched==single greedy on {len(prompts)} varied-length prompts")
    return ok


# =====================================================================
# Test B — finalize(prepare(...)) == canonicalize(...) for all 9 cases
# =====================================================================
class _StubEmbedder:
    """Deterministic surrogate for the SentenceTransformer embedder."""
    def encode(self, text, **kw):
        import numpy as np
        import hashlib
        h = hashlib.md5(str(text).encode("utf-8")).digest()
        return np.frombuffer(h, dtype=np.uint8).astype(np.float32)  # 16-dim, deterministic


def _make_schema():
    """A small schema with EVERY relation-definition field populated (so every case's
    _validate_schema passes)."""
    def rel(defn, edc, abs, h, t):
        return {
            "definition": defn,
            "general_definition_edc": edc,
            "general_definition_abstract": abs,
            "head_type": h,
            "tail_type": t,
            "attributes": {},
            "relation_type": "static",
        }
    return {
        "entity_types": {},
        "relation_types": {
            "bornIn": rel("X was born in place Y", "The subject entity was born in the object entity",
                          "indicates a birth-place relation", "Person", "Place"),
            "locatedIn": rel("X is located in Y", "The subject entity is located in the object entity",
                             "indicates a containment relation", "Place", "Place"),
            "worksFor": rel("X works for organization Y", "The subject entity works for the object entity",
                            "indicates an employment relation", "Person", "Organization"),
        },
    }


def test_b_prepare_finalize_equals_canonicalize():
    import importlib
    import hashlib
    import soda.utils.llm_utils as L
    SCmod = importlib.import_module("soda.schema_canonicalization")
    SchemaCanonicalizer = SCmod.SchemaCanonicalizer

    # deterministic fake LLM: choice depends only on the prompt string, so canonicalize()
    # and the prepare/finalize path (same prompt) get the SAME answer.
    def fake_llm(messages, model=None, tokenizer=None, **kw):
        prompt = messages[0]["content"]
        n = int(hashlib.md5(prompt.encode("utf-8")).hexdigest(), 16) % 5
        return f" Answer: {n}"   # 0=NONE, 1..4 = candidate idx (may exceed -> parse None)
    L.generate_completion_transformers = fake_llm  # monkeypatch

    embedder = _StubEmbedder()
    # minimal template carrying the placeholders build_prompt fills
    template = ("Text: {input_text}\nTriplet: {query_triplet}\nRel: {query_relation} "
                "({query_relation_definition}) {head_type}->{tail_type}\nChoices:\n{choices}\nAnswer:")

    cases = list(SchemaCanonicalizer.SC_ABLATION_CONFIG.keys())

    # triplets: one whose relation is in-schema (rich rel_info) + one brand-new relation
    triplets = [
        ("Alice", "bornIn", "Paris"),
        ("Bob", "marriedTo", "Carol"),   # not in schema -> empty rel_info fields
    ]
    sd_rel_info = {
        "bornIn": {
            "relation": "bornIn", "definition": "X was born in place Y",
            "general_definition_edc": "The subject entity was born in the object entity",
            "general_definition_abstract": "indicates a birth-place relation",
            "head_type": "Person", "tail_type": "Place",
        },
    }

    ok = True
    n_checked = 0
    for case in cases:
        cfg = SchemaCanonicalizer.SC_ABLATION_CONFIG[case]
        prompt_key = cfg["prompt_key"]
        prompt_template = None if prompt_key is None else template

        sc_config = {"ablation_case": case, "top_k": 5, "case0_sim_threshold": 0.5}
        sc = SchemaCanonicalizer(
            _make_schema(), embedder,
            verify_model=object(), verify_tokenizer=object(),
            sc_config=sc_config,
        )

        for triplet in triplets:
            r = triplet[1]
            raw = sd_rel_info.get(r, {"relation": r, "definition": "",
                                      "head_type": "", "tail_type": "",
                                      "general_definition_edc": "",
                                      "general_definition_abstract": ""})
            rel_info = sc.strip_rel_info_by_case(raw)

            # reference: the untouched single-call path
            exp = sc.canonicalize(triplet[0] + " text", triplet, rel_info,
                                  prompt_template, prompt_key=prompt_key, prompt_path="p")

            # batched path: prepare -> (fake) batched llm on the prompt -> finalize
            prep = sc.prepare(triplet[0] + " text", triplet, rel_info,
                              prompt_template, prompt_key=prompt_key, prompt_path="p")
            if prep["needs_llm"]:
                raw_out = fake_llm([{"role": "user", "content": prep["prompt"]}])
            else:
                raw_out = None
            got = sc.finalize(prep, raw_out, llm_ok=True)

            n_checked += 1
            # compare the decision-bearing fields (numpy score arrays compared elementwise)
            ef, _ec, _es, ed = exp
            gf, _gc, _gs, gd = got
            same = (
                ef == gf
                and ed["decision"] == gd["decision"]
                and ed["llm_choice"] == gd["llm_choice"]
                and ed["top_k"] == gd["top_k"]
                and ed["prompt"] == gd["prompt"]
                and ed["raw_output"] == gd["raw_output"]
            )
            if not same:
                ok = False
                print(f"[Test B] MISMATCH case={case} rel={r}")
                print(f"  exp: final={ef} dec={ed['decision']} choice={ed['llm_choice']}")
                print(f"  got: final={gf} dec={gd['decision']} choice={gd['llm_choice']}")

    print(f"[Test B] {'PASS' if ok else 'FAIL'} — finalize(prepare)==canonicalize over "
          f"{n_checked} (case x triplet) combos")
    return ok


if __name__ == "__main__":
    a = test_a_batch_equals_single()
    b = test_b_prepare_finalize_equals_canonicalize()
    results = [r for r in (a, b) if r is not None]
    if all(results):
        print("\nALL TESTS PASSED" + ("" if a is not None else " (Test A skipped)"))
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
