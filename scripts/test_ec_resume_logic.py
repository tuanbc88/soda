"""Standalone logic test for EC mid-stage resume (RESUME_UPGRADE_PLAN Phase 3).

Twin of test_sc_resume_logic.py. The local env lacks the heavy deps
(sentence_transformers/torch), so we cannot import edc_framework; this mirrors the EXACT
checkpoint dict + restore/guard logic added to entity_canonicalization() and asserts the
serialization contract holds — in particular that ALL THREE pieces of EC state survive a
round-trip (results, evolved entity schema, and the accumulated surface->type map), and
that a run interrupted mid-way then resumed produces the same output as an uninterrupted
one. The byte-identical end-to-end gate runs on the server.

Run:  python scripts/test_ec_resume_logic.py
"""
import os, json, tempfile

# ---- copy of _checkpoint_json / _ec_checkpoint (atomic write; nothing stripped) ----
def _checkpoint_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)

def _ec_checkpoint(path, done, n_items, cases, results, schema_per_case,
                   entity_type_map_per_case):
    _checkpoint_json({"done": done, "n_items": n_items, "cases": list(cases),
                      "results": results, "schema_per_case": schema_per_case,
                      "entity_type_map_per_case": entity_type_map_per_case}, path)

# ---- copy of the restore/guard block ----
def restore(ckpt_path, cases, n_items):
    schema_per_case = {c: {"entity_types": {}} for c in cases}
    results = {c: [] for c in cases}
    type_map = {c: {} for c in cases}
    start_idx = 0
    if os.path.exists(ckpt_path):
        prev = json.load(open(ckpt_path, encoding="utf-8"))
        if (isinstance(prev, dict) and prev.get("n_items") == n_items
                and list(prev.get("cases", [])) == list(cases)
                and 0 <= prev.get("done", 0) <= n_items):
            results = {c: list(prev["results"][c]) for c in cases}
            for c in cases:
                schema_per_case[c] = prev["schema_per_case"][c]
                type_map[c] = dict(prev.get("entity_type_map_per_case", {}).get(c, {}))
            start_idx = prev["done"]
    return results, schema_per_case, type_map, start_idx

# ---- a deterministic fake EC pass (schema-stateful, like the real one) ----
CASES = ["ec1_name_only", "ec2_name_definition", "ec3_definition_parent"]

def fake_ec_loop(items, cases, results, schema_per_case, type_map, start_idx,
                 ckpt_path=None, stop_after=None, ckpt_every=50):
    """Mimics the real loop: per item, per case, grow the schema + record the map."""
    for i in range(start_idx, len(items)):
        text, fine, member = items[i]
        for c in cases:
            ents = schema_per_case[c]["entity_types"]
            # deterministic "decision": reuse if the fine type is already known, else mint
            if fine in ents:
                final, is_new = fine, False
            else:
                ents[fine] = {"definition": f"def-{fine}", "parent": "Thing"}
                final, is_new = fine, True
            type_map[c][member] = final
            results[c].append({"text": text, "entities": [
                {"input_fine_type": fine, "pred_entity_type": final, "members": [member],
                 "top_k": list(ents)[:5], "is_new": is_new, "decision": "llm_reuse"}]})
        if ckpt_path and ((i + 1) % ckpt_every == 0 or (i + 1) == len(items)):
            _ec_checkpoint(ckpt_path, i + 1, len(items), cases, results,
                           schema_per_case, type_map)
        if stop_after is not None and (i + 1) >= stop_after:
            return results, schema_per_case, type_map, False   # "crashed"
    return results, schema_per_case, type_map, True

def fresh(cases):
    return ({c: [] for c in cases}, {c: {"entity_types": {}} for c in cases},
            {c: {} for c in cases})

def main():
    N = 120
    items = [(f"text {i}", f"Fine{i % 17}", f"ent_{i}") for i in range(N)]
    ok = True

    with tempfile.TemporaryDirectory() as td:
        # --- A) uninterrupted reference run (no checkpointing) ---
        r0, s0, m0 = fresh(CASES)
        ref_results, ref_schema, ref_map, done = fake_ec_loop(items, CASES, r0, s0, m0, 0)
        assert done

        # --- B) interrupted at item 73, then resumed from the checkpoint ---
        ck = os.path.join(td, "ec_resume.json")
        r1, s1, m1 = fresh(CASES)
        fake_ec_loop(items, CASES, r1, s1, m1, 0, ckpt_path=ck, stop_after=73, ckpt_every=50)
        # the checkpoint lands on the 50-modulo, so `done` should be 50 (not 73)
        prev = json.load(open(ck, encoding="utf-8"))
        print(f"[ckpt] done={prev['done']} (expected 50: the modulo-50 cadence)")
        ok &= prev["done"] == 50

        r2, s2, m2, start = restore(ck, CASES, N)
        print(f"[restore] start_idx={start}, results len={len(r2[CASES[0]])}")
        ok &= start == 50 and len(r2[CASES[0]]) == 50
        res_results, res_schema, res_map, done2 = fake_ec_loop(
            items, CASES, r2, s2, m2, start, ckpt_path=ck, ckpt_every=50)
        assert done2

        # --- C) resumed == uninterrupted, on ALL THREE pieces of state ---
        same_results = res_results == ref_results
        same_schema = res_schema == ref_schema
        same_map = res_map == ref_map
        print(f"[equal] results={same_results} schema={same_schema} type_map={same_map}")
        ok &= same_results and same_schema and same_map
        ok &= all(len(res_results[c]) == N for c in CASES)

        # --- D) guards: n_items / cases mismatch -> start fresh, never crash ---
        _, _, _, s_bad_n = restore(ck, CASES, N + 1)
        _, _, _, s_bad_cases = restore(ck, CASES[:2], N)
        print(f"[guard] wrong n_items -> start={s_bad_n}; wrong cases -> start={s_bad_cases}")
        ok &= s_bad_n == 0 and s_bad_cases == 0

        # --- E) type_map is genuinely restored (not recomputable from results alone) ---
        print(f"[type_map] entries after resume={len(res_map[CASES[0]])} (expected {N})")
        ok &= len(res_map[CASES[0]]) == N

    print("\nEC RESUME LOGIC:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

if __name__ == "__main__":
    main()
