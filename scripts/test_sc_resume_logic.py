"""Standalone logic test for SC mid-stage resume (RESUME_UPGRADE_PLAN Phase 2).

Local env lacks the heavy deps (sentence_transformers/torch), so we cannot import
edc_framework. This mirrors the EXACT checkpoint dict + restore/guard logic added to
schema_canonicalization() and asserts the serialization contract holds. The full
byte-identical gate runs on the server (see RESUME_UPGRADE_PLAN §Validation gate)."""
import os, json, tempfile

# ---- copy of _sc_checkpoint (strip debug; atomic write) ----
def _checkpoint_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)

def _sc_checkpoint(path, done, n_items, cases, results, schema_per_case):
    slim = {}
    for case, samples in results.items():
        slim[case] = [
            {"text": s["text"],
             "triplets": [{"input_triplet": t["input_triplet"],
                           "pred_relation": t["pred_relation"],
                           "top_k": t.get("top_k", [])}
                          for t in s["triplets"]]}
            for s in samples]
    _checkpoint_json({"done": done, "n_items": n_items, "cases": list(cases),
                      "results": slim, "schema_per_case": schema_per_case}, path)

# ---- copy of the restore/guard block ----
def restore(ckpt_path, cases, n_items):
    schema_per_case = {c: {"relation_types": {}, "entity_types": {}} for c in cases}
    results = {c: [] for c in cases}
    start_idx = 0
    if os.path.exists(ckpt_path):
        prev = json.load(open(ckpt_path, encoding="utf-8"))
        if (isinstance(prev, dict) and prev.get("n_items") == n_items
                and list(prev.get("cases", [])) == list(cases)
                and 0 <= prev.get("done", 0) <= n_items):
            results = {c: list(prev["results"][c]) for c in cases}
            for c in cases:
                schema_per_case[c] = prev["schema_per_case"][c]
            start_idx = prev["done"]
            return start_idx, results, schema_per_case, "resumed"
        return start_idx, results, schema_per_case, "mismatch->fresh"
    return start_idx, results, schema_per_case, "no-ckpt"

def make_results(cases, n):
    return {c: [{"text": f"item{i}",
                 "triplets": [{"input_triplet": ["h", f"r{i}", "t"],
                               "pred_relation": f"loc{i}",
                               "top_k": ["a", "b"],
                               "debug": {"prompt": "X"*500, "raw_output": "Y"*500}}]}
                for i in range(n)] for c in cases}

def make_schema(cases, nrel):
    return {c: {"relation_types": {f"rel{k}": {"definition": f"d{k}", "aliases": []}
                                   for k in range(nrel)},
                "entity_types": {}} for c in cases}

CASES = ["case1_embed_threshold", "case8_concat"]
d = tempfile.mkdtemp()
ck = os.path.join(d, "sc_resume.json")

# 1) checkpoint at done=50 (schema has 30 rels), then restore
res = make_results(CASES, 50)
sch = make_schema(CASES, 30)
_sc_checkpoint(ck, 50, 1165, CASES, res, sch)
start, r2, s2, status = restore(ck, CASES, 1165)
assert status == "resumed", status
assert start == 50, start
assert len(r2["case8_concat"]) == 50
assert len(s2["case8_concat"]["relation_types"]) == 30
# debug stripped
assert "debug" not in r2["case1_embed_threshold"][0]["triplets"][0]
# essential keys preserved
t0 = r2["case1_embed_threshold"][0]["triplets"][0]
assert t0["input_triplet"] == ["h", "r0", "t"] and t0["pred_relation"] == "loc0" and t0["top_k"] == ["a", "b"]
print("PASS 1: round-trip restore (start_idx=50, schema=30 rels, debug stripped, keys intact)")

# 2) checkpoint size sanity: slim << full (debug dropped)
full_bytes = len(json.dumps(res))
slim_bytes = os.path.getsize(ck)
assert slim_bytes < full_bytes, (slim_bytes, full_bytes)
print(f"PASS 2: checkpoint slimmer than raw results ({slim_bytes} < {full_bytes} bytes)")

# 3) guard: n_items mismatch -> fresh
_, _, _, st = restore(ck, CASES, 999)
assert st == "mismatch->fresh", st
# 4) guard: cases mismatch -> fresh
_, _, _, st = restore(ck, ["case1_embed_threshold"], 1165)
assert st == "mismatch->fresh", st
print("PASS 3+4: guards reject n_items/cases mismatch -> start fresh")

# 5) final-item checkpoint (n_items=1165, done=1165) restores empty todo (loop range(1165,1165)=[])
_sc_checkpoint(ck, 1165, 1165, CASES, make_results(CASES, 1165), make_schema(CASES, 159))
start, r5, s5, st = restore(ck, CASES, 1165)
assert st == "resumed" and start == 1165 and len(r5["case8_concat"]) == 1165
assert list(range(start, 1165)) == []   # nothing to redo on a completed SC
print("PASS 5: completed-SC checkpoint (done==n_items) resumes to an empty loop (idempotent)")

print("\nALL SC-RESUME LOGIC TESTS PASSED")
