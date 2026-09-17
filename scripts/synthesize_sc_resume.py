"""Synthesize sc_resume.json for a run whose SC is fully done on disk (case*.json +
canon_schema_case*.json) but which never wrote sc_resume.json (ran without
EDC_RESUME_INPLACE=1). With this file present, `RESUME_FROM=sc EDC_RESUME_INPLACE=1`
loads done==n_items, so the SC loop is empty (skipped) and the pipeline proceeds
straight to EC. Lets a completed-SC run finish its EC pass without re-running SC.

Matches the writer format in edc_framework.py `_sc_checkpoint` (slim results:
text + triplets[{input_triplet, pred_relation, top_k}]) and the loader guard
(n_items, cases, done, results{case:list}, schema_per_case{case:...}).

USAGE:
  python scripts/synthesize_sc_resume.py <run_dir>/iter0
  # then: RESUME_FROM=sc EDC_RESUME_INPLACE=1 ... bash run_soda.sh
"""
import os, sys, json, glob

def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: synthesize_sc_resume.py <iter_dir>")
    it = sys.argv[1]
    # case order MUST match SchemaCanonicalizer.SC_ABLATION_CONFIG.keys() -> read the case files,
    # sorted by their leading caseN index so order is deterministic & matches the pipeline.
    case_files = sorted(glob.glob(os.path.join(it, "case[0-9]*.json")),
                        key=lambda p: int(os.path.basename(p).split("_")[0].replace("case", "")))
    if not case_files:
        raise SystemExit(f"no case*.json in {it}")
    cases, results, schema_per_case = [], {}, {}
    n_items = None
    for cf in case_files:
        case = os.path.basename(cf)[:-5]   # strip .json  -> e.g. case3_name_gendef_edc
        cases.append(case)
        data = json.load(open(cf, encoding="utf-8"))
        if n_items is None:
            n_items = len(data)
        elif len(data) != n_items:
            raise SystemExit(f"item-count mismatch: {case} has {len(data)} vs {n_items}")
        slim = []
        for s in data:
            trips = []
            for t in s.get("triplets", []):
                trips.append({
                    "input_triplet": t.get("input_triplet", t.get("input")),
                    "pred_relation": t.get("pred_relation", t.get("pred")),
                    "top_k": t.get("top_k", []),
                })
            slim.append({"text": s.get("text", ""), "triplets": trips})
        results[case] = slim
        # schema for this case
        scf = os.path.join(it, f"canon_schema_{case}.json")
        schema_per_case[case] = json.load(open(scf, encoding="utf-8")) if os.path.exists(scf) else {}

    out = {"done": n_items, "n_items": n_items, "cases": cases,
           "results": results, "schema_per_case": schema_per_case}
    dst = os.path.join(it, "sc_resume.json")
    tmp = dst + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(tmp, dst)
    print(f"[ok] wrote {dst}: done={n_items}/{n_items}, {len(cases)} cases: {cases}")
    print("     schema present for:", [c for c in cases if schema_per_case[c]])

if __name__ == "__main__":
    main()
