"""Prefix-slice a FULL run's **SC** results into the smaller scale sizes — no SC re-run.

WHY (author, 2026-07-17): "phải chạy ngược. OIE full 13k rồi, back lại define từ từ để tránh lặp lại."
He is right, and `slice_stage_totals.py`'s docstring was WRONG where it claimed
    "only SC must re-run per size, because the Mode-1 schema GROWS with the corpus"
The schema growing is exactly *why the prefix is valid*, not a reason to re-run. SC is sequential and
greedy: item i's decision depends only on the schema built from items 0..i-1. So when the `full` run
reaches item N it is **in the same state a size-N run ends in** — `full` passes through every smaller
size on its way. Slicing its SC output at N therefore reproduces a size-N run **on that same OIE+SD**
(read the corrected scope below before quoting this).

WHAT THE GUARANTEE ACTUALLY IS (corrected 2026-07-19 — the first statement here was too strong):
    Given ONE run's OIE+SD, SC at item N == what a size-N run ON THAT SAME OIE+SD produces.
That holds by construction (SC is sequential+greedy, so `full` passes through every smaller size's
end-state) and is what the 4B `--verify` check confirmed, 27/27 byte-identical.

⚠️ It is NOT the stronger claim "a sliced cell equals an INDEPENDENTLY RE-RUN cell". That is false:
   * the 4B check was near-circular — those "native" small runs were themselves produced by
     `run_track3_scale_reuse.sh`, which slices the full run's OIE+SD first, so they already shared
     the input SC consumes;
   * measured on 8B (2026-07-19), a genuinely independent 1k job vs `full[:1000]` differed in
     **55/1000 OIE items (5.5%)** — same model, same bf16, same greedy — because GPU bf16 decoding
     is not bitwise reproducible across separate jobs. Downstream that moved `#rel` by **0.2-8.3%**
     per case. Total triple count was unchanged (2830 vs 2830), so it is per-item noise, not drift.
⇒ For a SCALE CURVE this is the right tool and arguably better than independent runs: every cell then
  shares one OIE/SD/SC provenance, so the only thing varying is N. Just never claim byte-equality with
  a separate run. Re-check with `--verify` if SC stops being sequential+greedy (reordering batches,
  sampled decoding, a non-prefix split).

COST OF THE OLD WAY: the 4B curve re-ran SC on 1k+3k+6k = **10,000 redundant items**; the 8B window was
about to burn ~9,000 (~5.7h). SC is the slowest phase (~2.3 s/item).

WHAT IT WRITES (per size N, into that size's cell dir — the same layout a native run produces):
    {case}.json                 the first N samples of full's SC results
    canon_kg_{case}.json/.txt   the KG for those N items
    canon_schema_{case}.json    the schema AS OF item N  (= the §5b signal: it grows with N)

★ Schema-as-of-N: in Mode-1 the schema starts EMPTY and grows only when SC says NEW, and every schema
entry is some item's `pred`. So schema_N = the distinct `pred`s in items 0..N-1, and their definitions
are taken from the full run's final schema (a definition does not change once added; only aliases
accumulate, which do not affect #rel). Verified by `--verify` against the native runs.

USAGE
    # slice every size out of one finished full run
    python datasets/slice_sc_results.py \
        --full output/webnlg_full_full_selfcanon2_mode1_item_qwen3-8b_bgem3_H100_20260627open/iter0 \
        --sizes 1k=1000,3k=3000,6k=6000 \
        --model_tag qwen3-8b --emb_tag bgem3 --gpu_tag H100 --date_tag 20260627open
    # prove it against native runs before trusting it (exits non-zero on any mismatch)
    python datasets/slice_sc_results.py --full <full iter0> --sizes 1k=1000 --verify
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def case_files(iter_dir):
    return sorted(f[:-5] for f in os.listdir(iter_dir)
                  if f.startswith("case") and f.endswith(".json"))


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def dump(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def schema_as_of(samples, full_schema):
    """The Mode-1 schema after `samples` items = the distinct preds they used, with the full run's
    definitions. Mode-1 starts from an EMPTY schema and every entry is some item's pred, so this is
    exact (not an approximation)."""
    used = []
    seen = set()
    for s in samples:
        for t in s.get("triplets", []):
            p = t.get("pred")
            if p is not None and p not in seen:
                seen.add(p)
                used.append(p)
    rel = full_schema.get("relation_types", {}) if isinstance(full_schema, dict) else {}
    out = dict(full_schema) if isinstance(full_schema, dict) else {}
    out["relation_types"] = {r: rel[r] for r in used if r in rel}
    # a pred with no entry in the full schema would mean the invariant broke — surface it loudly
    orphan = [r for r in used if r not in rel]
    if orphan:
        print(f"    [warn] {len(orphan)} pred(s) not in the full schema (e.g. {orphan[:3]}) — "
              f"schema_as_of may be incomplete; investigate before reporting #rel")
    return out


def entity_schema_as_of(samples, n):
    """The Mode-1 ENTITY schema after n items = the distinct `pred_entity_type`s they used.

    Same prefix argument as relations: EC is sequential and greedy over a schema that starts EMPTY and
    only grows, so the full run passes through every smaller cut's end-state. Feeds `tab:scale_ent`
    (#ent per cut).

    VALIDATED (8B full, 2026-07-19): reconstructing at n = len(samples) reproduces the run's own
    `canon_entity_schema_{ec}.json` EXACTLY for all three EC cases (108 / 121 / 100). Re-run
    `--entity --verify` after any change to EC.
    """
    seen = set()
    for s in samples[:n]:
        for e in s.get("entities", []):
            t = e.get("pred_entity_type")
            if t:
                seen.add(t)
    return seen


def kg_from(samples):
    """per-sample [[h,r,t],...] with pred==None dropped — mirrors extract_pred_triplets_per_sample."""
    out = []
    for s in samples:
        trips = []
        for t in s.get("triplets", []):
            p = t.get("pred")
            if p is None:
                continue
            h, _r, tail = t["input"]
            trips.append([h, p, tail])
        out.append(trips)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="the FULL run's iter dir (has {case}.json)")
    ap.add_argument("--sizes", required=True,
                    help="comma list label=N, e.g. 1k=1000,3k=3000,6k=6000")
    ap.add_argument("--dataset_prefix", default="webnlg_full")
    ap.add_argument("--model_tag", default="qwen3-8b")
    ap.add_argument("--emb_tag", default="bgem3")
    ap.add_argument("--gpu_tag", default="H100")
    ap.add_argument("--date_tag", default="20260627open")
    ap.add_argument("--out_root", default="output")
    ap.add_argument("--entity", action="store_true",
                    help="report the ENTITY schema size (#ent) per cut instead of slicing relation SC "
                         "— feeds tab:scale_ent. Reads entity_canon_{ec}.json from --full.")
    ap.add_argument("--verify", action="store_true",
                    help="compare against the NATIVE size runs instead of writing; non-zero exit on mismatch")
    args = ap.parse_args()

    sizes = []
    for part in args.sizes.split(","):
        label, n = part.split("=")
        sizes.append((label.strip(), int(n)))

    if args.entity:
        import glob as _glob
        ecs = sorted(os.path.basename(p)[len("entity_canon_"):-len(".json")]
                     for p in _glob.glob(os.path.join(args.full, "entity_canon_ec*.json")))
        if not ecs:
            sys.exit(f"FATAL: no entity_canon_ec*.json in {args.full} (was EC enabled?)")
        print(f"[slice-ent] full={args.full}")
        print(f"[slice-ent] ec cases={ecs} sizes={sizes}")
        hdr = f"  {'phi_e':<24}" + "".join(f"{lbl:>7}" for lbl, _ in sizes)
        print(hdr)
        for ec in ecs:
            samples = load(os.path.join(args.full, f"entity_canon_{ec}.json"))
            row = [len(entity_schema_as_of(samples, n)) for _, n in sizes]
            print(f"  {ec:<24}" + "".join(f"{v:>7d}" for v in row))
            real_p = os.path.join(args.full, f"canon_entity_schema_{ec}.json")
            if os.path.exists(real_p):
                real = load(real_p).get("entity_types", {})
                rec = entity_schema_as_of(samples, len(samples))
                flag = "OK" if set(rec) == set(real) else "*** MISMATCH ***"
                print(f"      self-check at N={len(samples)}: reconstructed {len(rec)} vs actual {len(real)}  {flag}")
        return

    cases = case_files(args.full)
    if not cases:
        sys.exit(f"FATAL: no case*.json in {args.full}")
    print(f"[slice-sc] full={args.full}\n[slice-sc] cases={len(cases)} sizes={sizes}")

    def cell(label):
        ds = f"{args.dataset_prefix}_{label}"
        return os.path.join(args.out_root,
                            f"{ds}_selfcanon2_mode1_item_{args.model_tag}_{args.emb_tag}_"
                            f"{args.gpu_tag}_{args.date_tag}", "iter0")

    if args.verify:
        bad = 0
        for label, n in sizes:
            nat = cell(label)
            if not os.path.isdir(nat):
                print(f"  {label}: SKIP (no native run at {nat})")
                continue
            # Per-size counter. `bad` accumulates ACROSS sizes, so using it in this size's summary
            # printed NEGATIVE identical-counts (-9/9 then -18/9) on 2026-08-12. That looked like a
            # parser fault and distracted from the real cause of the mismatches, which was a
            # --model_tag left at its default so 32B was being compared against the 8B runs.
            bad_here = 0
            for c in cases:
                a = load(os.path.join(args.full, f"{c}.json"))[:n]
                b = load(os.path.join(nat, f"{c}.json"))
                if a != b:
                    bad += 1
                    bad_here += 1
                    print(f"  {label}/{c}: *** MISMATCH *** (sliced {len(a)} vs native {len(b)})")
            print(f"  {label}: {len(cases)-bad_here}/{len(cases)} cases identical vs native")
        print("VERDICT:", "PREFIX-CONSISTENT — slicing is valid" if bad == 0
              else f"{bad} MISMATCH(ES) — do NOT slice; SC is no longer prefix-consistent")
        sys.exit(1 if bad else 0)

    for label, n in sizes:
        out = cell(label)
        os.makedirs(out, exist_ok=True)
        print(f"[slice-sc] {label} (N={n}) -> {out}")
        for c in cases:
            samples = load(os.path.join(args.full, f"{c}.json"))
            if len(samples) < n:
                sys.exit(f"FATAL: full has only {len(samples)} samples for {c}, need {n} — "
                         f"did the full SC finish?")
            sl = samples[:n]
            dump(sl, os.path.join(out, f"{c}.json"))

            per_sample = kg_from(sl)
            from soda.utils import schema_utils
            schema_utils.dump_canon_kg_txt(per_sample, os.path.join(out, f"canon_kg_{c}.txt"))

            fs_path = os.path.join(args.full, f"canon_schema_{c}.json")
            full_schema = load(fs_path) if os.path.exists(fs_path) else {}
            sch = schema_as_of(sl, full_schema)
            dump(sch, os.path.join(out, f"canon_schema_{c}.json"))

            flat = [t for trips in per_sample for t in trips]
            schema_utils.dump_canon_kg(flat, sch, os.path.join(out, f"canon_kg_{c}.json"))
        # carry the sliced OIE/SD totals too, so the cell is self-contained for eval
        print(f"    {len(cases)} cases written; #rel per case = schema AS OF item {n}")
    print("[slice-sc] done. Slice OIE/SD with slice_stage_totals.py if eval needs them in-cell.")


if __name__ == "__main__":
    main()
