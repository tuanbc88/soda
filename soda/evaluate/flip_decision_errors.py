"""Paired decision-error analysis behind the canonicalization flip (paper M4 / Sec. res:canon).

The manuscript reports, for WebNLG Mode 3, that the gold relation sat in the losing signal's own
retrieved candidate list in 88% of the triples it lost to the winning signal -- i.e. the loss is a
*decision* error, not a retrieval failure. That number is PAIRED: its denominator is the triples the
winner got right and the loser did not, which is why it differs from the aggregate CANON_MISS share
already produced by error_attribution_metric.py.

This script computes the paired number for any (winner, loser) case pair on any dataset, so the
mechanism claim can be stated cross-dataset rather than on one point. It also dumps the confusion
pairs (gold relation -> relation actually chosen) that the qualitative taxonomy is read off.

No GPU and no re-run: it reads the sc_mcp_logs.jsonl already on disk.

Usage
-----
    python soda/evaluate/flip_decision_errors.py --dataset webnlg \
        --pred_dir output/webnlg_selfcanon2_mode3_item_qwen3-8b_bgem3_A100_0627open/iter0 \
        --winner case3_name_gendef_edc --loser case5_name_detail

    # all three benchmarks at their seeded winner vs the signal the flip says should lose:
    python soda/evaluate/flip_decision_errors.py --preset flip
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from soda.evaluate.error_attribution_metric import build_gold_triples, build_oie_index
from soda.evaluate.retrieval_recall_metric import GT_CONFIG, load_kg_txt


# Seeded (Mode-3) winner per dataset and the competing signal from the other family.
# WebNLG's inventory is verbalized, so the concise fixed-template wins and the detailed signal loses;
# REBEL and Wiki-NRE are Wikidata-style, where the detailed typed signal wins and the concise loses.
# The pair is therefore mirrored on purpose -- that mirroring IS the mechanism claim under test.
PRESET_FLIP = [
    ("webnlg", "case3_name_gendef_edc", "case5_name_detail"),
    ("rebel", "case7_detail_typed", "case3_name_gendef_edc"),
    ("wiki-nre", "case7_detail_typed", "case3_name_gendef_edc"),
]

# The six pairs behind the table in RESULTS_flip_mechanism.md. The three extra pairs answer the question
# the flip pairs alone cannot: is a low decision-error share a property of the DATASET or of the losing
# SIGNAL? Holding the dataset fixed and swapping only the loser separates them.
PRESET_FULL = PRESET_FLIP + [
    ("webnlg", "case3_name_gendef_edc", "case7_detail_typed"),
    ("rebel", "case7_detail_typed", "case5_name_detail"),
    ("wiki-nre", "case7_detail_typed", "case5_name_detail"),
]

PRED_DIR_TMPL = "output/{ds}_selfcanon2_mode3_item_qwen3-8b_bgem3_A100_0627open/iter0"


def analyze(gt_kg, pred_dir, winner, loser):
    """Paired comparison of two canonicalization signals over the same gold triples."""
    gold_triples = build_gold_triples(load_kg_txt(gt_kg))
    win_index, _ = build_oie_index(pred_dir, winner)
    los_index, _ = build_oie_index(pred_dir, loser)

    n_gold = 0
    paired_losses = 0          # winner correct, loser not
    decision_errors = 0        # ... and gold was in the loser's own candidate list
    retrieval_errors = 0       # ... and it was not
    paired_wins = 0            # loser correct, winner not (the reverse flow, for symmetry)
    both_correct = 0
    confusions = Counter()     # (gold_rel, chosen_rel) over decision errors

    for idx, golds in enumerate(gold_triples):
        wrec = win_index.get(idx, {})
        lrec = los_index.get(idx, {})
        for (gh, gt_, gr) in golds:
            n_gold += 1
            key = (gh, gt_)
            we, le = wrec.get(key), lrec.get(key)
            # Only triples BOTH signals recovered at extraction are comparable: a triple neither
            # saw is an OIE loss, which this analysis is not about.
            if we is None or le is None:
                continue
            w_ok = gr in we["finals"]
            l_ok = gr in le["finals"]
            if w_ok and l_ok:
                both_correct += 1
                continue
            if l_ok and not w_ok:
                paired_wins += 1
                continue
            if not (w_ok and not l_ok):
                continue
            paired_losses += 1
            if gr in le["cands"]:
                decision_errors += 1
                for chosen in sorted(le["finals"]):
                    if chosen != gr:
                        confusions[(gr, chosen)] += 1
            else:
                retrieval_errors += 1

    share = round(100.0 * decision_errors / paired_losses, 1) if paired_losses else None
    return {
        "winner": winner,
        "loser": loser,
        "n_gold": n_gold,
        "both_correct": both_correct,
        "paired_losses": paired_losses,
        "decision_errors": decision_errors,
        "retrieval_errors": retrieval_errors,
        "decision_error_pct": share,
        "paired_wins_for_loser": paired_wins,
    }, confusions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="webnlg|rebel|wiki-nre")
    ap.add_argument("--pred_dir", help="iter dir holding sc_mcp_logs.jsonl")
    ap.add_argument("--winner")
    ap.add_argument("--loser")
    ap.add_argument("--preset", choices=["flip", "full"],
                    help="flip: the three seeded-winner pairs; full: all six pairs of the results table")
    ap.add_argument("--gt_kg", help="explicit gold KG .txt (overrides --dataset)")
    ap.add_argument("--out_dir", default="output/_flip_decision_errors")
    ap.add_argument("--top_confusions", type=int, default=25)
    args = ap.parse_args()

    if args.preset:
        preset = PRESET_FULL if args.preset == "full" else PRESET_FLIP
        jobs = [(ds, PRED_DIR_TMPL.format(ds=ds), w, l) for ds, w, l in preset]
    else:
        if not (args.winner and args.loser):
            ap.error("--winner and --loser are required without --preset")
        pred_dir = args.pred_dir or PRED_DIR_TMPL.format(ds=args.dataset)
        jobs = [(args.dataset, pred_dir, args.winner, args.loser)]

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    for ds, pred_dir, winner, loser in jobs:
        gt_kg = args.gt_kg or GT_CONFIG[ds]
        print(f"\n=== {ds}: winner {winner} vs loser {loser}")
        print(f"    gold {gt_kg}\n    pred {pred_dir}")
        res, confusions = analyze(gt_kg, pred_dir, winner, loser)
        res["dataset"] = ds
        summary.append(res)
        print(f"    comparable+winner-correct losses : {res['paired_losses']}")
        print(f"    gold was in loser's candidates   : {res['decision_errors']} "
              f"({res['decision_error_pct']}%)  <- decision error")
        print(f"    gold was not retrieved           : {res['retrieval_errors']}")
        print(f"    (reverse: loser right, winner wrong: {res['paired_wins_for_loser']})")

        conf_path = os.path.join(args.out_dir, f"confusions_{ds}_{winner}_vs_{loser}.csv")
        with open(conf_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gold_relation", "chosen_relation", "count"])
            for (gr, cr), n in confusions.most_common():
                w.writerow([gr, cr, n])
        print(f"    confusion pairs -> {conf_path}")
        for (gr, cr), n in confusions.most_common(args.top_confusions):
            print(f"        {n:4d}  {gr}  ->  {cr}")

    out = os.path.join(args.out_dir, "flip_decision_errors.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
