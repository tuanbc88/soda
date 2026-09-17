"""Recover (id, gold_type, confidence) from the damaged second annotation sheet.

The sheet came back with each data line wrapped in an outer pair of quotes (inner quotes
doubled), and the wrap did not survive every row. Two distinct damage patterns result, and a
single "read the Nth field from the end" rule silently mis-reads one of them:

  A. note survived as its own field   -> [... , gold_type, confidence, note]      note = ";;;;;;" or ""
  B. note merged into confidence      -> [... , gold_type, "3;;;;;;;"]

Everything left of gold_type may carry extra commas, so the anchor has to come from the right.
The recovery is then VERIFIED rather than trusted: id, mention and context are known from the
original blind sheet, so every recovered row is checked against it by id and by mention text.
Nothing here infers a label; a row that cannot be verified is reported, not repaired.
"""
import argparse, csv, re, sys, unicodedata

def norm(s):
    s = unicodedata.normalize("NFC", (s or "")).strip().lower()
    return re.sub(r"\s+", " ", s)

def unwrap(line):
    """Stage 1: the whole record is usually one quoted field. Stage 2: parse the real record."""
    outer = next(csv.reader([line]))
    return next(csv.reader([outer[0]])) if len(outer) == 1 else outer

SEMI_ONLY = re.compile(r"^;*$")
CONF_MERGED = re.compile(r"^\s*([0-3])\s*;+\s*$")

TYPE_TOKEN = re.compile(r"^[A-Za-zÀ-ỹ0-9_/-]{1,40}$")

def split_tail(rec):
    """Return (gold_type, confidence, pattern) using the right-hand anchor.

    Trailing empty cells are dropped by the exporter, so a record that stops after context
    is an UNLABELLED row, not a damaged one. That case has to be recognised before any
    right-hand anchor is applied, or the context text gets read as a type label.
    """
    if len(rec) <= 3:
        return "", "", "blank"
    last = rec[-1].strip()
    if SEMI_ONLY.match(last):                 # pattern A: note is its own (junk) field
        return rec[-3].strip(), rec[-2].strip(), "A"
    m = CONF_MERGED.match(last)
    if m:                                     # pattern B: note merged into confidence
        return rec[-2].strip(), m.group(1), "B"
    if last.isdigit():                        # pattern C: note dropped entirely
        return rec[-2].strip(), last, "C"
    # pattern D: the annotator wrote a real note ("Thieu context"), so the note field is
    # prose rather than junk. Same anchor as A, but only if it yields a plausible label.
    cand_gold, cand_conf = rec[-3].strip(), rec[-2].strip()
    if TYPE_TOKEN.match(cand_gold) and (cand_conf == "" or cand_conf.strip(";").isdigit()):
        return cand_gold, cand_conf.strip(";"), "D"
    # pattern E: no label at all, and a note saying why ("Thieu context", "Khong lien quan").
    # A refusal to label is a finding about the sample, not a damaged row.
    if cand_gold == "" and cand_conf == "":
        return "", "", "declined"
    return None, None, "?"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--damaged", required=True, help="the returned sheet to recover")
    ap.add_argument("--original", required=True, help="the blind sheet, used to VERIFY each row")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    orig = {r["id"].strip(): r for r in csv.DictReader(open(args.original, encoding="utf-8-sig"))}
    raw = open(args.damaged, encoding="utf-8-sig").read().splitlines()

    rows, unresolved = {}, []
    for lineno, ln in enumerate(raw[1:], start=2):
        if not ln.strip():
            continue
        rec = unwrap(ln)
        m = re.match(r"^\s*(\d+)", rec[0])
        if not m:
            unresolved.append((lineno, "no leading id", rec[0][:60])); continue
        rid = m.group(1)
        gold, conf, pat = split_tail(rec)
        if gold is None:
            unresolved.append((lineno, f"id={rid} tail unparsed", rec[-1][:40])); continue

        # VERIFY against the blind sheet: the id must exist, and the mention we recovered
        # must be the mention that id actually carries.
        src = orig.get(rid)
        if src is None:
            unresolved.append((lineno, f"id={rid} not in original", "")); continue
        blob = norm(" ".join(rec))
        ok_mention = norm(src["mention"]) in blob
        # a label that reads as prose is the mis-anchored case this guards against
        looks_like_prose = gold != "" and (len(gold) > 40 or " " in gold.strip())
        if not ok_mention or looks_like_prose:
            unresolved.append((lineno, f"id={rid} verify failed", f"gold={gold[:40]!r}")); continue

        note = rec[-1].strip().strip(";") if pat in ("A", "D", "declined") else ""
        rows[rid] = {"id": rid, "gold_type": gold, "confidence_1_3": conf, "pattern": pat,
                     "mention": src["mention"], "context": src["context"], "note": note}

    missing = [i for i in orig if i not in rows and not any(u[1].startswith(f"id={i} ") for u in unresolved)]

    print(f"original rows          : {len(orig)}")
    print(f"recovered + verified   : {len(rows)}")
    print(f"unresolved             : {len(unresolved)}")
    print(f"absent from the sheet  : {len(missing)} -> {sorted(missing, key=int)[:20]}")
    from collections import Counter
    print("pattern mix            :", dict(Counter(r['pattern'] for r in rows.values())))
    labelled = {k: v for k, v in rows.items() if v["gold_type"]}
    print(f"of those, labelled     : {len(labelled)}")
    if unresolved:
        print("\n--- rows needing a human eye ---")
        for lineno, why, extra in unresolved:
            print(f"  line {lineno}: {why} {extra}")

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        # Same schema as the undamaged sheet, so the scorer consumes it unchanged.
        w = csv.writer(fh)
        w.writerow(["id", "mention", "context", "gold_type", "confidence_1_3", "note"])
        for rid in sorted(rows, key=int):
            r = rows[rid]
            w.writerow([r["id"], r["mention"], r["context"], r["gold_type"],
                        r["confidence_1_3"], r["note"]])

if __name__ == "__main__":
    main()
