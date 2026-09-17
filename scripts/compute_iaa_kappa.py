import io, sys, csv, os, openpyxl
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
RESP = r"E:\git-workspace\edc\assets\review\IAA_annotator2_gold_audit__response.xlsx"
OUT = r"E:\git-workspace\edc\output"

def cohen_kappa(pairs, labels):
    n = len(pairs)
    if not n: return 0.0, 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    pa = {l: sum(1 for a, _ in pairs if a == l) / n for l in labels}
    pb = {l: sum(1 for _, b in pairs if b == l) / n for l in labels}
    pe = sum(pa[l] * pb[l] for l in labels)
    return po, ((po - pe) / (1 - pe) if pe < 1 else 1.0)

def verdict(q1, q2, q3):
    q1, q2, q3 = (str(x).strip().lower() for x in (q1, q2, q3))
    if q1 == "no" or q2 == "no": return "NOT_VERBALIZED"
    if q3 in ("no",): return "UNSUPPORTED"
    if q3 == "inferable": return "IMPLICIT_OK"
    if q3 in ("stated", "yes"): return "SUPPORTED"   # 'yes' = B's collapsed vocabulary
    return None

def read_sheet(sheet):
    ws = openpyxl.load_workbook(RESP, data_only=True)[sheet]
    hdr = [str(c.value).strip() if c.value else "" for c in ws[1]]
    def ci(*names):
        for n in names:
            if n in hdr: return hdr.index(n) + 1
        return None
    c1, c2, c3 = ci("q1", "q1_head_in_sentence"), ci("q2", "q2_tail_in_sentence"), ci("q3", "q3_relation_expressed")
    out = []
    for r in range(2, ws.max_row + 1):
        v = verdict(ws.cell(row=r, column=c1).value, ws.cell(row=r, column=c2).value, ws.cell(row=r, column=c3).value)
        out.append(v)
    return out

def read_llm(ds):
    p = os.path.join(OUT, "gold_audit_" + ds, "gold_audit.csv")
    if not os.path.exists(p): return None
    return [r["verdict"].strip() for r in csv.DictReader(open(p, encoding="utf-8"))]

LAB = ["SUPPORTED", "IMPLICIT_OK", "UNSUPPORTED", "NOT_VERBALIZED"]
BIN = {"SUPPORTED": "OK", "IMPLICIT_OK": "OK", "UNSUPPORTED": "SUSPECT", "NOT_VERBALIZED": "SUSPECT"}

sets = [("webnlg", "WebNLG", "WebNLG_2"), ("rebel", "REbel", "Rebel_2"), ("wiki-nre", "Wiki-NRE", "Wiki_NRE_2")]
for ds, sa, sb in sets:
    A, B, L = read_sheet(sa), read_sheet(sb), read_llm(ds)
    n = min(len(A), len(B), len(L) if L else 10 ** 9)
    print(f"\n===== {ds.upper()}  (comparable rows n={n}; A={len(A)}, B={len(B)}, LLM={len(L) if L else 0}) =====")
    print(f"  A dist  : {dict(Counter(A[:n]))}")
    print(f"  B dist  : {dict(Counter(B[:n]))}")
    if L: print(f"  LLM dist: {dict(Counter(L[:n]))}")
    for nm, x, y in [("A vs B", A, B), ("LLM vs A", L, A), ("LLM vs B", L, B)]:
        if x is None or y is None: continue
        pr = [(x[i], y[i]) for i in range(n) if x[i] and y[i]]
        po, k = cohen_kappa(pr, LAB)
        prb = [(BIN[a], BIN[b]) for a, b in pr]
        pob, kb = cohen_kappa(prb, ["OK", "SUSPECT"])
        print(f"  {nm:9s} n={len(pr):3d}  agree={po:.3f} kappa={k:+.3f}   | binary(OK/SUSPECT) agree={pob:.3f} kappa={kb:+.3f}")
