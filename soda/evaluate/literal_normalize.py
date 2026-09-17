"""
Literal/value canonicalization for FAIR text-to-KG evaluation (RQ2 assess asset).

WHY (DECISIONS 2026-06-24). The OIE_MISS diagnosis showed the dominant "miss" on
webnlg is NOT extraction failure but SURFACE/LITERAL-FORMAT mismatch: the gold KG
stores literals in canonical DBpedia/RDF form (dates as YYYY-MM-DD, numbers as
"3500.0") while the text — and therefore a faithful extractor — uses natural forms
("10th of March 1983", "3,500"). Scoring those as wrong over-charges the extractor;
it is, in the author's words, "danh do" (an unfair trick). The temporal-expression
normalization subfield (TimeML/TIMEX3, HeidelTime/SUTime) exists precisely because
raw-vs-canonical literal mismatch is otherwise unusable.

This module provides a single canonicalizer applied SYMMETRICALLY to predicted and
gold triples before scoring, so the metric measures CONTENT, not literal formatting.
It serves two consumers:
  - `run_full_evaluation.py` -> a 2nd strict/exact/partial table on normalized triples
    (reported ALONGSIDE the raw one; the delta = the "literal-format gap" metric).
  - `oie_miss_diagnosis.py`  -> the F_literal_format bucket (a gold literal that is
    realized in the text under normalization, just in a different surface form).

SCOPE: dates + numeric/measurement literals (the clean, defensible "format" cases,
~74% of the webnlg A bucket). ALIASES ("United States"/"USA") and MORPHOLOGY
("Singing"/"singer") are deliberately NOT normalized here — they need entity linking
/ lemmatization, a different remedy — so they stay attributable separately.

English-only (the 3 gold datasets are English). dependency-free (regex).
"""

import re

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))

# a bare number, optional sign, optional thousands separators, optional decimals
_NUM_FULL = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$|^[+-]?\d+(?:\.\d+)?$")
_NUM_FIND = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+-]?\d+(?:\.\d+)?")

_ISO_DATE = re.compile(r"^(\d{4})-(\d{1,2})(?:-(\d{1,2}))?$")


def _strip(s):
    if not isinstance(s, str):
        s = str(s)
    return s.strip().strip('"').strip("'").strip()


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------
def parse_number(s):
    """Full-string number -> float, else None ('3,990'->3990.0, '3500.0'->3500.0).
    Requires the WHOLE trimmed string to be a number so '11th_Mississippi' is NOT a number."""
    t = _strip(s)
    if not _NUM_FULL.match(t):
        return None
    try:
        return float(t.replace(",", ""))
    except ValueError:
        return None


def find_numbers(text):
    out = []
    for m in _NUM_FIND.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _num_str(x):
    """Canonical numeric string: integer-valued floats lose the '.0' ('3500.0'->'3500')."""
    return str(int(x)) if float(x).is_integer() else repr(float(x))


# ---------------------------------------------------------------------------
# dates  ->  (year, month, day|None)
# ---------------------------------------------------------------------------
def parse_date(s):
    t = _strip(s)
    m = _ISO_DATE.match(t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), m.group(3)
        return (y, mo, int(d) if d else None)
    low = t.lower()
    # "10th of March 1983" / "10 March 1983"
    m = re.match(rf"^(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RE})\.?,?\s+(\d{{4}})$", low)
    if m:
        return (int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))
    # "March 10th, 1983" / "March 10 1983"
    m = re.match(rf"^({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})$", low)
    if m:
        return (int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))
    # "March 1983"
    m = re.match(rf"^({_MONTH_RE})\.?\s+(\d{{4}})$", low)
    if m:
        return (int(m.group(2)), _MONTHS[m.group(1)], None)
    return None


def find_dates(text):
    """All (y, m, d|None) dates mentioned in free text."""
    low = (text or "").lower()
    out = []
    for m in re.finditer(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RE})\.?,?\s+(\d{{4}})", low):
        out.append((int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1))))
    for m in re.finditer(rf"({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})", low):
        out.append((int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2))))
    for m in re.finditer(r"(\d{4})-(\d{1,2})-(\d{1,2})", low):
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return out


def dates_match(a, b, lenient=True):
    """a,b = (y,m,d|None). lenient: same year + same {month,day} multiset (absorbs the
    webnlg month/day convention swap). strict: exact y,m,d."""
    if a is None or b is None:
        return False
    if a[0] != b[0]:
        return False
    if not lenient:
        return a == b
    am = {x for x in a[1:] if x is not None}
    bm = {x for x in b[1:] if x is not None}
    return am == bm if (am and bm) else (a[1] == b[1])


# ---------------------------------------------------------------------------
# classification + canonical string (for eval)
# ---------------------------------------------------------------------------
def classify_literal(s):
    if parse_date(s) is not None:
        return "date"
    if parse_number(s) is not None:
        return "number"
    return "other"


def normalize_literal_str(s):
    """Canonical string for symmetric pred/gold matching. Dates -> ISO, numbers ->
    canonical numeric, otherwise a light surface clean (the downstream metric does the
    rest of the token normalization). Aliases/morphology are intentionally untouched."""
    t = _strip(s)
    d = parse_date(t)
    if d is not None:
        y, mo, dd = d
        return f"{y:04d}-{mo:02d}-{dd:02d}" if dd is not None else f"{y:04d}-{mo:02d}"
    n = parse_number(t)
    if n is not None:
        return _num_str(n)
    out = t.lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", out).strip()


def normalize_triplet(trip):
    """[h, r, t] -> [norm(h), r, norm(t)]. Relation left as-is (literal-format only)."""
    if not isinstance(trip, (list, tuple)) or len(trip) < 3:
        return trip
    return [normalize_literal_str(trip[0]), trip[1], normalize_literal_str(trip[2])]


# ---------------------------------------------------------------------------
# F-bucket helper for the OIE_MISS diagnosis
# ---------------------------------------------------------------------------
def literal_realized_in_text(value, text, lenient_date=True, num_tol=1e-6):
    """True if `value` is a date/number literal whose normalized form appears in the
    source text (so the fact IS verbalized, just in a different surface form -> F)."""
    kind = classify_literal(value)
    if kind == "number":
        gv = parse_number(value)
        return any(abs(gv - x) <= num_tol for x in find_numbers(text))
    if kind == "date":
        gd = parse_date(value)
        return any(dates_match(gd, td, lenient=lenient_date) for td in find_dates(text))
    return False


if __name__ == "__main__":
    # self-test
    assert parse_number("3,990") == 3990.0
    assert parse_number("3500.0") == 3500.0
    assert parse_number("11th_Mississippi") is None
    assert _num_str(3500.0) == "3500"
    assert parse_date("1983-10-03") == (1983, 10, 3)
    assert parse_date("10th of March 1983") == (1983, 3, 10)
    assert parse_date("November 18th 1923") == (1923, 11, 18)
    assert parse_date("October 13, 1964") == (1964, 10, 13)
    assert classify_literal("3500.0") == "number"
    assert classify_literal("1983-10-03") == "date"
    assert classify_literal("United States") == "other"
    assert normalize_literal_str("3500.0") == "3500"
    assert normalize_literal_str("November 18th 1923") == "1923-11-18"
    # number realized in text
    assert literal_realized_in_text("3500.0", "has a 3500 long runway")
    assert literal_realized_in_text("610.0", "is 610 metres above sea level")
    # date realized in text (lenient absorbs the month/day swap)
    assert literal_realized_in_text("1923-11-18", "Born on November 18th 1923")
    assert literal_realized_in_text("1983-10-03", "first aired on the 10th of March, 1983")
    # alias NOT a literal-format case
    assert not literal_realized_in_text("United States", "located in the USA")
    # symmetric triplet normalization
    assert normalize_triplet(["Al Asad", "runwayLength", "3990.0"])[2] == "3990"
    print("literal_normalize self-test: ALL PASS")
