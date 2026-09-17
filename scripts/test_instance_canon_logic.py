"""Local unit tests for the entity-INSTANCE canonicalizer (RQ §8.5, DECISIONS 2026-07-16b).

CPU-only, no LLM, no server: a fake embedder returns scripted cosines, so the MERGE LOGIC is
tested deterministically and independently of bge-m3. What it cannot test is whether real
embeddings put the right pairs above threshold — that is a server/eval question.

Run:  python scripts/test_instance_canon_logic.py
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soda import entity_instance_canonicalization as eic  # noqa: E402


class FakeEmbedder:
    """Encodes each text as a unit vector so that cos(a,b) = SIMS[(a,b)], defaulting to 0.

    Implemented by assigning each text its own basis vector and then mixing in the requested
    similarity — for the small scripted cases here an explicit Gram-matrix factorization is
    simplest and exact.
    """

    def __init__(self, sims):
        self.sims = sims

    def encode(self, texts, **kwargs):
        n = len(texts)
        gram = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                s = self.sims.get((texts[i], texts[j]), self.sims.get((texts[j], texts[i]), 0.0))
                gram[i, j] = gram[j, i] = s
        # nearest PSD factorization (clip negative eigenvalues), then renormalize rows
        w, v = np.linalg.eigh(gram)
        w = np.clip(w, 1e-9, None)
        x = v @ np.diag(np.sqrt(w))
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-12)


def _canon(surfaces, sims, case, type_map=None, desc_map=None, desc_sims=None):
    freq = {s: 1 for s in surfaces}
    all_sims = dict(sims)
    if desc_sims:
        all_sims.update(desc_sims)
    ic = eic.EntityInstanceCanonicalizer(FakeEmbedder(all_sims))
    return ic.canonicalize(freq, type_map or {}, desc_map or {}, case)


def _merged(canon_map, a, b):
    return canon_map.get(a) == canon_map.get(b)


def test_fold_equal_merges_without_embeddings():
    cmap, _ = _canon(["HCMUT", "hcmut!", "Other"], {}, "ic1_name")
    assert _merged(cmap, "HCMUT", "hcmut!"), "fold-equal must merge"
    assert not _merged(cmap, "HCMUT", "Other")
    print("PASS  fold-equal merges (rule a, no embedder needed)")


def test_underscore_surfaces_tokenize(self=None):
    """★ Regression: SODA's OIE emits `Barack_Obama`; `_` is \\w so a naive fold leaves ONE token,
    which silently kills token-subset AND both guards on exactly the hotpot/musique chunk corpora
    where §8.5 is evaluated. See _fold's docstring."""
    assert eic._tokens("Barack_Obama") == {"barack", "obama"}, eic._tokens("Barack_Obama")
    assert eic._fold("Barack_Obama") == eic._fold("Barack Obama"), "underscored == spaced form"
    # rule (b) must fire through underscores
    assert eic._tokens("Obama") <= eic._tokens("Barack_Obama")
    # the guards must still SEE the words they guard on
    assert eic._tokens("Halifax_County") & eic.TYPE_NOUNS, "type-noun guard must see 'county'"
    assert eic._is_temporal("February_1917"), "temporal guard must see the year"
    assert eic._is_temporal("the_1990s"), "temporal guard must see the decade"
    print("PASS  underscore-joined surfaces tokenize (rule b + both guards survive OIE's format)")


def test_underscore_guards_actually_block():
    """The guards must block through underscores, not just tokenize."""
    cmap, _ = _canon(["Halifax", "Halifax_County"], {("Halifax", "Halifax_County"): 0.99}, "ic1_name")
    assert not _merged(cmap, "Halifax", "Halifax_County"), "type-noun guard must block underscored"
    cmap, _ = _canon(["1917", "February_1917"], {("1917", "February_1917"): 0.99}, "ic1_name")
    assert not _merged(cmap, "1917", "February_1917"), "temporal guard must block underscored"
    cmap, _ = _canon(["Obama", "Barack_Obama"], {("Obama", "Barack_Obama"): 0.95}, "ic1_name")
    assert _merged(cmap, "Obama", "Barack_Obama"), "but a legit subset must still merge"
    print("PASS  guards block through underscores; legitimate underscored subsets still merge")


def test_token_subset_merges():
    cmap, _ = _canon(["Obama", "Barack Obama"], {("Obama", "Barack Obama"): 0.95}, "ic1_name")
    assert _merged(cmap, "Obama", "Barack Obama"), "token-subset above cos_sub must merge"
    print("PASS  token-subset merges (rule b)")


def test_temporal_guard_blocks_fuzzy_merge():
    # the real over-merge that hurt sp_f1 before the 2026-07-12 fix
    cmap, _ = _canon(["1990s", "early 2000s"], {("1990s", "early 2000s"): 0.99}, "ic1_name")
    assert not _merged(cmap, "1990s", "early 2000s"), "TEMPORAL guard must block"
    cmap, _ = _canon(["1917", "February 1917"], {("1917", "February 1917"): 0.99}, "ic1_name")
    assert not _merged(cmap, "1917", "February 1917"), "TEMPORAL guard must block"
    print("PASS  temporal guard blocks 1990s/early 2000s and 1917/February 1917")


def test_type_noun_guard_blocks_subset_and_alias():
    cmap, _ = _canon(["Halifax", "Halifax County"], {("Halifax", "Halifax County"): 0.99},
                     "ic1_name")
    assert not _merged(cmap, "Halifax", "Halifax County"), "TYPE-NOUN guard must block subset"
    cmap, _ = _canon(["singer", "songwriter"], {("singer", "songwriter"): 0.99}, "ic1_name")
    assert not _merged(cmap, "singer", "songwriter"), "TYPE-NOUN guard must block alias"
    print("PASS  type-noun guard blocks Halifax/Halifax County and singer/songwriter")


def test_guards_hold_even_when_type_agrees():
    """The guards encode IDENTITY — type agreement must not override them (ic2/ic3)."""
    tm = {"Halifax": "Location", "Halifax County": "Location"}
    cmap, _ = _canon(["Halifax", "Halifax County"], {("Halifax", "Halifax County"): 0.99},
                     "ic2_name_type", type_map=tm)
    assert not _merged(cmap, "Halifax", "Halifax County"), "guard must survive type agreement"
    tm = {"1990s": "Time", "early 2000s": "Time"}
    cmap, _ = _canon(["1990s", "early 2000s"], {("1990s", "early 2000s"): 0.99},
                     "ic2_name_type", type_map=tm)
    assert not _merged(cmap, "1990s", "early 2000s"), "guard must survive type agreement"
    print("PASS  guards hold in ic2 even when the canonical types agree")


def test_type_is_a_permit_ic2_merges_where_ic1_does_not():
    """The core design claim: ic2 merges a SUPERSET of ic1 (type PERMITS a looser match)."""
    sims = {("USA", "United States"): 0.94}       # below cos_alias 0.97, above cos_alias_typed 0.93
    surfaces = ["USA", "United States"]

    cmap1, st1 = _canon(surfaces, sims, "ic1_name")
    assert not _merged(cmap1, "USA", "United States"), "ic1 (name-only, strict) must NOT merge 0.94"

    tm = {"USA": "Country", "United States": "Country"}
    cmap2, st2 = _canon(surfaces, sims, "ic2_name_type", type_map=tm)
    assert _merged(cmap2, "USA", "United States"), "ic2 must merge when types agree"
    assert st2["merge_signal_counts"].get("typed_alias") == 1
    assert st2["n_surfaces_rewritten"] > st1["n_surfaces_rewritten"]
    print("PASS  ic2 merges USA/United States at cos 0.94 where ic1 does not (type = permit)")


def test_conflicting_type_blocks_the_loosened_merge():
    sims = {("Mercury", "Mercury"): 1.0, ("Mercury planet", "Mercury element"): 0.94}
    tm = {"Mercury planet": "Planet", "Mercury element": "ChemicalElement"}
    cmap, _ = _canon(["Mercury planet", "Mercury element"], sims, "ic2_name_type", type_map=tm)
    assert not _merged(cmap, "Mercury planet", "Mercury element"), "conflicting types must not merge"
    print("PASS  conflicting canonical types block the loosened merge")


def test_ic3_description_permits_the_loosest_merge():
    sims = {("MIT", "Massachusetts Institute of Technology"): 0.91}   # below typed 0.93
    surfaces = ["MIT", "Massachusetts Institute of Technology"]
    tm = {s: "University" for s in surfaces}
    dm = {"MIT": "a private research university in Cambridge",
          "Massachusetts Institute of Technology": "a private research university in Cambridge MA"}
    desc_sims = {(dm["MIT"], dm["Massachusetts Institute of Technology"]): 0.95}

    cmap2, _ = _canon(surfaces, sims, "ic2_name_type", type_map=tm)
    assert not _merged(cmap2, *surfaces), "ic2 must NOT merge at 0.91 (below cos_alias_typed)"

    cmap3, st3 = _canon(surfaces, sims, "ic3_name_type_desc", type_map=tm, desc_map=dm,
                        desc_sims=desc_sims)
    assert _merged(cmap3, *surfaces), "ic3 must merge when the descriptions agree too"
    assert st3["merge_signal_counts"].get("typed_desc_alias") == 1
    print("PASS  ic3 merges MIT/Massachusetts Institute of Technology via description agreement")


def test_ic3_disagreeing_description_blocks():
    sims = {("Mercury A", "Mercury B"): 0.91}
    tm = {"Mercury A": "Thing", "Mercury B": "Thing"}
    dm = {"Mercury A": "the smallest planet in the solar system",
          "Mercury B": "a liquid metal chemical element"}
    desc_sims = {(dm["Mercury A"], dm["Mercury B"]): 0.20}
    cmap, _ = _canon(["Mercury A", "Mercury B"], sims, "ic3_name_type_desc",
                     type_map=tm, desc_map=dm, desc_sims=desc_sims)
    assert not _merged(cmap, "Mercury A", "Mercury B"), "disagreeing descriptions must block"
    print("PASS  ic3 blocks when descriptions disagree despite matching type")


def test_missing_type_degrades_to_ic1():
    """No EC types available -> ic2 must behave exactly like ic1 (the caller warns about this)."""
    sims = {("USA", "United States"): 0.94}
    cmap1, st1 = _canon(["USA", "United States"], sims, "ic1_name")
    cmap2, st2 = _canon(["USA", "United States"], sims, "ic2_name_type", type_map={})
    assert cmap1 == cmap2, "ic2 without types must equal ic1"
    assert st1["n_surfaces_rewritten"] == st2["n_surfaces_rewritten"]
    print("PASS  ic2 degrades to ic1 when no entity types are available")


def test_rewrite_dedups_and_drops_self_loops():
    canon_map = {"Obama": "Barack Obama", "Barack Obama": "Barack Obama", "USA": "United States",
                 "United States": "United States"}
    per_sample = [
        [["Obama", "president_of", "USA"], ["Barack Obama", "president_of", "United States"]],
        [["USA", "same_as", "United States"]],
    ]
    out = eic.rewrite_triplets_per_sample(per_sample, canon_map)
    assert out[0] == [["Barack Obama", "president_of", "United States"]], \
        f"duplicate after rewrite must collapse, got {out[0]}"
    assert out[1] == [], f"self-loop after rewrite must drop, got {out[1]}"
    print("PASS  rewrite dedups collapsed duplicates and drops self-loops")


def test_surface_freq_union_across_relation_cases():
    """Surfaces are relation-case-invariant; a case that drops a triplet must not lose surfaces."""
    tbc = {
        "case1": [["A", "r1", "B"], ["B", "r2", "C"]],
        "case2": [["A", "r9", "B"]],                       # case2 dropped the B-C triplet
    }
    freq = eic.collect_surface_freq(tbc)
    assert set(freq) == {"A", "B", "C"}, f"union must keep C, got {set(freq)}"
    print("PASS  surface frequency unions across relation cases")


def test_descriptions_collected_from_sd():
    sd = [{"entities": [["HCMUT", "Organization", "University", "a public university"],
                        ["bad_row"],
                        ["X", "Thing", "Thing", ""]]}]
    dm = eic.collect_entity_descriptions(sd)
    assert dm == {"HCMUT": "a public university"}, dm
    print("PASS  SD1 descriptions collected, malformed/empty rows skipped")


def test_determinism():
    surfaces = ["USA", "United States", "Obama", "Barack Obama", "1990s"]
    sims = {("USA", "United States"): 0.98, ("Obama", "Barack Obama"): 0.95}
    a, _ = _canon(surfaces, sims, "ic1_name")
    b, _ = _canon(list(reversed(surfaces)), sims, "ic1_name")
    assert a == b, "canon map must not depend on input order"
    print("PASS  canonicalization is deterministic under input reordering")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} TESTS PASSED")
    print("NB: merge LOGIC only (scripted cosines). Whether real bge-m3 embeddings place the right")
    print("    pairs above threshold is a server/eval question — see RQ §8.5 + item #10 (threshold")
    print("    calibration). Thresholds here are UNTUNED defaults.")
