"""Entity-INSTANCE canonicalization (RQ §8.5, IN-PIPELINE; DECISIONS 2026-07-16b,
author-approved: 3-case signal axis + eduhcmut `8-` gold).

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
Three entity passes now exist; keep them straight:
  * SC  (relation canon)                -> maps a local RELATION onto a global relation type.
  * EC  (`entity_canonicalization.py`)  -> maps a local entity fine-TYPE onto a global entity
                                           TYPE. Nodes keep their raw OIE surface string, so EC
                                           cannot move retrieval recall (RQ §8.2).
  * IC  (this file)                     -> merges entity INSTANCES: clusters surface forms and
                                           rewrites head/tail to a canonical surface, so chunks
                                           that only exact-string matching would leave apart now
                                           share a node. This is the GraphRAG-recall lever (§8.5).

WHY IN-PIPELINE (the point of the 3 cases)
------------------------------------------
The post-hoc aligner (`kg2rag/adapters/entity_instance_align.py`) runs after KGC on the adapted
KG2RAG dict, so the ONLY signal it has is the surface string + its embedding. Having nothing else,
it had to buy precision with tight thresholds (0.90/0.97) + guards, and merged just 2.3-2.5% of
surfaces -> recall-null, a cost/connectivity lever only (RESULTS_kg2rag_rq3 §8b).

In-pipeline, the merge decision can additionally see the SD1 locals (coarse/fine type, description)
and the EC canonical type. So IC is a CaSA-style SIGNAL ablation, mirroring the relation 9-case and
entity-type 3-case design spaces:

    ic1_name            surface name only                    (~= the post-hoc aligner)
    ic2_name_type       name + canonical entity TYPE agrees
    ic3_name_type_desc  name + type + SD1 description agrees

MECHANISM: type is a PERMIT, not a veto (this is what makes the hypothesis testable). Every case
shares the same strict name-only rules; ic2/ic3 additionally ALLOW a merge at a LOOSER cosine when
the extra signal corroborates it. So ic2/ic3 merge a SUPERSET of ic1. Had the extra signal merely
added constraints, ic2/ic3 would merge fewer by construction and the hypothesis could not be tested.

  HYPOTHESIS (DECISIONS 2026-07-16b): type-conditioning loosens the thresholds at equal precision
  -> more merges -> the recall lever has room to move.
  CONFIRMS: ic2/ic3 merge materially more surfaces than ic1 at equal-or-better purity AND lift
            extrinsic recall@k.
  REFUTES : ic2/ic3 ~= ic1 on merge rate and recall => instance canon is a cost lever in general,
            not just in the RQ3 setup; close §8.5 as such.

GUARDS (ported verbatim from the aligner :52-75/:121-176 — learned from REAL over-merges, keep them
in every case; they encode identity, which no amount of type agreement overrides):
  * TEMPORAL  : year/decade/month/century surfaces only merge when fold-equal
                ("1990s" vs "early 2000s", "1917" vs "February 1917").
  * TYPE-NOUN : a token-subset merge is blocked when the differing tokens include a category noun
                ("Halifax" vs "Halifax County"); a semantic alias between two DIFFERENT single
                type-nouns is blocked ("singer" vs "songwriter", "comedian" vs "actor").
  NB the aligner's TITLE GUARD is deliberately NOT ported: it anchors on hotpot/musique document
  titles, which title-less corpora such as the eduhcmut chunks do not have (RQ §8.5). On hotpot the
  post-hoc arm still has it.

SCOPE GUARD (§8.2, non-negotiable): this writes NEW artifacts and never touches
`canon_kg_{case}.txt` — the headline-scored KG keeps raw-surface entities so the EDC comparison
basis stays clean. GraphRAG-scope corpora only; the caller warns on gold-triple sets.

THRESHOLDS BELOW ARE UNTUNED DEFAULTS (the strict pair is inherited from the aligner, where it was
set by fixing observed over-merges; the loosened typed/desc pair is a starting guess). They are
subject to RQ item #10 (threshold human-calibration) and must not be quoted as tuned.
"""
import re
import json
from collections import Counter


# ---------------------------------------------------------------- surface folding

def _fold(s):
    """Casefold + collapse punctuation/underscores/whitespace to single spaces.

    ★ The `_` handling is NOT cosmetic and differs from the post-hoc aligner's `_fold`. SODA's OIE
    emits multiword entities underscore-joined (`Barack_Obama`, `February_1917`) — underscore is a
    SPACE SUBSTITUTE by construction, not a character. Python's `\\w` includes `_`, so the aligner's
    regex leaves it in place; that is harmless there because it runs AFTER `soda_to_kg2rag.py` has
    already converted underscores to spaces. IN-PIPELINE we see the underscored form, where keeping
    `_` would silently kill all three rules at once:
      * token-subset  : `Barack_Obama` tokenizes to ONE token -> `Obama` is never a subset (rule b dead);
      * TYPE-NOUN guard: `halifax_county` is not in TYPE_NOUNS -> the guard never fires and the exact
                         over-merges it was written to stop (Halifax vs Halifax County) come back;
      * TEMPORAL guard : `\\b\\d{3,4}s?\\b` finds no word boundary inside `February_1917` -> dead.
    Splitting on `_` fixes all three and is a no-op on space-separated corpora (no underscores to
    split), so it does not change behavior relative to the aligner on its own inputs. It also makes
    `Barack_Obama` and `Barack Obama` fold-equal, which is correct — they are the same entity.
    """
    s = re.sub(r"[^\w\s]|_", " ", str(s).casefold())
    return " ".join(s.split())


def _tokens(s):
    return set(_fold(s).split())


# Category/type nouns: if a token-subset pair differs by one of these, the extra token changes the
# entity's IDENTITY (town vs county, one occupation vs another) -> must NOT merge.
# Ported from kg2rag/adapters/entity_instance_align.py:52 (keep the two in sync).
TYPE_NOUNS = frozenset("""
county city state province district region territory town village borough municipality area
river lake island mountain sea ocean bay valley peninsula desert forest
university college school academy institute hospital airport station stadium arena museum
library church temple mosque prison park bridge tower castle palace
company corporation firm agency organization organisation association federation union party
band group orchestra choir ensemble team club league franchise squad
film movie album song single track series season episode novel book magazine newspaper journal
comic game show programme program tour
actor actress singer songwriter comedian musician composer producer director writer author poet
novelist painter artist politician president governor senator mayor player coach manager engineer
scientist lawyer doctor physician professor journalist economist philosopher soldier general
""".split())

_YEAR = re.compile(r"\b\d{3,4}s?\b")
_MONTHS = frozenset("january february march april may june july august september october "
                    "november december".split())


def _is_temporal(s):
    """True if the surface is a year / decade / month / century — these must only merge when
    fold-equal (rule a), never fuzzily (blocks '1990s' vs 'early 2000s')."""
    toks = _tokens(s)
    return bool(_YEAR.search(_fold(s)) or (toks & _MONTHS) or ("century" in toks))


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
        return ra


# ---------------------------------------------------------------- the 3-case design space

IC_ABLATION_CONFIG = {
    # use_type : a merge may be permitted at the loosened cosine when the two surfaces carry the
    #            SAME canonical entity type (from EC).
    # use_desc : additionally require their SD1 descriptions to agree (cosine >= desc_sim_min)
    #            before the most-loosened cosine applies.
    "ic1_name": {
        "use_type": False,
        "use_desc": False,
        "description": "surface name only (~= the post-hoc aligner; the §8.5 baseline)",
    },
    "ic2_name_type": {
        "use_type": True,
        "use_desc": False,
        "description": "name + canonical entity type must agree to permit a looser match",
    },
    "ic3_name_type_desc": {
        "use_type": True,
        "use_desc": True,
        "description": "name + type + SD1 description agreement permits the loosest match",
    },
}

DEFAULT_IC_CONFIG = {
    "cos_sub": 0.90,          # token-subset gate (all cases; aligner default)
    "cos_alias": 0.97,        # semantic alias, name-only evidence (all cases; aligner default)
    "cos_alias_typed": 0.93,  # ic2/ic3: alias permitted here when the canonical types agree
    "cos_alias_desc": 0.90,   # ic3: alias permitted here when types AND descriptions agree
    "desc_sim_min": 0.85,     # ic3: how similar two SD1 descriptions must be to count as agreeing
    "batch_size": 128,
}


class EntityInstanceCanonicalizer:
    """Cluster entity surface forms into instances under one signal case.

    Deliberately embedder-injected: in-pipeline we already hold `sc_embedder`, so IC reuses it
    rather than loading a second SentenceTransformer (the post-hoc aligner had to load its own).

    Usage:
        ic = EntityInstanceCanonicalizer(embedder, ic_config)
        canon_map, stats = ic.canonicalize(surfaces_freq, type_map, desc_map, case)
    """

    def __init__(self, embedder, ic_config=None):
        self.embedder = embedder
        self.cfg = dict(DEFAULT_IC_CONFIG)
        if ic_config:
            self.cfg.update(ic_config)
        self._emb_cache = {}

    # -------------------------------------------------- embedding (cached across the 3 cases)

    def _encode(self, texts, key):
        """Encode once per (key, texts-identity); the 3 IC cases share the same surface list."""
        if key in self._emb_cache:
            return self._emb_cache[key]
        import numpy as np
        vec = np.asarray(
            self.embedder.encode(texts, batch_size=self.cfg["batch_size"],
                                 normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )
        self._emb_cache[key] = vec
        return vec

    # -------------------------------------------------- the merge decision

    def _permit_alias(self, cos, i, j, case_cfg, types, desc_sim_fn):
        """Return (permitted, signal_name) for a semantic-alias merge of surfaces i,j at cosine
        `cos`. Name-only evidence needs the strict threshold; corroborating signals lower it."""
        if cos >= self.cfg["cos_alias"]:
            return True, "semantic_alias"

        if not case_cfg["use_type"]:
            return False, None
        ti, tj = types.get(i), types.get(j)
        if ti is None or tj is None or ti != tj:
            # no type evidence (or conflicting) -> only the strict name rule above could have fired
            return False, None

        if case_cfg["use_desc"]:
            if cos >= self.cfg["cos_alias_desc"] and desc_sim_fn(i, j) >= self.cfg["desc_sim_min"]:
                return True, "typed_desc_alias"
        if cos >= self.cfg["cos_alias_typed"]:
            return True, "typed_alias"
        return False, None

    # -------------------------------------------------- main entry

    def canonicalize(self, surfaces_freq, type_map, desc_map, case):
        """Cluster surfaces under one IC case.

        Args:
            surfaces_freq : {surface: frequency} — the union over all relation cases. (Surfaces are
                            relation-case-invariant: h/t come from `input_triplet`, only the
                            predicted relation varies — see edc_framework.extract_pred_triplets*.)
            type_map      : {surface: canonical_entity_type} from EC ({} when EC is off -> ic2/ic3
                            have no type evidence and degrade to ic1; the caller warns).
            desc_map      : {surface: SD1 description} ({} -> ic3 degrades to ic2).
            case          : one of IC_ABLATION_CONFIG.

        Returns (canon_map, stats): canon_map = {surface: canonical_surface} for every input
        surface (identity for unmerged ones).
        """
        import numpy as np

        if case not in IC_ABLATION_CONFIG:
            raise ValueError(f"unknown IC case {case!r}; expected one of {list(IC_ABLATION_CONFIG)}")
        case_cfg = IC_ABLATION_CONFIG[case]

        surfaces = sorted(surfaces_freq)                 # deterministic order
        idx = {s: i for i, s in enumerate(surfaces)}
        n = len(surfaces)
        uf = _UF(n)
        if n == 0:
            return {}, self._stats(case, [], 0, Counter(), {})

        # rule (a): fold-equal — no embedding needed, applies in every case
        fold_groups = {}
        for s in surfaces:
            fold_groups.setdefault(_fold(s), []).append(idx[s])
        for grp in fold_groups.values():
            for j in grp[1:]:
                uf.union(grp[0], j)

        # rules (b)+(c): embedding-gated
        emb = self._encode(surfaces, "surfaces")
        toks = [_tokens(s) for s in surfaces]
        types = {i: type_map.get(s) for i, s in enumerate(surfaces) if type_map.get(s)}

        # ic3 only: description embeddings, indexed by surface position
        desc_emb = None
        if case_cfg["use_desc"] and desc_map:
            descs = [str(desc_map.get(s, "") or "") for s in surfaces]
            if any(descs):
                desc_emb = self._encode(descs, "descs")

        def desc_sim(i, j):
            if desc_emb is None:
                return 0.0
            return float(desc_emb[i] @ desc_emb[j])

        # the widest gate we could fire on for this case = the lowest threshold in play
        gate = self.cfg["cos_sub"]
        if case_cfg["use_type"]:
            gate = min(gate, self.cfg["cos_alias_typed"])
        if case_cfg["use_desc"]:
            gate = min(gate, self.cfg["cos_alias_desc"])

        signal_counts = Counter()
        chunk = 1024
        for s0 in range(0, n, chunk):
            sims = emb[s0:s0 + chunk] @ emb.T             # (chunk, n)
            rows, cols = np.where(sims >= gate)
            for r0, c in zip(rows, cols):
                i = s0 + int(r0)
                j = int(c)
                if j <= i:
                    continue
                cos = float(sims[r0, c])
                # TEMPORAL guard: these only merge via rule (a)
                if _is_temporal(surfaces[i]) or _is_temporal(surfaces[j]):
                    continue
                if toks[i] and toks[j] and (toks[i] <= toks[j] or toks[j] <= toks[i]):
                    if cos < self.cfg["cos_sub"]:
                        continue
                    # TYPE-NOUN guard: the extra noun changes identity, not just adds a qualifier
                    if (toks[i] ^ toks[j]) & TYPE_NOUNS:
                        continue
                    uf.union(i, j)
                    signal_counts["token_subset"] += 1
                else:
                    # TYPE-NOUN guard: two DIFFERENT single type-nouns are categories, not aliases
                    if toks[i] != toks[j] and toks[i] <= TYPE_NOUNS and toks[j] <= TYPE_NOUNS:
                        continue
                    ok, signal = self._permit_alias(cos, i, j, case_cfg, types, desc_sim)
                    if not ok:
                        continue
                    uf.union(i, j)
                    signal_counts[signal] += 1

        # canonical per cluster = most frequent surface (tie: longest)
        clusters = {}
        for s in surfaces:
            clusters.setdefault(uf.find(idx[s]), []).append(s)
        canon_map = {}
        merges = []
        for members in clusters.values():
            best = max(members, key=lambda s: (surfaces_freq[s], len(s)))
            for s in members:
                canon_map[s] = best
            if len(members) > 1:
                merges.append({
                    "canonical": best,
                    "members": sorted(members, key=lambda s: -surfaces_freq[s]),
                    # carry the type so a human reviewing sample_merges can see WHY it merged
                    "types": sorted({type_map[s] for s in members if type_map.get(s)}),
                })

        return canon_map, self._stats(case, merges, n, signal_counts, type_map)

    # -------------------------------------------------- stats

    def _stats(self, case, merges, n_surfaces, signal_counts, type_map):
        sizes = Counter(len(m["members"]) for m in merges)
        # Merge purity proxy (no gold needed): a cluster is type-consistent if every member that HAS
        # a canonical type carries the same one.
        #
        # ⚠️ READ THIS BEFORE QUOTING IT. The proxy CONFLATES two different things and is only a
        # smell test, never a quality number:
        #   1. IC over-merged two entities of different kinds  <- what we want to detect;
        #   2. EC assigned inconsistent types to what is the SAME string  <- EC noise, not an IC error.
        # (2) is real and common: on hotpot, fold-equal clusters like Upper_house / upper_house /
        # Upper_House come back typed BOTH `Person` and `Thing`, which drags the rate to ~0.46 while
        # every one of those merges is definitionally correct. Rule (a) merges identical strings, so
        # ANY type disagreement inside a fold-equal cluster is (2) by construction.
        # So: read this per merge-signal, not in aggregate, and treat a drop between ic1 and ic2/ic3
        # as the signal — for ic2/ic3 the typed_alias path is ~1.0 pure by construction, so a low
        # overall rate there just means the fold-equal (a) clusters dominate. The real over-merge
        # check is human review of `sample_merges`.
        typed_clusters = [m for m in merges if len(m["types"]) >= 1]
        pure = sum(1 for m in typed_clusters if len(m["types"]) == 1)
        return {
            "ic_case": case,
            "signals": IC_ABLATION_CONFIG[case],
            "n_surfaces": n_surfaces,
            "n_clusters_merged": len(merges),
            "n_surfaces_rewritten": sum(len(m["members"]) - 1 for m in merges),
            "merge_rate": round(sum(len(m["members"]) - 1 for m in merges) / n_surfaces, 4)
                          if n_surfaces else 0.0,
            "merge_signal_counts": dict(signal_counts),
            "cluster_size_hist": {str(k): v for k, v in sorted(sizes.items())},
            "largest_cluster": max((len(m["members"]) for m in merges), default=1),
            "type_consistency": {
                "_caveat": "SMELL TEST ONLY — conflates IC over-merge with EC type noise on "
                           "identical strings (fold-equal clusters can disagree on type). Do not "
                           "quote as a quality number; human-review sample_merges instead.",
                "n_clusters_with_types": len(typed_clusters),
                "n_type_consistent": pure,
                "rate": round(pure / len(typed_clusters), 4) if typed_clusters else None,
            },
            "n_surfaces_with_type": sum(1 for s in type_map if type_map[s]),
            "sample_merges": sorted(merges, key=lambda m: -len(m["members"]))[:15],
            "params": dict(self.cfg),
        }


# ---------------------------------------------------------------- KG rewriting + IO

def rewrite_triplets_per_sample(triplets_per_sample, canon_map):
    """Apply a canon_map to per-sample [[h,r,t],...] lists. Dedups within a sample and drops
    self-loops after rewriting (h==t), matching the post-hoc aligner and upstream KG2RAG."""
    out_samples = []
    for trips in triplets_per_sample:
        seen, out = set(), []
        for triple in trips:
            h, r, t = triple[0], triple[1], triple[2]
            h2 = canon_map.get(str(h), str(h))
            t2 = canon_map.get(str(t), str(t))
            k = (h2, str(r), t2)
            if k not in seen and h2 != t2:
                seen.add(k)
                out.append([h2, str(r), t2])
        out_samples.append(out)
    return out_samples


def collect_surface_freq(triplets_by_case):
    """Union surface frequencies over every relation case.

    Surfaces are relation-case-invariant (h/t come from `input_triplet`), but a triplet whose
    relation failed to canonicalize is dropped per-case, so the union is the safe basis.
    """
    freq = Counter()
    for triplets in triplets_by_case.values():
        for triple in triplets:
            freq[str(triple[0])] += 1
            freq[str(triple[2])] += 1
    return freq


def collect_entity_descriptions(sd_dict_list):
    """{entity_surface: SD1 description} — first non-empty description wins."""
    desc = {}
    for sd in sd_dict_list or []:
        for item in (sd or {}).get("entities", []) or []:
            if not (isinstance(item, (list, tuple)) and len(item) == 4):
                continue
            ent_name, _coarse, _fine, d = item
            if d and str(ent_name) not in desc:
                desc[str(ent_name)] = str(d)
    return desc


def save_stats(stats, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[IC] {stats['ic_case']}: {stats['n_clusters_merged']} clusters merged "
          f"({stats['n_surfaces_rewritten']}/{stats['n_surfaces']} surfaces = "
          f"{stats['merge_rate']:.1%}, largest {stats['largest_cluster']}) -> {path}")
