"""
Per-ENTITY instance-of (P31) gold entity types (RESEARCH_QUESTIONS.md §8.2, item 12).

The current entity-canon metric (clustering_metric.py) grades an entity by the type its RELATION
SLOT implies (schema head_type / tail_type). That is coarse: an entity's gold type is inferred from
the role it plays in a triple, so the SAME surface can be scored against different gold types in
different triples, and a surface that is genuinely e.g. a Settlement is graded as whatever its slot
wants. This module builds the STRONGER reference that ENTITY_TYPE_SCHEMA.md §8 names as future work:
entity-link each gold SURFACE string to its KB entity and read its instance-of (P31) class, mapped
into our existing low-level schema taxonomy -> a per-entity gold type, independent of the relation.

  - webnlg (DBpedia): the gold surfaces ARE DBpedia resource localnames (Ciudad_Ayala, ALCO_RS-3), so
        <dbr:surface> a ?t (dbo classes) links with NO ambiguity. Most-specific dbo class -> schema type.
  - rebel / wiki-nre (Wikidata): surface is a label -> QID (label match; prefer an item that HAS a P31
        and is not a disambiguation page) -> wdt:P31 -> map into taxonomy via P279* walk. Label linking
        is noisy => rebel/wiki-nre are REVIEW-GATED (ambiguous surfaces flagged in _report.review).

Mapping KB instance-of -> our taxonomy keeps the SAME low-level label space as the slot-based gold
(schemas/*_schema.json entity_types), so the P31 metric is directly comparable: same labels, but the
type now comes from the entity's identity, not its slot. Clustering metrics are label-invariant, so
the exact label name does not matter for scoring; we still map to the taxonomy for granularity parity.

Output: schemas/gold_entity_p31_{dataset}.json
  { "_meta", "_report",
    <norm_surface>: { "surface", "kb_id", "kb_types":[..], "schema_type": <low-level type or null>,
                      "status": linked|literal|unlinked|no_p31|no_type_match|ambiguous } }
Raw KB responses are cached to schemas/.p31_cache_{dataset}.json (reviewable, makes re-runs cheap).

STAGES:
  python schemas/build_gold_entity_p31.py --dataset webnlg          # DBpedia, clean, ~345 surfaces
  python schemas/build_gold_entity_p31.py --dataset wiki-nre        # Wikidata, review-gated
  python schemas/build_gold_entity_p31.py --dataset rebel           # Wikidata, review-gated
Needs network + requests. Reuses the type->QID map (entity_type_wikidata_map.json) + DBO_CLASS from
build_gold_entity_types.py. DECISIONS: 2026-07-12 (§8.2 item 12).

CAVEAT (state in paper): this is a per-entity accuracy/metric refinement ONLY. It does NOT affect
recall (the benchmark gold KGs score each triple independently on surface strings; there is no entity
alignment in the gold to exploit) and is DISTINCT from entity-INSTANCE alignment (§8.5), the node-merge
lever that moves GraphRAG/multi-hop recall.
"""
import os, re, ast, json, time, argparse
from collections import Counter
import requests

# reuse the grounding infra (endpoints, type map, dbo class map, subclass walks, sparql helper)
from build_gold_entity_types import (
    sparql, chunks, dbo_class_of, is_dbo_class, dbpedia_subclasses, wd_superclasses,
    schema_types, TYPE_QID, DBO_CLASS, WDQS, DBPS, UA, TYPE_MAP_PATH, DBO_PREFIX,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REF = {ds: os.path.join(HERE, "..", "soda", "evaluate", "references", f"{ds}.txt")
       for ds in ("webnlg", "rebel", "wiki-nre")}
SOURCE = {"webnlg": "dbpedia", "rebel": "wikidata", "wiki-nre": "wikidata"}
DBR = "http://dbpedia.org/resource/"
DISAMBIG_QID = "Q4167410"   # Wikidata "disambiguation page" -> never a real entity type


# --------------------------------------------------------------------------- surfaces
def norm(x):
    """Same normalisation clustering_metric.norm uses, so the P31 gold keys match the metric's surface keys."""
    if not isinstance(x, str):
        x = str(x)
    x = x.strip().strip('"').strip("'")
    x = x.replace(",_", "_").replace(",", "_")
    x = re.sub(r"\s+", "_", x)
    x = re.sub(r"[^a-zA-Z0-9_]", "", x)
    return re.sub(r"_+", "_", x).strip("_").lower()


_NUM = re.compile(r"^[-+−]?[\d.,]+(\s*\(.*\))?$")   # 1777539 / 1604.0 / -6 / '17068.8 (millimetres)'


def is_literal(surf):
    """Values (numbers, quoted strings, dates) have no P31 -> Value, not scored as an entity type."""
    s = surf.strip()
    if not s:
        return True
    if s[0] in '"“' or s[-1] in '"”':   # quoted string literal
        return True
    if _NUM.match(s):
        return True
    return False


def load_surfaces(dataset):
    """Distinct raw gold surfaces + which slots (head/tail) each appears in (for a sanity report)."""
    surf, roles = {}, {}
    for line in open(REF[dataset], encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            trips = ast.literal_eval(line)
        except Exception:
            continue
        for t in trips:
            if len(t) < 3:
                continue
            for raw, role in ((str(t[0]), "head"), (str(t[2]), "tail")):
                s = raw.strip()
                surf.setdefault(s, s)
                roles.setdefault(s, set()).add(role)
    return surf, roles


def cache_load(dataset):
    p = os.path.join(HERE, f".p31_cache_{dataset}.json")
    return (json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}), p


# --------------------------------------------------------------------------- DBpedia (webnlg)
def dbpedia_resource_types(surfaces, cache):
    """resource localname -> list of dbo class localnames (rdf:type in the dbo namespace)."""
    todo = [s for s in surfaces if s not in cache]
    for ch in chunks(todo, 25):
        values = " ".join("<%s%s>" % (DBR, urlq(s)) for s in ch)
        q = (DBO_PREFIX + "SELECT ?s ?t WHERE { VALUES ?s { %s } ?s a ?t . "
             "FILTER(STRSTARTS(STR(?t),\"http://dbpedia.org/ontology/\")) }" % values)
        got = {}
        for b in sparql(DBPS, q):
            s = b["s"]["value"][len(DBR):]; t = b["t"]["value"].rsplit("/", 1)[-1]
            got.setdefault(s, []).append(t)
        # decode back: the VALUES IRIs are URL-encoded; match by decoded localname
        dec = {urlq(s): s for s in ch}
        for enc_s, ts in got.items():
            key = dec.get(enc_s, enc_s)
            cache[key] = sorted(set(ts))
        for s in ch:                       # entities with no dbo type -> record empty (linked-but-untyped)
            cache.setdefault(s, [])
        time.sleep(1)
    return cache


def urlq(s):
    import urllib.parse
    return urllib.parse.quote(s, safe="_,()'-.")


def dbpedia_redirects(surfaces):
    """resource localname -> its dbo:wikiPageRedirects target localname (for surfaces that don't type directly)."""
    out = {}
    for ch in chunks(list(surfaces), 25):
        values = " ".join("<%s%s>" % (DBR, urlq(s)) for s in ch)
        q = (DBO_PREFIX + "SELECT ?s ?r WHERE { VALUES ?s { %s } ?s dbo:wikiPageRedirects ?r . }" % values)
        dec = {urlq(s): s for s in ch}
        for b in sparql(DBPS, q):
            s = dec.get(b["s"]["value"][len(DBR):], b["s"]["value"][len(DBR):])
            out[s] = b["r"]["value"][len(DBR):]
        time.sleep(1)
    return out


def map_dbo_to_schema(dbo_types, schema_type_to_cls, hier):
    """Given an entity's dbo type set, return the MOST SPECIFIC schema low-level type that applies.
    A schema type T (dbo class C) applies if C is in the entity's dbo type set OR is an ancestor of one
    of them. Most specific = the applicable T whose class has the LONGEST subclass chain (deepest)."""
    if not dbo_types:
        return None
    ent = set(dbo_types)
    ent_anc = set()                        # entity types + all their ancestors
    for c in dbo_types:
        ent_anc |= hier.get(c, {c})
    applicable = [(T, C) for T, C in schema_type_to_cls.items() if C in ent_anc]
    if not applicable:
        return None
    # most specific = class with the most ancestors (deepest in the dbo tree)
    return max(applicable, key=lambda TC: (TC[1] in ent, len(hier.get(TC[1], {TC[1]}))))[0]


def build_dbpedia(dataset, surf, roles, cache):
    schema_type_to_cls = {}
    for h, t in schema_types(dataset).values():
        for T in (h, t):
            if T and dbo_class_of(T) and is_dbo_class(dbo_class_of(T)):
                schema_type_to_cls[T] = dbo_class_of(T)
    ents = [s for s in surf if not is_literal(s)]
    print(f"  {len(surf)} surfaces ({len(ents)} entity / {len(surf)-len(ents)} literal); querying DBpedia types...")
    dbpedia_resource_types(ents, cache)
    # redirect pass: resources that typed empty may be redirects (People's_Republic_of_China -> China);
    # follow dbo:wikiPageRedirects and attribute the target's dbo types back to the surface.
    empties = [s for s in ents if not cache.get(s)]
    redirect = {}
    if empties:
        redirect = dbpedia_redirects(empties)
        tgts = sorted(set(redirect.values()))
        print(f"  {len(empties)} untyped -> {len(redirect)} redirects; typing {len(tgts)} targets...")
        tgt_cache = {}
        dbpedia_resource_types(tgts, tgt_cache)
        for s, tgt in redirect.items():
            if tgt_cache.get(tgt):
                cache[s] = tgt_cache[tgt]
    # subclass hierarchy over every dbo class seen (entity types + schema classes)
    all_cls = set(schema_type_to_cls.values())
    for s in ents:
        all_cls |= set(cache.get(s, []))
    hier = dbpedia_subclasses(sorted(c for c in all_cls if is_dbo_class(c)))
    out = {}
    for s in surf:
        if is_literal(s):
            out[norm(s)] = {"surface": s, "kb_id": None, "kb_types": [], "schema_type": None,
                            "status": "literal"}
            continue
        types = cache.get(s, [])
        tgt = redirect.get(s)
        if not types:
            out[norm(s)] = {"surface": s, "kb_id": DBR + urlq(s), "kb_types": [], "schema_type": None,
                            "status": "unlinked"}   # resource has no dbo type (or does not resolve)
            continue
        T = map_dbo_to_schema(types, schema_type_to_cls, hier)
        rec = {"surface": s, "kb_id": DBR + urlq(tgt if tgt else s), "kb_types": types,
               "schema_type": T, "status": "linked" if T else "no_type_match"}
        if tgt:
            rec["via_redirect"] = tgt
        out[norm(s)] = rec
    return out


# --------------------------------------------------------------------------- Wikidata (rebel / wiki-nre)
def wd_label_to_qids(labels, cache):
    """label -> [QIDs] by exact rdfs:label (en). Cached. Noisy -> disambiguated downstream by P31."""
    todo = [l for l in labels if ("lab:" + l) not in cache]
    for ch in chunks(todo, 50):
        values = " ".join('"%s"@en' % l.replace('\\', '\\\\').replace('"', '\\"') for l in ch)
        q = ("SELECT ?label ?item WHERE { VALUES ?label { %s } ?item rdfs:label ?label . "
             "FILTER(STRSTARTS(STR(?item),\"http://www.wikidata.org/entity/Q\")) }" % values)
        got = {}
        for b in sparql(WDQS, q):
            lab = b["label"]["value"]; qid = b["item"]["value"].rsplit("/", 1)[-1]
            got.setdefault(lab, []).append(qid)
        for l in ch:
            cache["lab:" + l] = got.get(l, [])[:20]     # cap candidates
        time.sleep(1)
    return cache


def wd_altlabel_to_qids(labels, cache):
    """Fallback: exact skos:altLabel (en) match for surfaces that miss on rdfs:label (gold surfaces are
    often an ALIAS/short form, e.g. 'London School of Economics' vs the canonical '... and Political
    Science'). Only fills surfaces still empty; records src=altlabel."""
    todo = [l for l in labels if not cache.get("lab:" + l)]
    for ch in chunks(todo, 50):
        values = " ".join('"%s"@en' % l.replace('\\', '\\\\').replace('"', '\\"') for l in ch)
        q = ("SELECT ?label ?item WHERE { VALUES ?label { %s } ?item skos:altLabel ?label . "
             "FILTER(STRSTARTS(STR(?item),\"http://www.wikidata.org/entity/Q\")) }" % values)
        got = {}
        for b in sparql(WDQS, q):
            lab = b["label"]["value"]; qid = b["item"]["value"].rsplit("/", 1)[-1]
            got.setdefault(lab, []).append(qid)
        for l in ch:
            if got.get(l):
                cache["lab:" + l] = got[l][:20]; cache["src:" + l] = "altlabel"
        time.sleep(1)
    return cache


def wbsearch(label):
    """MediaWiki wbsearchentities fuzzy entity search -> [QIDs] (handles surfaces that are neither the
    canonical label nor a registered alias). Fuzzy => recorded src=search and re-checked via P31 mapping."""
    try:
        r = requests.get("https://www.wikidata.org/w/api.php",
                         params={"action": "wbsearchentities", "search": label, "language": "en",
                                 "uselang": "en", "format": "json", "limit": 5, "type": "item"},
                         headers={"User-Agent": UA}, timeout=30)
        if r.status_code == 200:
            return [h["id"] for h in r.json().get("search", [])]
    except Exception as e:
        print(f"    [wbsearch] {label!r} failed: {e}")
    return []


def wd_search_fallback(labels, cache):
    """Last-resort fuzzy entity search for surfaces still unlinked after rdfs:label + skos:altLabel."""
    todo = [l for l in labels if not cache.get("lab:" + l)]
    for i, s in enumerate(todo):
        qs = cache.get("srch:" + s)
        if qs is None:
            qs = wbsearch(s); cache["srch:" + s] = qs; time.sleep(0.3)
        if qs:
            cache["lab:" + s] = qs[:20]; cache["src:" + s] = "search"
        if (i + 1) % 100 == 0:
            print(f"    [wbsearch] {i+1}/{len(todo)}")
    return cache


def wd_p31(qids, cache):
    """QID -> [P31 QIDs]. Cached."""
    todo = [q for q in qids if ("p31:" + q) not in cache]
    for ch in chunks(todo, 60):
        values = " ".join("wd:%s" % q for q in ch)
        query = ("SELECT ?q ?c WHERE { VALUES ?q { %s } ?q wdt:P31 ?c . }" % values)
        got = {}
        for b in sparql(WDQS, query):
            qq = b["q"]["value"].rsplit("/", 1)[-1]; c = b["c"]["value"].rsplit("/", 1)[-1]
            got.setdefault(qq, []).append(c)
        for q in ch:
            cache["p31:" + q] = got.get(q, [])
        time.sleep(1)
    return cache


def build_wikidata(dataset, surf, roles, cache):
    qid_of_schema = {}
    tmap = {k: v for k, v in json.load(open(TYPE_MAP_PATH, encoding="utf-8")).items() if not k.startswith("_")}
    for h, t in schema_types(dataset).values():
        for T in (h, t):
            q = tmap.get(T, {}).get("qid") if T else None
            if q:
                qid_of_schema[T] = q
    ents = [s for s in surf if not is_literal(s)]
    print(f"  {len(surf)} surfaces ({len(ents)} entity); Wikidata label->QID...")
    wd_label_to_qids(ents, cache)
    for s in ents:                      # mark exact-label hits (fallbacks tag their own src)
        if cache.get("lab:" + s) and ("src:" + s) not in cache:
            cache["src:" + s] = "label"
    miss = [s for s in ents if not cache.get("lab:" + s)]
    if miss:
        print(f"  {len(miss)} exact-label misses -> skos:altLabel fallback...")
        wd_altlabel_to_qids(miss, cache)
    miss = [s for s in ents if not cache.get("lab:" + s)]
    if miss:
        print(f"  {len(miss)} still missing -> wbsearchentities fuzzy fallback...")
        wd_search_fallback(miss, cache)
    # candidate QIDs -> P31
    cand_qids = sorted({q for s in ents for q in cache.get("lab:" + s, [])})
    print(f"  {len(cand_qids)} candidate QIDs; fetching P31...")
    wd_p31(cand_qids, cache)
    # superclass walk over all P31 classes + schema type QIDs
    p31_all = {c for q in cand_qids for c in cache.get("p31:" + q, []) if c != DISAMBIG_QID}
    sup = wd_superclasses(sorted(p31_all | set(qid_of_schema.values())))
    schema_qids = set(qid_of_schema.values())
    def map_p31(p31s):
        # most specific schema type whose QID is an ancestor of a (non-disambig) P31 class
        best, best_depth = None, -1
        for c in p31s:
            if c == DISAMBIG_QID:
                continue
            anc = sup.get(c, {c})
            for T, q in qid_of_schema.items():
                if q in anc:
                    d = len(sup.get(q, {q}))   # deeper schema class = more specific
                    if d > best_depth:
                        best, best_depth = T, d
        return best
    out = {}
    for s in surf:
        if is_literal(s):
            out[norm(s)] = {"surface": s, "kb_id": None, "kb_types": [], "schema_type": None, "status": "literal"}
            continue
        qids = cache.get("lab:" + s, [])
        if not qids:
            out[norm(s)] = {"surface": s, "kb_id": None, "kb_types": [], "schema_type": None, "status": "unlinked"}
            continue
        # choose the candidate QID that yields a taxonomy-mappable, non-disambiguation P31
        chosen, chosen_p31, chosen_T = None, [], None
        for q in qids:
            p31 = [c for c in cache.get("p31:" + q, []) if c != DISAMBIG_QID]
            if not p31:
                continue
            T = map_p31(p31)
            if T:
                chosen, chosen_p31, chosen_T = q, p31, T
                break
            if chosen is None:                 # fall back to first with any real P31
                chosen, chosen_p31 = q, p31
        if chosen is None:
            out[norm(s)] = {"surface": s, "kb_id": qids[0], "kb_types": [], "schema_type": None,
                            "status": "no_p31"}
            continue
        status = "linked" if chosen_T else "no_type_match"
        if len(qids) > 1:
            status = status if chosen_T else "ambiguous"   # flag multi-candidate misses for review
        out[norm(s)] = {"surface": s, "kb_id": chosen, "kb_types": chosen_p31, "schema_type": chosen_T,
                        "status": status, "n_candidates": len(qids), "link_via": cache.get("src:" + s)}
    return out


# --------------------------------------------------------------------------- driver
def build(dataset):
    print(f"== build P31 gold: {dataset} ({SOURCE[dataset]}) ==")
    surf, roles = load_surfaces(dataset)
    cache, cache_path = cache_load(dataset)
    if SOURCE[dataset] == "dbpedia":
        out = build_dbpedia(dataset, surf, roles, cache)
    else:
        out = build_wikidata(dataset, surf, roles, cache)
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    by_status = Counter(v["status"] for v in out.values())
    linked = by_status.get("linked", 0)
    ent_total = sum(1 for v in out.values() if v["status"] != "literal")
    typed = Counter(v["schema_type"] for v in out.values() if v["schema_type"])
    review = {k: v for k, v in out.items() if v["status"] in ("ambiguous", "no_type_match", "no_p31")}
    link_via = Counter(v.get("link_via") for v in out.values() if v.get("link_via"))
    rep = {"dataset": dataset, "source": SOURCE[dataset],
           "n_surfaces": len(out), "n_entity": ent_total, "n_literal": by_status.get("literal", 0),
           "by_status": dict(by_status),
           "entity_coverage": round(linked / ent_total, 4) if ent_total else 0.0,
           "n_schema_types_used": len(typed), "type_hist": dict(typed.most_common()),
           "link_via": dict(link_via), "n_review": len(review)}
    outp = os.path.join(HERE, f"gold_entity_p31_{dataset}.json")
    json.dump({"_meta": {"dataset": dataset, "source": SOURCE[dataset],
                         "granularity": "per-entity instance-of (P31) -> low-level schema type"},
               "_report": rep, **out}, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  status: {dict(by_status)}")
    print(f"  entity coverage (linked/entity): {rep['entity_coverage']}  ({linked}/{ent_total})")
    print(f"  schema types used: {rep['n_schema_types_used']}  review: {rep['n_review']}")
    print(f"  -> {outp}  (+ cache {os.path.basename(cache_path)})")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webnlg", choices=["webnlg", "rebel", "wiki-nre", "all"])
    a = ap.parse_args()
    targets = ["webnlg", "rebel", "wiki-nre"] if a.dataset == "all" else [a.dataset]
    for ds in targets:
        build(ds)
