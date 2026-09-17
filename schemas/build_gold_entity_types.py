"""
Ground the (Claude-authored) low-level entity-types against the source KB (RESEARCH_QUESTIONS.md §8.2).

We do NOT discard the low-level entity_types taxonomy (Settlement/Country/EducationalInstitution/... each
with a `parent`) -- the author deliberately chose fine-grained types. We make them claimable as "gold" by
GROUNDING each relation slot's schema head/tail type against the source KB's domain/range constraint:

  - rebel / wiki-nre : relation name = Wikidata property label.
        label -> PID -> type constraints (P2302): subject-type (wd:Q21503250) -> allowed SUBJECT classes,
        value-type (wd:Q21510865) -> allowed OBJECT classes; qualifier P2308 = the allowed class QIDs.
  - webnlg : relation name = DBpedia ontology property (camelCase).
        dbo:<name> rdfs:domain / rdfs:range.

Grounding per relation slot, given the schema type T (-> QID/dbo-class) and the KB allowed-class set S:
  confirmed : T (or a P279* superclass of T) is in S, OR T == range/domain class      -> KB backs the type
  narrower  : T is a strict subclass of an allowed class (schema is MORE specific)     -> consistent, finer
  broader   : T is a strict superclass of an allowed class (schema is MORE general)    -> consistent, coarser
  mismatch  : T disjoint from S                                                        -> FLAG for author
  unverified: KB has no constraint / property not found                               -> keep T, not grounded

Output (per dataset): schemas/gold_entity_types_{dataset}.json
  { "_meta", "_report", relation: { schema_head, schema_tail, prop, head_status, tail_status,
                                     head_allowed:[..], tail_allowed:[..] } }
Plus the reusable, REVIEWABLE artifact schemas/entity_type_wikidata_map.json (type -> QID + verified label).

STAGES (run independently; later stages need earlier outputs):
  python schemas/build_gold_entity_types.py --stage types            # build+verify the 67 type->QID map
  python schemas/build_gold_entity_types.py --stage ground --dataset all
Needs network + requests. Wikidata SPARQL is batched (chunked VALUES). DECISIONS 2026-06-25.

CAVEAT (state in paper): per-slot, authoritative-REFERENCE grounding, not a per-entity ground truth.
Per-entity P31 (instance-of) entity linking is the stronger future tier (deferred).
"""
import os, re, csv, json, time, argparse, urllib.parse
from collections import Counter
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = {"webnlg": os.path.join(HERE, "webnlg_schema.csv"),
       "rebel":  os.path.join(HERE, "rebel_schema.csv"),
       "wiki-nre": os.path.join(HERE, "wiki-nre_schema.csv")}
SCHEMA = {ds: os.path.join(HERE, f"{ds}_schema.json") for ds in CSV}
SCHEMA["webnlg_full"] = os.path.join(HERE, "webnlg_full_schema.json")   # scale set (no CSV; rels from schema)
SOURCE = {"webnlg": "dbpedia", "rebel": "wikidata", "wiki-nre": "wikidata", "webnlg_full": "dbpedia"}
TYPE_MAP_PATH = os.path.join(HERE, "entity_type_wikidata_map.json")
OVERRIDES_PATH = os.path.join(HERE, "gold_entity_types_overrides.json")

WDQS = "https://query.wikidata.org/sparql"
DBPS = "https://dbpedia.org/sparql"
UA = "SODA-gold-entity-types/1.0 (research; contact tuanbc88@hcmut.edu.vn, tuanbc88@gmail.com)"
SUBJ_CONSTRAINT = "Q21503250"   # subject-type constraint  -> head / domain
VAL_CONSTRAINT  = "Q21510865"   # value-type   constraint  -> tail / range

# Curated schema-type -> Wikidata QID (Claude-proposed; the script VERIFIES each by fetching its English
# label so a wrong QID is caught, not trusted). None -> resolve by label search + flag for author review.
# Covers the 67 distinct types across the three schemas.
TYPE_QID = {
    "Person": "Q215627", "Organization": "Q43229", "Company": "Q783794",
    "EducationalInstitution": "Q2385804", "Location": "Q2221906", "Country": "Q6256",
    "AdministrativeTerritory": "Q56061", "Settlement": "Q486972", "Continent": "Q5107",
    "BodyOfWater": "Q15324", "Watercourse": "Q355304", "MountainRange": "Q46831",
    "TerrainFeature": "Q271669", "HistoricalRegion": "Q1620908", "Building": "Q41176",
    "Venue": "Q17350442", "TransportHub": "Q2298537", "TransportLine": "Q728937",
    "CreativeWork": "Q17537576", "Book": "Q571", "Film": "Q11424", "MediaWork": "Q2431196",
    "MusicWork": "Q2188189", "TelevisionShow": "Q15416", "VideoGame": "Q7889",
    "Series": "Q5398426", "Franchise": "Q196600", "Genre": "Q483394", "Award": "Q618779",
    "Event": "Q1656682", "Competition": "Q476300", "Conflict": "Q180684", "Election": "Q40231",
    "SportsSeason": "Q27020041", "Language": "Q34770", "WritingSystem": "Q8192",
    "EthnicGroup": "Q41710", "Religion": "Q9174", "ReligiousOrder": "Q2061186",
    "Movement": "Q2198855", "PoliticalParty": "Q7278", "MilitaryUnit": "Q176799",
    "MilitaryBranch": "Q781132", "MilitaryRank": "Q56019", "Occupation": "Q12737077",
    "Position": "Q4164871", "Title": "Q216353", "VoiceType": "Q1063547",
    "Sport": "Q349", "SportsTeam": "Q12973014", "SportsLeague": "Q623109",
    "Family": "Q8436", "Taxon": "Q16521", "Drug": "Q8386", "MedicalCondition": "Q12136",
    "Instrument": "Q34379", "Currency": "Q8142", "Product": "Q2424752", "Vehicle": "Q42889",
    "Field": "Q11862829", "Industry": "Q8148", "TimeZone": "Q12143", "Class": "Q16889133",
    # new low-level types added 2026-06-25 from author mismatch adjudication (DECISIONS 2026-06-25)
    "SportsPosition": "Q1781513", "SportsDiscipline": "Q2312410", "HeritageDesignation": "Q17504995",
    "Typeface": "Q17451", "Engine": "Q44167", "Airport": "Q1248784", "Food": "Q2095",
    # literal / non-entity slots -> not entity types (skipped in grounding)
    "Date": None, "NumericValue": None, "TextValue": None, "Entity": None,
}


# schema type -> DBpedia ontology class localname (webnlg grounding). Default = identity attempt.
# Literals -> None (skipped). DBpedia uses British spelling / a slightly different backbone than our names.
DBO_CLASS = {
    "Organization": "Organisation", "CreativeWork": "Work", "MediaWork": "Work",
    "MusicWork": "MusicalWork", "Location": "Place", "AdministrativeTerritory": "AdministrativeRegion",
    "BodyOfWater": "BodyOfWater", "Settlement": "Settlement", "Conflict": "MilitaryConflict",
    "MedicalCondition": "Disease", "EthnicGroup": "EthnicGroup", "Watercourse": "Stream",
    "TransportHub": "Station", "TransportLine": "RouteOfTransportation", "Venue": "ArchitecturalStructure",
    # new low-level types added 2026-06-25 (author mismatch adjudication)
    "Vehicle": "MeanOfTransportation", "Engine": "Engine", "Airport": "Airport", "Food": "Food",
    # scale-set (webnlg_full) new types -> dbo class where one exists; else identity (-> review)
    "Taxon": "Species", "CelestialBody": "CelestialBody", "MilitaryUnit": "MilitaryUnit", "Sport": "Sport",
    "Date": None, "NumericValue": None, "TextValue": None, "Entity": None,
}


def dbo_class_of(schema_type):
    if schema_type in DBO_CLASS:
        return DBO_CLASS[schema_type]
    return schema_type   # identity attempt (most webnlg types mirror dbo class names)


def sparql(endpoint, query, retries=3):
    headers = {"User-Agent": UA, "Accept": "application/sparql-results+json"}
    r = None
    for attempt in range(retries):
        try:
            r = requests.get(endpoint, params={"query": query}, headers=headers, timeout=60)
            if r.status_code == 200:
                return r.json()["results"]["bindings"]
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"    [sparql] attempt {attempt+1} failed: {e}")
            time.sleep(2 * (attempt + 1))
    print(f"    [sparql] giving up (status {getattr(r,'status_code','?')})")
    return []


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def load_relations(dataset):
    if not (dataset in CSV and os.path.exists(CSV[dataset])):
        # no CSV (e.g. the scale set): take relations from the schema
        return list(json.load(open(SCHEMA[dataset], encoding="utf-8")).get("relation_types", {}))
    rels, seen = [], set()
    with open(CSV[dataset], encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0].strip() and row[0].strip() not in seen:
                seen.add(row[0].strip()); rels.append(row[0].strip())
    return rels


def schema_types(dataset):
    """relation -> (head_type, tail_type) from the (Claude-authored) schema JSON."""
    s = json.load(open(SCHEMA[dataset], encoding="utf-8"))
    rt = s.get("relation_types", {})
    out = {}
    for rel, info in rt.items():
        if isinstance(info, dict):
            out[rel] = (info.get("head_type"), info.get("tail_type"))
    return out


def all_schema_types():
    u = set()
    for ds in CSV:
        for h, t in schema_types(ds).values():
            u.add(h); u.add(t)
    u.discard(None)
    return sorted(u)


# ===================================================================== STAGE: types
def verify_qid_labels(qids):
    """QID -> English label (to sanity-check the curated map)."""
    out = {}
    for ch in chunks(qids, 80):
        values = " ".join("wd:%s" % q for q in ch)
        q = ("SELECT ?item ?itemLabel WHERE { VALUES ?item { %s } "
             "SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\". } }" % values)
        for b in sparql(WDQS, q):
            out[b["item"]["value"].rsplit("/", 1)[-1]] = b.get("itemLabel", {}).get("value", "")
        time.sleep(1)
    return out


def label_search_pid(types_without_qid):
    """Fallback: exact rdfs:label search for items (classes) named like the schema type."""
    out = {t: [] for t in types_without_qid}
    for ch in chunks(types_without_qid, 50):
        values = " ".join('"%s"@en' % t for t in ch)
        q = ("SELECT ?label ?item WHERE { VALUES ?label { %s } ?item rdfs:label ?label . "
             "?item wdt:P279 [] . }" % values)   # require it be a class (has a superclass)
        for b in sparql(WDQS, q):
            lab = b["label"]["value"]; it = b["item"]["value"].rsplit("/", 1)[-1]
            if lab in out and it not in out[lab]:
                out[lab].append(it)
        time.sleep(1)
    return out


def build_type_map():
    types = all_schema_types()
    curated = {t: TYPE_QID.get(t) for t in types}
    qids = sorted({q for q in curated.values() if q})
    print(f"  verifying {len(qids)} curated QIDs (label check) ...")
    labels = verify_qid_labels(qids)
    missing = [t for t in types if not curated.get(t)]
    print(f"  label-searching {len(missing)} unmapped types: {missing}")
    searched = label_search_pid(missing) if missing else {}
    out = {}
    for t in types:
        qid = curated.get(t)
        if qid:
            out[t] = {"qid": qid, "wd_label": labels.get(qid, ""),
                      "status": "curated_verified" if labels.get(qid) else "curated_unverified"}
        else:
            cand = searched.get(t, [])
            out[t] = {"qid": cand[0] if cand else None, "candidates": cand or None,
                      "wd_label": "", "status": "search_hit" if cand else "UNRESOLVED"}
    rep = {"n_types": len(types), "by_status": dict(Counter(v["status"] for v in out.values())),
           "review": {t: v for t, v in out.items()
                      if v["status"] in ("UNRESOLVED", "search_hit", "curated_unverified")}}
    json.dump({"_report": rep, **out}, open(TYPE_MAP_PATH, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  -> {TYPE_MAP_PATH}")
    print(f"  status: {rep['by_status']}")
    if rep["review"]:
        print("  NEEDS REVIEW:")
        for t, v in rep["review"].items():
            print(f"    {t:28s} qid={v.get('qid')}  wd_label='{v.get('wd_label')}'  ({v['status']})")
    return rep


# ===================================================================== STAGE: ground
def wd_labels_to_pids(labels):
    out = {l: [] for l in labels}
    for ch in chunks(labels, 60):
        values = " ".join('"%s"@en' % l.replace('"', '\\"') for l in ch)
        q = ("SELECT ?label ?p WHERE { VALUES ?label { %s } "
             "?p a wikibase:Property ; rdfs:label ?label . }" % values)
        for b in sparql(WDQS, q):
            lab = b["label"]["value"]; pid = b["p"]["value"].rsplit("/", 1)[-1]
            if lab in out and pid not in out[lab]:
                out[lab].append(pid)
        time.sleep(1)
    # alias fallback for misses
    miss = [l for l in labels if not out[l]]
    for ch in chunks(miss, 60):
        values = " ".join('"%s"@en' % l.replace('"', '\\"') for l in ch)
        q = ("SELECT ?label ?p WHERE { VALUES ?label { %s } "
             "?p a wikibase:Property ; skos:altLabel ?label . }" % values)
        for b in sparql(WDQS, q):
            lab = b["label"]["value"]; pid = b["p"]["value"].rsplit("/", 1)[-1]
            if lab in out and pid not in out[lab]:
                out[lab].append(pid)
        time.sleep(1)
    return out


def wd_pid_constraints(pids):
    out = {p: {"head": {}, "tail": {}} for p in pids}
    for ch in chunks(pids, 60):
        values = " ".join("wd:%s" % p for p in ch)
        q = ("SELECT ?p ?ct ?cls ?clsLabel WHERE { VALUES ?p { %s } "
             "?p p:P2302 ?s . ?s ps:P2302 ?ct . VALUES ?ct { wd:%s wd:%s } ?s pq:P2308 ?cls . "
             "SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\". } }"
             % (values, SUBJ_CONSTRAINT, VAL_CONSTRAINT))
        for b in sparql(WDQS, q):
            p = b["p"]["value"].rsplit("/", 1)[-1]; ct = b["ct"]["value"].rsplit("/", 1)[-1]
            cls = b["cls"]["value"].rsplit("/", 1)[-1]; lab = b.get("clsLabel", {}).get("value", cls)
            if p in out:
                out[p]["head" if ct == SUBJ_CONSTRAINT else "tail"][cls] = lab
        time.sleep(1)
    return out


def wd_superclasses(qids):
    """QID -> set of its P279* superclass QIDs (incl. self)."""
    out = {q: {q} for q in qids}
    for ch in chunks(qids, 40):
        values = " ".join("wd:%s" % q for q in ch)
        q = ("SELECT ?q ?sup WHERE { VALUES ?q { %s } ?q wdt:P279* ?sup . }" % values)
        for b in sparql(WDQS, q):
            qq = b["q"]["value"].rsplit("/", 1)[-1]; sup = b["sup"]["value"].rsplit("/", 1)[-1]
            out.setdefault(qq, {qq}).add(sup)
        time.sleep(1)
    return out


def status_for(schema_qid, schema_sup, allowed, allowed_sup):
    """Ground one slot: schema type QID vs the KB allowed-class set (both directions on P279*)."""
    if not allowed:
        return "unverified"
    A = set(allowed)
    if schema_qid in A:
        return "confirmed"
    if schema_sup & A:        # schema type is a SUBCLASS of an allowed class -> consistent, finer
        return "narrower"
    if schema_qid in allowed_sup:  # schema type is a SUPERCLASS of an allowed class -> consistent, coarser
        return "broader"
    return "mismatch"         # disjoint -> genuine flag for author


def ground_wikidata(dataset, tmap):
    rels = load_relations(dataset)
    st = schema_types(dataset)
    lab2pid = wd_labels_to_pids(rels)
    pids = sorted({p for ps in lab2pid.values() for p in ps})
    cons = wd_pid_constraints(pids)
    sqids = {tmap.get(t, {}).get("qid") for ht in st.values() for t in ht if t} - {None}
    allowed_qids = set()
    for p in pids:
        allowed_qids |= set(cons.get(p, {}).get("head", {})) | set(cons.get(p, {}).get("tail", {}))
    sup = wd_superclasses(sorted(sqids | allowed_qids))   # P279* for schema types AND allowed classes
    result = {}
    for rel in rels:
        h_t, t_t = st.get(rel, (None, None))
        pids_r = lab2pid.get(rel, [])
        with_con = [p for p in pids_r if cons.get(p, {}).get("head") or cons.get(p, {}).get("tail")]
        chosen = (with_con or pids_r or [None])[0]
        c = cons.get(chosen, {"head": {}, "tail": {}})
        h_qid = tmap.get(h_t, {}).get("qid"); t_qid = tmap.get(t_t, {}).get("qid")
        h_asup = set().union(*[sup.get(a, {a}) for a in c["head"]]) if c["head"] else set()
        t_asup = set().union(*[sup.get(a, {a}) for a in c["tail"]]) if c["tail"] else set()
        result[rel] = {
            "schema_head": h_t, "schema_tail": t_t, "prop": chosen,
            "head_allowed": c["head"], "tail_allowed": c["tail"],
            "head_status": status_for(h_qid, sup.get(h_qid, {h_qid}), c["head"], h_asup) if h_qid else "literal_or_unmapped",
            "tail_status": status_for(t_qid, sup.get(t_qid, {t_qid}), c["tail"], t_asup) if t_qid else "literal_or_unmapped",
        }
    return result


DBO_PREFIX = ("PREFIX dbo: <http://dbpedia.org/ontology/> "
              "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> ")


def is_dbo_class(c):
    """A real dbo ontology class localname (CamelCase id), not a datatype like 'XMLSchema#date'."""
    return bool(c) and bool(re.match(r"^[A-Za-z][A-Za-z0-9_]*$", c))


def dbpedia_subclasses(classes):
    """dbo class localname -> set of its rdfs:subClassOf* superclass localnames (incl. self)."""
    classes = [c for c in classes if is_dbo_class(c)]
    out = {c: {c} for c in classes}
    for ch in chunks(classes, 30):
        # full IRIs (datatype/# names already filtered) so the VALUES list never breaks the query
        values = " ".join("<http://dbpedia.org/ontology/%s>" % c for c in ch)
        q = (DBO_PREFIX + "SELECT ?c ?sup WHERE { VALUES ?c { %s } ?c rdfs:subClassOf* ?sup . }" % values)
        for b in sparql(DBPS, q):
            c = b["c"]["value"].rsplit("/", 1)[-1]; sup = b["sup"]["value"].rsplit("/", 1)[-1]
            out.setdefault(c, {c}).add(sup)
        time.sleep(1)
    return out


def ground_dbpedia(dataset, tmap):
    """webnlg: relation = DBpedia ontology property; ground schema type (-> dbo class) vs dbo domain/range
    using the dbo subclass hierarchy (same confirmed/narrower/broader/mismatch logic as Wikidata)."""
    rels = load_relations(dataset)
    st = schema_types(dataset)
    dr = {n: {"domain": None, "range": None} for n in rels}
    for ch in chunks(rels, 25):   # small chunks: DBpedia Virtuoso 400s on large VALUES+OPTIONAL
        # full IRI (not dbo:<name>): some webnlg names contain '/' which breaks prefixed-name syntax
        values = " ".join("<http://dbpedia.org/ontology/%s>" % n for n in ch)
        q = (DBO_PREFIX + "SELECT ?p ?domain ?range WHERE { VALUES ?p { %s } "
             "OPTIONAL { ?p rdfs:domain ?domain . } OPTIONAL { ?p rdfs:range ?range . } }" % values)
        for b in sparql(DBPS, q):
            p = b["p"]["value"].rsplit("/", 1)[-1]
            if p in dr:
                if b.get("domain"): dr[p]["domain"] = b["domain"]["value"].rsplit("/", 1)[-1]
                if b.get("range"):  dr[p]["range"] = b["range"]["value"].rsplit("/", 1)[-1]
        time.sleep(1)
    # gather all dbo classes (mapped schema types + domains/ranges); filter out datatypes (XMLSchema#...)
    schema_cls = {dbo_class_of(t) for ht in st.values() for t in ht if t} - {None}
    dr_cls = {d["domain"] for d in dr.values()} | {d["range"] for d in dr.values()}
    hier = dbpedia_subclasses(sorted({c for c in (schema_cls | dr_cls) if is_dbo_class(c)}))
    result = {}
    for rel in rels:
        h_t, t_t = st.get(rel, (None, None))
        d = dr.get(rel, {"domain": None, "range": None})
        h_c = dbo_class_of(h_t); t_c = dbo_class_of(t_t)
        def slot(schema_cls_name, kb_cls):
            if not kb_cls or not is_dbo_class(kb_cls):
                return "unverified" if not kb_cls else "literal_or_unmapped"  # datatype range -> literal
            if not schema_cls_name:
                return "literal_or_unmapped"
            return status_for(schema_cls_name, hier.get(schema_cls_name, {schema_cls_name}),
                              {kb_cls: kb_cls}, hier.get(kb_cls, {kb_cls}))
        result[rel] = {
            "schema_head": h_t, "schema_tail": t_t, "prop": "dbo:%s" % rel,
            "head_allowed": {d["domain"]: d["domain"]} if d["domain"] else {},
            "tail_allowed": {d["range"]: d["range"]} if d["range"] else {},
            "head_status": slot(h_c, d["domain"]),
            "tail_status": slot(t_c, d["range"]),
        }
    return result


def load_overrides(dataset):
    """Author adjudication: {relation: {head/tail: 'accepted'}} -> force that slot status."""
    if not os.path.exists(OVERRIDES_PATH):
        return {}
    d = json.load(open(OVERRIDES_PATH, encoding="utf-8"))
    ov = dict(d.get(dataset, {}))
    if dataset == "webnlg_full":   # scale set inherits the sample's adjudicated accepts (same relations)
        for rel, slots in d.get("webnlg", {}).items():
            ov.setdefault(rel, slots)
    return ov


def ground(dataset, tmap):
    print(f"== ground {dataset} ({SOURCE[dataset]}) ==")
    result = ground_dbpedia(dataset, tmap) if SOURCE[dataset] == "dbpedia" else ground_wikidata(dataset, tmap)
    # apply author adjudication (accepted mismatches -> 'accepted', counted KB-consistent downstream)
    applied = 0
    for rel, slots in load_overrides(dataset).items():
        if rel in result:
            for slot, st in slots.items():
                if f"{slot}_status" in result[rel]:
                    result[rel][f"{slot}_status"] = st; applied += 1
    if applied:
        print(f"  applied {applied} author overrides (accepted)")
    cnt = Counter()
    for v in result.values():
        cnt[("head", v["head_status"])] += 1; cnt[("tail", v["tail_status"])] += 1
    statuses = sorted({s for (_, s) in cnt})
    rep = {"n_relations": len(result),
           "head": {s: cnt[("head", s)] for s in statuses if cnt[("head", s)]},
           "tail": {s: cnt[("tail", s)] for s in statuses if cnt[("tail", s)]},
           "mismatches": sorted(r for r, v in result.items()
                                if "mismatch" in (v["head_status"], v["tail_status"]))}
    out = os.path.join(HERE, f"gold_entity_types_{dataset}.json")
    json.dump({"_meta": {"dataset": dataset, "source": SOURCE[dataset],
                         "granularity": "per-slot, schema low-level type grounded vs KB domain/range"},
               "_report": rep, **result}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  head: {rep['head']}\n  tail: {rep['tail']}\n  -> {out}")
    if rep["mismatches"]:
        print(f"  MISMATCH (review): {rep['mismatches'][:20]}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["types", "ground", "all"])
    ap.add_argument("--dataset", default="all", choices=["webnlg", "rebel", "wiki-nre", "webnlg_full", "all"])
    a = ap.parse_args()
    if a.stage in ("types", "all"):
        print("== build type->QID map ==")
        build_type_map()
    if a.stage in ("ground", "all"):
        if not os.path.exists(TYPE_MAP_PATH):
            raise SystemExit("type map missing -> run --stage types first")
        tmap = {k: v for k, v in json.load(open(TYPE_MAP_PATH, encoding="utf-8")).items()
                if not k.startswith("_")}
        targets = ["webnlg", "rebel", "wiki-nre"] if a.dataset == "all" else [a.dataset]
        for ds in targets:
            ground(ds, tmap)
