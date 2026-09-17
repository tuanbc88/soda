# Per-entity P31 gold — build + adjudication record

Audit trail for `schemas/gold_entity_p31_{webnlg,rebel,wiki-nre}.json` (builder
`build_gold_entity_p31.py`; method in `ENTITY_TYPE_SCHEMA.md` §10). "Adjudication" here is
pipeline-level, not per-entity hand-labelling: we validate the linker, spot-check quality, quantify
how much the identity-based P31 gold sharpens the slot-derived gold, and classify the residual
review buckets. Built 2026-07-12.

## Linking method (recap)
- **WebNLG (DBpedia).** Gold surfaces ARE DBpedia resource localnames → `<dbr:surface> a ?t` (dbo
  classes), unambiguous, + a `dbo:wikiPageRedirects` pass. Most-specific covered dbo class → schema type.
- **REBEL / Wiki-NRE (Wikidata).** Surface → QID by, in order: (1) exact `rdfs:label` (en);
  (2) exact `skos:altLabel` (en) fallback — gold surfaces are often an ALIAS/short form (e.g. *London
  School of Economics* vs the canonical *… and Political Science*); (3) `wbsearchentities` fuzzy search.
  Then `wdt:P31` → walk `P279*` → the most-specific schema type whose QID is an ancestor of a
  (non-disambiguation) P31 class. Each candidate is validated by whether its P31 maps into the taxonomy,
  which bounds fuzzy-search mislinks. `link_via` (label/altlabel/search) is recorded per surface.

## Coverage + sharpening (final, improved linker)

| dataset | source | entity surfaces | linked | coverage | link_via (label/alt/search) | schema types | P31-vs-slot agreement |
|---|---|--:|--:|--:|---|--:|--:|
| **webnlg** | DBpedia | 251 | 163 | **0.65** | direct-resource (+redirect) | 20 | **50.5%** |
| **wiki-nre** | Wikidata | 2,335 | 2,297 | **0.984** | 2227 / 35 / 35 | 17 | **72.4%** |
| **rebel** | Wikidata | 3,519 | 3,340 | **0.949** | 2644 / 640 / 115 | 53 | **45.9%** |

The Wikidata linker's fallbacks matter most on REBEL: exact-`rdfs:label` alone reached only 73.7%
(833 unlinked); `skos:altLabel` recovered 640 and `wbsearchentities` 115 → **94.9%** (72 unlinked).
On Wiki-NRE the same fallbacks lifted 95.4% → 98.4%.

**Reading:** the P31 gold disagrees with the slot-derived gold on **half of WebNLG occurrences**,
**~1/4 of Wiki-NRE**, and **>half of REBEL** — the slot gold labels an entity by the *role* it plays
(WebNLG's `Ciudad_Ayala` is scored as Organization / Country / Entity across triples), the P31 gold by
its *identity* (always `Settlement`). Wiki-NRE disagrees least (type-homogeneous, place/person heavy);
REBEL most (196 relations, the most varied argument types) → the sharpening is largest exactly where the
slot gold is coarsest.
Re-scoring the entity-canon against the P31 gold (WebNLG) gives higher coverage (0.73 vs 0.47 of
occurrences) and higher, better-separated B³ with the **ec2 mode-flip preserved** — i.e. the §12c
entity-canon finding is robust to the gold-type definition (`ENTITY_TYPE_SCHEMA.md` §10).

## Residual review buckets (not scored; reported honestly)
- **`unlinked`** — no KB entity resolved. WebNLG: resource has no `dbo:` type on the live endpoint
  (DBpedia-completeness limit). Wiki-NRE: 29 left after the 3-stage linker (rare surfaces). These are
  a coverage limit, not a wrong label.
- **`no_type_match`** — entity linked + has a P31, but the schema's argument **taxonomy has no slot**
  for it (e.g. Wiki-NRE stadium `Q1076486` / metro `Q5503`; WebNLG Sport / MilitaryStructure). Honest
  taxonomy gap, self-excluded from scoring.
- **`ambiguous`** — multiple label candidates and none mapped cleanly (e.g. Wiki-NRE *Hindi*, 5
  candidates). Flagged, not scored.
- **`no_p31` / `literal`** — item without an instance-of / a Value (number, quoted string, date).

**Verdict:** the WebNLG + Wiki-NRE P31 golds are validated and usable (spot-checks correct, including
the fuzzy-search recoveries, which the P31-mapping check guards). REBEL is the noisiest surface set →
see its row + `link_via`; treat search-linked REBEL entities as the review-priority subset if a number
looks off. No per-entity hand-fixes were applied; the pipeline + these buckets are the audit record.

## Reproduce
```
python schemas/build_gold_entity_p31.py --dataset webnlg     # DBpedia
python schemas/build_gold_entity_p31.py --dataset wiki-nre   # Wikidata
python schemas/build_gold_entity_p31.py --dataset rebel      # Wikidata
```
Network + `requests`. Raw KB responses cache to `schemas/.p31_cache_{ds}.json` (gitignored) so re-runs
only issue the new queries.
