# Typed entity-schema construction for the EDC benchmarks (methodology + provenance)

A reproducible, auditable account of how the **entity-type layer** of `schemas/{webnlg,rebel,wiki-nre}_schema.json`
was built. The original EDC benchmarks ship **gold relations only**; the typed entity schema is a value-added
annotation layer we contribute. This file documents the pipeline, the integrity guards, the artifacts, and the
epistemic status, so the contribution can be reviewed and reproduced. (Decision trail: `DECISIONS.md` 2026-06-25;
RQ ledger: `RESEARCH_QUESTIONS.md` §8.2.)

## 1. Motivation
The WebNLG / REBEL / Wiki-NRE schemas released with EDC contain, per relation, only a **name** and a
**general definition** (`schemas/*_schema.csv`). They have **no entity types**: the gold KGs are
`(entity, relation, entity)` surface-string triples with no type labels on the entities. Entity types are
needed (a) to drive **entity canonicalization** (assign each entity a canonical type) and (b) to **assess** it
(score the entity-canon stage against a reference). Because no entity-type gold exists, we construct one.

## 2. What the typed schema adds
For every relation we add a typed signature, and we add a shared entity-type taxonomy:

- **Per relation:** `head_type`, `tail_type` (the entity type each argument slot takes), plus the
  retained `name` + general definition + a `detailed_definition`.
- **Entity-type taxonomy** (`entity_types`): each type is **two-level** —
  - a **low-level type** (e.g. `Settlement`, `EducationalInstitution`, `MilitaryUnit`, `Typeface`),
  - a coarse **`parent`** from a fixed 8-class upper layer
    `{Agent, Place, Work, Artifact, Category, Event, Thing, Value}`,
  - a natural-language **`definition`**, and an `attributes` placeholder (for future attributed-KG work).

Current sizes: **webnlg 159 rel / 30 types · rebel 196 rel / 65 types · wiki-nre 45 rel / 20 types**
(74 distinct types across the three).

## 3. Construction pipeline (v0 → v4)
The layer was built in four versioned stages; intermediate snapshots are kept as evidence
(`schemas/webnlg_schema_159.json`, `_old.json`, `_pre_retype.json`).

| ver | stage | who/how | artifact |
|---|---|---|---|
| **v0** | source | EDC release: gold **relations only** (name + general def) | `schemas/*_schema.csv` |
| **v1** | **LLM type inference** | **ChatGPT-4** infers, per relation, `head_type`/`tail_type` and drafts the low-level type + coarse-parent + definition taxonomy | `webnlg_schema_159.json` |
| **v2** | **LLM retype / standardize** | **Claude Opus 4.8** re-infers and standardizes the typing across all three datasets (consistent parents, definitions, head/tail) | `_pre_retype.json` → live `*_schema.json`; generator `schemas/build_rebel_wikinre_schema.py` (relation names + general def **pinned verbatim** to the CSV so they can never drift; only the **typing** is the LLM part) |
| **v3** | **KB grounding + human review** | `schemas/build_gold_entity_types.py` grounds every typed slot against the **source KB** and flags inconsistencies; the author adjudicates the flags | `gold_entity_types_*.json`, `entity_type_wikidata_map.json`, `gold_entity_types_overrides.json`, `gold_entity_types_MISMATCH_review.md` |
| **v4** | **final grounded gold** | fixes + accepts applied; re-grounded | mismatch = 0 (100% of constrained slots consistent) |

webnlg's v1/v2 received **heavier human curation** than rebel/wiki-nre (hence "author-grade" in `DATASETS.md`);
all three are nonetheless **LLM-inferred** at base.

## 4. KB grounding (v3) in detail
The v1/v2 types are LLM-authored, so scoring entity-canon against them would be **circular**. v3 grounds each
relation slot's type against an **authoritative, citable** knowledge base, using the dataset's own provenance:

- **REBEL & Wiki-NRE → Wikidata.** Their relation names **are Wikidata property labels** (e.g. `place of birth`
  = P19). We resolve label → PID, then read the property's **type constraints** (`P2302`): the
  *subject-type constraint* (`Q21503250`) gives the allowed **head** classes, the *value-type constraint*
  (`Q21510865`) the allowed **tail** classes (qualifier `P2308`).
- **WebNLG → DBpedia.** Its relation names are DBpedia ontology properties (camelCase, e.g. `birthPlace`).
  We read `rdfs:domain` / `rdfs:range` of `dbo:<name>`.

**Type → KB-class resolution, with a hallucination guard.** Each schema entity-type is mapped to a Wikidata
QID (`entity_type_wikidata_map.json`) / DBpedia class. The QIDs are **LLM-proposed but machine-verified**: the
builder fetches each QID's English label and flags any whose label does not match the type. *This caught 5
hallucinated/wrong QIDs* (e.g. `MilitaryBranch` → a gene id `CPSG_01835-t26_1`; `TransportHub` → "restaurant";
`TransportLine` → "parachute cord"), which were corrected from verified search results. **LLMs hallucinate KB
identifiers; verification is mandatory, not optional.**

**Bidirectional consistency check.** A slot is scored by comparing its type `T` to the KB allowed-class set `S`
along the subclass hierarchy (`P279*` / `rdfs:subClassOf*`), in **both** directions:

| status | condition | meaning |
|---|---|---|
| `confirmed` | `T ∈ S` | exact KB match |
| `narrower` | `T` is a subclass of some class in `S` | schema is more specific (consistent) |
| `broader` | `T` is a superclass of some class in `S` | schema is more general (consistent) |
| `mismatch` | `T` disjoint from `S` | flagged for human review |
| `literal` / `unverified` | slot is a literal (Date/number) / KB has no constraint | not gradeable |

The both-direction check matters: without it, e.g. `Person ⊃ human` (a property whose constraint lists `human`)
would be a false mismatch. (Adding it dropped REBEL head mismatches from 40 to 6.)

**Consistency rates** (of slots that have a KB constraint), before human review:
**Wiki-NRE 98–100% · REBEL 92–95% · WebNLG 81–89%** — i.e. the LLM-authored types are mostly KB-consistent;
only a minority needed review.

## 5. Human adjudication (v3 → v4)
The 36 flagged mismatches were exported to `gold_entity_types_MISMATCH_review.md` with the schema type beside
the KB allowed classes. The author reviewed each (Decision + analysis + a rationale note).

**Rubric:** prefer the **low-level type that reflects the dataset's actual entity semantics** over a KB ontology
quirk; `fix` only genuine type errors, `accept` where the KB merely models the property differently/more narrowly.

- **15 `fix`** → corrected `head_type`/`tail_type` (e.g. `list of works` tail `Person` → `CreativeWork`;
  `ethnicGroup` tail `Organization` → `EthnicGroup`; WebNLG `course` head → `Food` — its gold subjects are
  dishes, e.g. *Bionico*, colliding with DBpedia's race-`course`).
- **9 new low-level types** were created to host fixes (REBEL: `SportsPosition`, `SportsDiscipline`,
  `HeritageDesignation`, `Typeface`; WebNLG: `Vehicle`, `EthnicGroup`, `Engine`, `Airport`, `Food`), each with
  parent + definition + a verified QID.
- **21 `accept`** → recorded in `gold_entity_types_overrides.json`, which forces those slots to status
  `accepted` (counted KB-consistent). Examples: `head of state` head `Country` (KB says `state`); `creator` tail
  `Person` (KB says the broader `Agent`).

After applying: re-grounding yields **mismatch = 0** on all three datasets (100% of constrained slots
confirmed/narrower/broader/accepted).

## 6. Artifacts & reproducibility
| file | role |
|---|---|
| `schemas/build_rebel_wikinre_schema.py` | v1/v2 generator (names pinned to CSV; typing is the LLM part) |
| `schemas/{ds}_schema.json` | the typed schema (v4) |
| `schemas/build_gold_entity_types.py` | v3 grounder (`--stage types` builds+verifies the QID map; `--stage ground` grounds each slot). Uses live Wikidata/DBpedia SPARQL; batched |
| `schemas/entity_type_wikidata_map.json` | type → QID + verified label (the reusable, auditable map) |
| `schemas/gold_entity_types_{ds}.json` | per-slot grounded status + KB allowed classes |
| `schemas/gold_entity_types_overrides.json` | the 21 author accepts (adjudication record) |
| `schemas/gold_entity_types_MISMATCH_review.md` | the 36-row review with decisions + rationale (audit trail) |
| `schemas/build_gold_entity_p31.py` | **per-entity P31 gold** builder (§10; surface → instance-of → taxonomy) |
| `schemas/gold_entity_p31_{ds}.json` | per-entity P31 gold (all 3 built: webnlg 0.65 / wiki-nre 0.98 / rebel 0.95 coverage) |
| `schemas/gold_entity_p31_review.md` | P31 build + adjudication record (coverage, sharpening, review buckets) |

Rebuild: `python schemas/build_gold_entity_types.py --stage types` then `--stage ground --dataset all`
(needs network + `requests`). SPARQL endpoints: `query.wikidata.org`, `dbpedia.org/sparql`.

## 7. Use in evaluation
The grounded gold feeds the entity-canonicalization metric: `soda/evaluate/clustering_metric.py` scores the
entity-canon (clustering) against the schema types, and additionally over only the **KB-confirmed/accepted**
slots (`clustering_entity_grounded.csv`, with a `grounded_coverage`). On WebNLG, grounding lifts entity-canon
B³ ~+0.13 and sharpens the mode-dependent signal flip (see `RESULTS_webnlg_matrix.md` §12c).

## 8. Epistemic status & limitations
- The entity-type layer is **LLM-inferred, KB-grounded, and human-verified** — a value-added annotation layer,
  **not** original gold. We state this plainly.
- Grounding is **per-slot domain/range** (a relation's allowed argument types), an *authoritative-reference*
  signal, **not** per-entity ground truth. A stronger reference for *this TYPE metric* entity-links each gold
  surface to its `P31` (instance-of) class (a per-entity type gold). **BUILT for WebNLG** (2026-07-12,
  `schemas/build_gold_entity_p31.py` → `gold_entity_p31_webnlg.json`); see §10. This is a
  **metric/accuracy refinement only — it does NOT affect recall**, because the benchmark gold KGs score
  each triple independently with surface-form entities (no entity alignment in the gold to exploit). It is
  **distinct from entity-INSTANCE alignment** (`RESEARCH_QUESTIONS.md` §8.5), which merges coreferent nodes and
  is the only entity lever that moves GraphRAG / multi-hop recall (on the GraphRAG-scope corpora, not these
  gold-triple benchmarks).
- **Coverage:** only the slots with a KB constraint are validated (~47% of WebNLG entity occurrences; the rest
  are literal slots or properties with no constraint). Reported honestly, not hidden.
- **Single annotator** (the author) performed the human review; a second annotator + agreement would strengthen
  it (planned, mirrors the eduhcmut retrieval-gold SV-2 plan).

## 9. Novelty / related resources (why this is a contribution)
To our knowledge, **no prior work releases a per-relation entity-type schema for the WebNLG / REBEL / Wiki-NRE
relation-extraction & canonicalization benchmarks.** Calibrated against what exists:
- **WebNLG** ships 16 data-level **DBpedia categories** (Airport, Astronaut, Athlete, Politician, ...) attached
  to each data-text pair — these are *domains of the example*, not **per-relation head/tail argument types**.
  WebNLG covers ~450 DBpedia properties but does not release a typed relation schema.
- The **open-KG canonicalization line** (CESI / ReVerb45K, COMBO, and EDC itself) canonicalizes noun/relation
  phrases using **entity linking and embeddings as side information**; none release an entity-**type** schema
  with a coarse taxonomy for these datasets.
- **EDC** (our baseline) provides **relations only** (name + general definition).
- The underlying entities *do* carry **source-KB types** (WebNLG entities are DBpedia resources; REBEL/Wiki-NRE
  are Wikidata-derived). So the novelty is **not** that entity types exist somewhere, but that they are
  **packaged as a grounded, adjudicated, per-relation typed schema for the canonicalization task** — which SODA
  both *requires* (head/tail types are canonicalization signals) and *assesses* (the entity-canon metric).

Honest scope of the claim: *first released per-relation typed schema for these benchmarks*, not *first entity
typing of these entities*. (Searched 2026-06-25: EDC, REBEL, GenIE, CESI/COMBO/ReVerb45K, WebNLG data cards.)

## 10. Per-entity P31 gold (the sharper reference tier — WebNLG built 2026-07-12)
The §4 grounding types a relation *slot*; an entity is then graded by whatever type the slot it sits in wants.
That is coarse: the SAME surface is scored against different gold types in different triples, and its gold type
reflects its **role**, not its **identity**. `schemas/build_gold_entity_p31.py` builds the stronger reference —
each distinct gold **surface** is entity-linked to its KB entity and its **instance-of** class is mapped into the
same low-level taxonomy, giving one identity-based type per surface.

- **WebNLG (DBpedia).** The gold surfaces *are* DBpedia resource localnames (`Ciudad_Ayala`, `ALCO_RS-3`), so
  `<dbr:surface> a ?t` (dbo classes) links with **no ambiguity**; a `dbo:wikiPageRedirects` pass recovers
  redirects (`People's_Republic_of_China`→`China`). The entity's **most-specific** dbo class that a schema type
  covers becomes its type. Literals (numbers, quoted strings) → `literal` (a Value, not typed).
- **REBEL / Wiki-NRE (Wikidata).** Surface→QID by exact `rdfs:label`, then two fallbacks because gold
  surfaces are often an alias/short form: exact `skos:altLabel`, then `wbsearchentities` fuzzy search. Each
  candidate is validated by whether its P31 maps into the taxonomy (bounds fuzzy mislinks); `link_via` is
  recorded per surface, and residual `ambiguous`/`no_type_match`/`unlinked` are flagged for review.

**Built for all three (2026-07-12; full audit + per-bucket classification in `gold_entity_p31_review.md`):**

| dataset | linked | coverage | link_via (label/alt/search) | types | P31-vs-slot disagreement |
|---|--:|--:|---|--:|--:|
| webnlg (DBpedia) | 163/251 | 0.65 | direct-resource (+redirect) | 20 | 50% |
| wiki-nre (Wikidata) | 2297/2335 | 0.98 | 2227/35/35 | 17 | 28% |
| rebel (Wikidata) | 3340/3519 | 0.95 | 2644/640/115 | 53 | 54% |

**The P31 gold disagrees with the slot-derived gold on ~half of WebNLG/REBEL occurrences** (Wiki-NRE ~1/4)
— e.g. `Ciudad_Ayala` is always a `Settlement`, but the slot gold calls it Organization / Entity / Country
depending on the relation; the sharpening is largest where the slot gold is coarsest (REBEL's 196 varied
relations). Re-scoring the entity-canon against it: **higher coverage** (WebNLG 0.73 vs 0.47 of occurrences;
REBEL 0.92 vs 0.63) **and higher, better-separated B³**, with the **mode-dependent flip preserved**
(self-canon → ec2 name+definition wins) — so the §12c entity-canon story is **robust to the gold-type
definition**. WebNLG's 65% is a live-DBpedia completeness limit (79 resources have no `dbo:` type);
the Wikidata sets reach 95–98% after the alias/search fallbacks (`rdfs:label`-only was 74% on REBEL).

**Wired into eval:** `clustering_metric.py run_entity` emits a third variant `clustering_entity_p31.csv`
(alongside all-slots + grounded) whenever `gold_entity_p31_{ds}.json` exists — default-safe (absent file →
variant skipped); `aggregate_eval_xlsx.py` adds an `Entity_p31` sheet. Reproduce:
`python schemas/build_gold_entity_p31.py --dataset {webnlg,wiki-nre,rebel}` (network + `requests`; caches raw
KB responses to `schemas/.p31_cache_{ds}.json`).
