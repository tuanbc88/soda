"""
Generator for rebel_schema.json and wiki-nre_schema.json.

WHY THIS EXISTS (research-log discipline):
- The reference paper (clear-nus/edc) ships only relation NAME + a generic
  "subject/object" general_definition per dataset (the *.csv files here).
- This project extends each relation with: a richer detailed `definition`,
  typed `head_type`/`tail_type`, and a shared `entity_types` taxonomy, so the
  SC ablation cases (case3 detail / case4 detail+head-tail / case5 strict /
  entity-canon) have real signal.
- These extensions are LLM-AUTHORED (by Claude) and MUST BE REVIEWED by the
  author before any headline number is quoted. The mapping below is the audit
  trail: every head/tail/detail is visible and editable here, and the script
  re-verifies name+count parity against the CSV (and the gold reference) so a
  typo cannot silently desync Mode 3 (frozen schema) from the gold KG.

The authoritative relation NAMES + general_definition come from the CSV at run
time (not retyped here), so names can never drift from the reference paper.

Run:
    python schemas/build_rebel_wikinre_schema.py
"""

import csv
import json
import os
import ast

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# ---------------------------------------------------------------------------
# Shared entity-type taxonomy (7-parent system, same spirit as webnlg_schema).
# parent in: Agent / Place / Work / Artifact / Category / Event / Value / Thing
# ---------------------------------------------------------------------------
ENTITY_TYPES = {
    # --- Agent ---
    "Person": ("Agent", "A human individual."),
    "Organization": ("Agent", "An organized body such as an institution, association, or government entity."),
    "Company": ("Agent", "A commercial business organization."),
    "EducationalInstitution": ("Agent", "An organization that provides education, such as a school or university."),
    "SportsTeam": ("Agent", "An organized group of athletes competing together."),
    "SportsLeague": ("Agent", "An organization governing a sports competition between teams."),
    "PoliticalParty": ("Agent", "An organized political group seeking office or influence."),
    "MilitaryUnit": ("Agent", "An organized armed force, branch, or unit."),
    "MilitaryBranch": ("Agent", "A branch or division of armed forces (e.g. army, navy, air force)."),
    "Family": ("Agent", "A group of people related by kinship or lineage."),
    "ReligiousOrder": ("Agent", "An organized religious community or order."),
    # --- Place ---
    "Location": ("Place", "A geographic place or spatial area."),
    "Country": ("Place", "A sovereign nation state."),
    "Settlement": ("Place", "A populated place such as a city, town, or village."),
    "AdministrativeTerritory": ("Place", "A region defined by administrative boundaries."),
    "Continent": ("Place", "One of the principal large continuous land masses of the Earth."),
    "BodyOfWater": ("Place", "A significant accumulation of water such as a lake, sea, or ocean."),
    "Watercourse": ("Place", "A flowing body of water such as a river or stream."),
    "TerrainFeature": ("Place", "A natural physical feature of the land surface."),
    "MountainRange": ("Place", "A connected series of mountains."),
    "Building": ("Place", "A constructed architectural structure."),
    "Venue": ("Place", "A place where events or matches are held."),
    "HistoricalRegion": ("Place", "A region defined by historical rather than current boundaries."),
    "TransportHub": ("Place", "A transportation facility such as an airport, station, or port."),
    "TransportLine": ("Place", "A transportation route or transit line."),
    # --- Work ---
    "CreativeWork": ("Work", "An intellectual or artistic creation."),
    "MediaWork": ("Work", "A film, television, or media production."),
    "Film": ("Work", "A motion picture or movie."),
    "TelevisionShow": ("Work", "A television or broadcast program."),
    "Book": ("Work", "A written or printed literary work."),
    "MusicWork": ("Work", "A musical composition, song, or recording."),
    "VideoGame": ("Work", "An electronic interactive game."),
    "Series": ("Work", "An ordered set of related works."),
    "Franchise": ("Work", "A media franchise spanning multiple related works."),
    # --- Artifact ---
    "Product": ("Artifact", "A manufactured object or product."),
    "Vehicle": ("Artifact", "A machine for transporting people or goods."),
    "Instrument": ("Artifact", "A device used to produce music."),
    "Drug": ("Artifact", "A medicinal substance used for treatment."),
    "Currency": ("Artifact", "A system of money used in a place."),
    # --- Category (abstract classes / types) ---
    "Genre": ("Category", "A category of artistic, musical, or literary composition."),
    "Occupation": ("Category", "A profession or type of work a person does."),
    "Sport": ("Category", "A competitive physical or skill-based activity."),
    "Award": ("Category", "A prize or recognition granted for achievement."),
    "Language": ("Category", "A system of human communication."),
    "Religion": ("Category", "A system of faith and worship."),
    "Industry": ("Category", "A branch of economic or commercial activity."),
    "Field": ("Category", "A domain of study, research, or work."),
    "Movement": ("Category", "An artistic, cultural, or intellectual movement."),
    "Position": ("Category", "An office, post, or role that can be held."),
    "Title": ("Category", "An honorific or formal title."),
    "MilitaryRank": ("Category", "A rank within a military hierarchy."),
    "WritingSystem": ("Category", "A system of visual symbols for recording language."),
    "MedicalCondition": ("Category", "A disease, disorder, or medical condition."),
    "Taxon": ("Category", "A biological taxonomic group."),
    "VoiceType": ("Category", "A classification of the human singing voice."),
    "EthnicGroup": ("Category", "A group sharing a common cultural or ancestral identity."),
    "Class": ("Category", "A class or type of thing (used for taxonomic relations like subclass/instance of)."),
    # --- Event ---
    "Event": ("Event", "Something that happens at a given time and place."),
    "Conflict": ("Event", "An armed conflict, war, or battle."),
    "Election": ("Event", "A formal process of choosing a person for office."),
    "Competition": ("Event", "An organized contest between participants."),
    "SportsSeason": ("Event", "A single season of a sports club or competition."),
    # --- Value ---
    "Date": ("Value", "A calendar date or year."),
    "NumericValue": ("Value", "A numeric quantity or measurement."),
    "TextValue": ("Value", "A free-text or string value."),
    # --- Thing (root / generic) ---
    "Entity": ("Thing", "A generic entity used when the domain or range is open or heterogeneous."),
}

# ---------------------------------------------------------------------------
# Per-relation mapping: name -> (head_type, tail_type, detailed_definition)
# names MUST match the CSV exactly (verified below).
# ---------------------------------------------------------------------------
REBEL = {
    "country": ("Entity", "Country", "Links an item to the sovereign country in which it is located."),
    "contains administrative territorial entity": ("AdministrativeTerritory", "AdministrativeTerritory", "Links an administrative region to a smaller administrative region it contains."),
    "contains settlement": ("AdministrativeTerritory", "Settlement", "Links an administrative region to a populated settlement located within it."),
    "located in the administrative territorial entity": ("Entity", "AdministrativeTerritory", "Links an item to the administrative region that contains it."),
    "date of birth": ("Person", "Date", "Links a person to their date of birth."),
    "place of birth": ("Person", "Location", "Links a person to the place where they were born."),
    "date of death": ("Person", "Date", "Links a person to their date of death."),
    "place of death": ("Person", "Location", "Links a person to the place where they died."),
    "start time": ("Entity", "Date", "Links an item or event to the time at which it began."),
    "location": ("Entity", "Location", "Links an item to the place where it is physically located."),
    "country of citizenship": ("Person", "Country", "Links a person to the country of which they are a citizen."),
    "subclass of": ("Class", "Class", "Links a class to a more general class of which it is a subtype."),
    "language used": ("Entity", "Language", "Links an item to a language it employs."),
    "mouth of the watercourse": ("Watercourse", "BodyOfWater", "Links a river to the body of water into which it flows."),
    "mountain range": ("Location", "MountainRange", "Links a mountain or peak to the range it belongs to."),
    "tributary": ("Watercourse", "Watercourse", "Links a watercourse to a tributary that flows into it."),
    "member of": ("Entity", "Organization", "Links a person or item to an organization or group it belongs to."),
    "inception": ("Entity", "Date", "Links an item to the date it was founded or created."),
    "publication date": ("CreativeWork", "Date", "Links a work to the date it was published or released."),
    "genre": ("CreativeWork", "Genre", "Links a creative work to its artistic or literary genre."),
    "director": ("Film", "Person", "Links a film or show to the person who directed it."),
    "cast member": ("Film", "Person", "Links a film or show to a performer in its cast."),
    "media franchise": ("CreativeWork", "Franchise", "Links a work to the media franchise it belongs to."),
    "instance of": ("Entity", "Class", "Links an item to the class of which it is a specific instance."),
    "point in time": ("Event", "Date", "Links an event to the specific point in time at which it occurred."),
    "participant in": ("Entity", "Event", "Links a person or group to an event they took part in."),
    "applies to jurisdiction": ("Entity", "AdministrativeTerritory", "Links a rule or office to the jurisdiction in which it applies."),
    "number of seats": ("Organization", "NumericValue", "Links a body to its number of seats."),
    "sport": ("Entity", "Sport", "Links a team, athlete, or event to the sport it concerns."),
    "position played on team / speciality": ("Person", "Position", "Links an athlete to the position or speciality they play on a team."),
    "publisher": ("CreativeWork", "Organization", "Links a work to the organization that published it."),
    "uses": ("Entity", "Entity", "Links an item to a method, tool, or resource it employs."),
    "founded by": ("Organization", "Person", "Links an organization to the person who founded it."),
    "discontinued date": ("Entity", "Date", "Links a product or service to the date it was discontinued."),
    "editor": ("CreativeWork", "Person", "Links a work to the person who edited it."),
    "work period (start)": ("Person", "Date", "Links a person to the year they began their professional activity."),
    "record label": ("MusicWork", "Organization", "Links a musical work or artist to its record label."),
    "country of origin": ("Entity", "Country", "Links a work or product to the country it originates from."),
    "member of sports team": ("Person", "SportsTeam", "Links an athlete to a sports team they play for."),
    "work location": ("Person", "Location", "Links a person to the place where they worked."),
    "headquarters location": ("Organization", "Location", "Links an organization to the place of its headquarters."),
    "heritage designation": ("Building", "Class", "Links a site to its official heritage protection status."),
    "maintained by": ("Entity", "Organization", "Links infrastructure to the organization that maintains it."),
    "part of": ("Entity", "Entity", "Links an item to a larger whole of which it is a part."),
    "occupant": ("Building", "Entity", "Links a building to the entity that occupies it."),
    "home venue": ("SportsTeam", "Venue", "Links a sports team to the venue it uses as its home ground."),
    "platform": ("VideoGame", "Product", "Links software or a game to a platform on which it runs."),
    "has part": ("Entity", "Entity", "Links a whole to a component part it contains."),
    "narrative location": ("CreativeWork", "Location", "Links a work to the location where its narrative is set."),
    "father": ("Person", "Person", "Links a person to their father."),
    "child": ("Person", "Person", "Links a person to their child."),
    "located in or next to body of water": ("Location", "BodyOfWater", "Links a place to the body of water in or next to which it lies."),
    "organizer": ("Event", "Entity", "Links an event to the person or organization that organized it."),
    "capital": ("Country", "Settlement", "Links a country or region to its capital city."),
    "conflict": ("Entity", "Conflict", "Links a person or unit to an armed conflict they were involved in."),
    "capital of": ("Settlement", "AdministrativeTerritory", "Links a city to the country or region it is the capital of."),
    "screenwriter": ("Film", "Person", "Links a film or show to the person who wrote its screenplay."),
    "connecting line": ("Location", "Entity", "Links a station or stop to a transport line that connects it."),
    "sports discipline competed in": ("Person", "Sport", "Links an athlete to the sports discipline they compete in."),
    "position held": ("Person", "Position", "Links a person to a public office or position they held."),
    "legislated by": ("Entity", "Organization", "Links a law or institution to the body that legislated it."),
    "subsidiary": ("Company", "Company", "Links a company to a subsidiary it controls."),
    "owned by": ("Entity", "Entity", "Links an item to the entity that owns it."),
    "manufacturer": ("Product", "Company", "Links a product to the company that manufactures it."),
    "patron saint": ("Entity", "Person", "Links a place or institution to its patron saint."),
    "religion": ("Entity", "Religion", "Links a person or group to their religion."),
    "based on": ("CreativeWork", "CreativeWork", "Links a work to the earlier work it is based on."),
    "creator": ("CreativeWork", "Person", "Links a work to the person who created it."),
    "notable work": ("Person", "CreativeWork", "Links a creator to a notable work they produced."),
    "author": ("Book", "Person", "Links a written work to its author."),
    "writing language": ("CreativeWork", "Language", "Links a written work to the language it was written in."),
    "named after": ("Entity", "Entity", "Links an item to the entity it is named after."),
    "list of works": ("CreativeWork", "Person", "Links a list-work to the creator whose works it enumerates."),
    "archives at": ("Entity", "Organization", "Links a person or body to the institution holding its archives."),
    "ethnic group": ("Entity", "EthnicGroup", "Links a person or place to an ethnic group."),
    "league": ("SportsTeam", "SportsLeague", "Links a sports team to the league it plays in."),
    "field of this occupation": ("Occupation", "Field", "Links an occupation to the field of work it belongs to."),
    "practiced by": ("Occupation", "Person", "Links an occupation or practice to those who practice it."),
    "distributed by": ("Film", "Company", "Links a work to the company that distributes it."),
    "collection": ("CreativeWork", "Organization", "Links an artwork to the collection that holds it."),
    "has works in the collection": ("Organization", "Person", "Links a collection to a creator whose works it holds."),
    "member of political party": ("Person", "PoliticalParty", "Links a person to a political party they belong to."),
    "located on terrain feature": ("Location", "TerrainFeature", "Links a place to the terrain feature it sits on."),
    "continent": ("Location", "Continent", "Links a place to the continent it is part of."),
    "replaced by": ("Entity", "Entity", "Links an item to the successor that replaced it."),
    "replaces": ("Entity", "Entity", "Links an item to the predecessor it replaced."),
    "original broadcaster": ("TelevisionShow", "Organization", "Links a program to its original broadcaster."),
    "performer": ("MusicWork", "Person", "Links a work to the artist who performed it."),
    "language of work or name": ("CreativeWork", "Language", "Links a work or name to its language."),
    "country for sport": ("Person", "Country", "Links an athlete to the country they represent in sport."),
    "political alignment": ("Organization", "Class", "Links an organization to its political alignment."),
    "chairperson": ("Organization", "Person", "Links an organization to its chairperson."),
    "followed by": ("CreativeWork", "CreativeWork", "Links a work to the work that follows it in a sequence."),
    "follows": ("CreativeWork", "CreativeWork", "Links a work to the work it follows in a sequence."),
    "parent organization": ("Company", "Company", "Links an organization to its parent organization."),
    "producer": ("Film", "Person", "Links a work to the person or company that produced it."),
    "lyrics by": ("MusicWork", "Person", "Links a song to the person who wrote its lyrics."),
    "librettist": ("MusicWork", "Person", "Links an opera or musical to its librettist."),
    "composer": ("MusicWork", "Person", "Links a musical work to its composer."),
    "location of formation": ("Organization", "Location", "Links a group to the place where it was formed."),
    "basin country": ("BodyOfWater", "Country", "Links a body of water to a country in its drainage basin."),
    "present in work": ("Entity", "CreativeWork", "Links an entity to a work in which it appears."),
    "characters": ("CreativeWork", "Person", "Links a work to a character that appears in it."),
    "developer": ("VideoGame", "Organization", "Links software or a game to its developer."),
    "legislative body": ("AdministrativeTerritory", "Organization", "Links a jurisdiction to its legislative body."),
    "influenced by": ("Person", "Person", "Links a creator to a person or movement that influenced them."),
    "owner of": ("Entity", "Entity", "Links an owner to an item it owns."),
    "product or material produced": ("Company", "Product", "Links a producer to the product or material it produces."),
    "production company": ("Film", "Company", "Links a work to the company that produced it."),
    "participant": ("Event", "Entity", "Links an event to a participant who took part in it."),
    "instrument": ("Person", "Instrument", "Links a musician to an instrument they play."),
    "occupation": ("Person", "Occupation", "Links a person to their occupation."),
    "family": ("Person", "Family", "Links a person to the family they belong to."),
    "educated at": ("Person", "EducationalInstitution", "Links a person to an institution where they were educated."),
    "part of the series": ("CreativeWork", "Series", "Links a work to the series it is part of."),
    "movement": ("Person", "Movement", "Links a creator to an artistic movement they belong to."),
    "candidacy in election": ("Person", "Election", "Links a person to an election they were a candidate in."),
    "candidate": ("Election", "Person", "Links an election to a candidate running in it."),
    "head coach": ("SportsTeam", "Person", "Links a team to its head coach."),
    "contains": ("Entity", "Entity", "Links a container to an item it contains."),
    "different from": ("Entity", "Entity", "Links an item to another item it should not be confused with."),
    "military rank": ("Person", "MilitaryRank", "Links a person to a military rank they held."),
    "place served by transport hub": ("Entity", "Settlement", "Links a transport hub to the settlement it serves."),
    "original language of film or TV show": ("Film", "Language", "Links a film or show to its original language."),
    "executive body": ("AdministrativeTerritory", "Organization", "Links a jurisdiction to its executive body."),
    "chief executive officer": ("Company", "Person", "Links a company to its chief executive officer."),
    "game mechanics": ("VideoGame", "Class", "Links a game to a mechanic it features."),
    "partially coincident with": ("Entity", "Entity", "Links an item to another that partially overlaps with it."),
    "target": ("Entity", "Entity", "Links an action or work to the entity it targets."),
    "head of state": ("Country", "Person", "Links a state to its head of state."),
    "military branch": ("Person", "MilitaryUnit", "Links a person to the military branch they serve in."),
    "employer": ("Person", "Organization", "Links a person to their employer."),
    "facet of": ("Entity", "Entity", "Links a topic to a broader topic it is a facet of."),
    "presenter": ("TelevisionShow", "Person", "Links a program to its presenter or host."),
    "league system": ("SportsLeague", "Class", "Links a league to the league system it operates within."),
    "sports season of league or competition": ("SportsSeason", "SportsLeague", "Links a sports season to its league or competition."),
    "main subject": ("CreativeWork", "Entity", "Links a work to its main subject."),
    "operator": ("Entity", "Organization", "Links a facility to the organization that operates it."),
    "studied by": ("Entity", "Field", "Links a subject to the field that studies it."),
    "studies": ("Field", "Entity", "Links a field of study to the subject it studies."),
    "authority": ("Entity", "Organization", "Links an area to the authority that governs it."),
    "track gauge": ("Entity", "NumericValue", "Links a railway to its track gauge."),
    "competition class": ("Entity", "Class", "Links a competitor to the competition class it races in."),
    "religious order": ("Person", "ReligiousOrder", "Links a person to a religious order they belong to."),
    "victory": ("Person", "Competition", "Links a competitor to a competition they won."),
    "winner": ("Competition", "Person", "Links a competition to its winner."),
    "office held by head of government": ("AdministrativeTerritory", "Position", "Links a jurisdiction to the office held by its head of government."),
    "game mode": ("VideoGame", "Class", "Links a game to a game mode it offers."),
    "head of government": ("AdministrativeTerritory", "Person", "Links a jurisdiction to its head of government."),
    "field of work": ("Person", "Field", "Links a person to their field of work."),
    "title of chess person": ("Person", "Title", "Links a chess player to their chess title."),
    "terminus": ("Entity", "Location", "Links a route or line to its terminus."),
    "foundational text": ("Organization", "CreativeWork", "Links an institution to its foundational text."),
    "participating team": ("Event", "SportsTeam", "Links a competition to a team participating in it."),
    "located in present-day administrative territorial entity": ("Entity", "AdministrativeTerritory", "Links a historical item to the present-day administrative region containing its location."),
    "possible treatment": ("MedicalCondition", "Drug", "Links a medical condition to a possible treatment for it."),
    "medical condition treated": ("Drug", "MedicalCondition", "Links a treatment to a medical condition it treats."),
    "drug used for treatment": ("MedicalCondition", "Drug", "Links a condition to a drug used to treat it."),
    "floruit": ("Person", "Date", "Links a person to the period during which they were active."),
    "season of club or team": ("SportsSeason", "SportsTeam", "Links a season to the club or team it belongs to."),
    "shares border with": ("Location", "Location", "Links a place to another place it shares a border with."),
    "officeholder": ("Position", "Person", "Links an office to the person currently holding it."),
    "office held by head of the organization": ("Organization", "Position", "Links an organization to the office held by its head."),
    "after a work by": ("CreativeWork", "Person", "Links an adaptation to the author of the original work."),
    "space launch vehicle": ("Entity", "Vehicle", "Links a mission or payload to its launch vehicle."),
    "start point": ("Entity", "Location", "Links a route to its starting point."),
    "end time": ("Entity", "Date", "Links an item or event to the time at which it ended."),
    "currency": ("Country", "Currency", "Links a place to the currency used there."),
    "inflows": ("BodyOfWater", "BodyOfWater", "Links a body of water to a watercourse that flows into it."),
    "lakes on river": ("Watercourse", "BodyOfWater", "Links a river to a lake located on it."),
    "area": ("Location", "NumericValue", "Links a place to its measured area."),
    "said to be the same as": ("Entity", "Entity", "Links an item to another item said to be identical to it."),
    "industry": ("Company", "Industry", "Links a company to the industry it operates in."),
    "type foundry": ("Product", "Company", "Links a typeface to the foundry that created it."),
    "dissolved, abolished or demolished date": ("Entity", "Date", "Links an item to the date it was dissolved, abolished, or demolished."),
    "has cause": ("Event", "Entity", "Links an effect or event to its cause."),
    "historical region": ("Location", "HistoricalRegion", "Links a place to a historical region it belongs to."),
    "depicts": ("CreativeWork", "Entity", "Links a work to an entity it depicts."),
    "significant event": ("Entity", "Event", "Links an item to a significant event in its history."),
    "licensed to broadcast to": ("Organization", "Location", "Links a broadcaster to the area it is licensed to serve."),
    "relative": ("Person", "Person", "Links a person to a relative."),
    "sibling": ("Person", "Person", "Links a person to their sibling."),
    "allegiance": ("Person", "Organization", "Links a person or unit to the entity they are loyal to."),
    "speaker": ("Entity", "Person", "Links a speech or proceeding to the person who spoke."),
    "is a list of": ("CreativeWork", "Class", "Links a list-work to the category of items it lists."),
    "signatory": ("Entity", "Person", "Links a treaty or document to one of its signatories."),
    "commemorates": ("Entity", "Entity", "Links a memorial to the person or event it commemorates."),
    "separated from": ("Entity", "Entity", "Links an item to the entity it was separated from."),
    "award received": ("Entity", "Award", "Links a recipient to an award they received."),
    "interested in": ("Person", "Entity", "Links a person to a topic they are interested in."),
    "residence": ("Person", "Location", "Links a person to the place where they reside."),
    "writing system": ("Language", "WritingSystem", "Links a language to its writing system."),
    "diplomatic relation": ("Country", "Country", "Links a country to another it has diplomatic relations with."),
    "endemic to": ("Taxon", "Location", "Links a species to the location it is endemic to."),
    "taxonomic type": ("Taxon", "Taxon", "Links a taxon to its taxonomic type."),
    "voice type": ("Person", "VoiceType", "Links a singer to their voice type."),
}

# head/tail types here are MERGED from the prior curated wiki-nre_schema.json
# (finer typing: MediaWork / TransportHub / TransportLine / MilitaryBranch) +
# the `parent` taxonomy (added here) + detailed definitions. See Research Log
# 2026-06-16 (merge decision C).
WIKINRE = {
    "located in the administrative territorial entity": ("Location", "AdministrativeTerritory", "Links a place to the administrative region that contains it."),
    "country": ("Location", "Country", "Links a place to the sovereign country in which it is located."),
    "director": ("MediaWork", "Person", "Links a film or show to the person who directed it."),
    "cast member": ("MediaWork", "Person", "Links a film or show to a performer in its cast."),
    "screenwriter": ("MediaWork", "Person", "Links a film or show to the person who wrote its screenplay."),
    "part of": ("Entity", "Entity", "Links an item to a larger whole of which it is a part."),
    "located on terrain feature": ("Location", "TerrainFeature", "Links a place to the terrain feature it sits on."),
    "original language of film or TV show": ("MediaWork", "Language", "Links a film or show to its original language."),
    "employer": ("Person", "Organization", "Links a person to their employer."),
    "place served by transport hub": ("TransportHub", "Location", "Links a transport hub to the place it serves."),
    "named after": ("Entity", "Entity", "Links an item to the entity it is named after."),
    "owned by": ("Entity", "Organization", "Links an item to the organization that owns it."),
    "country of citizenship": ("Person", "Country", "Links a person to the country of which they are a citizen."),
    "participant in": ("Person", "Event", "Links a person or group to an event they took part in."),
    "place of birth": ("Person", "Location", "Links a person to the place where they were born."),
    "educated at": ("Person", "EducationalInstitution", "Links a person to an institution where they were educated."),
    "production company": ("MediaWork", "Organization", "Links a work to the organization that produced it."),
    "distributed by": ("CreativeWork", "Organization", "Links a work to the organization that distributes it."),
    "film editor": ("MediaWork", "Person", "Links a film to the person who edited it."),
    "member of sports team": ("Person", "SportsTeam", "Links an athlete to a sports team they play for."),
    "place of death": ("Person", "Location", "Links a person to the place where they died."),
    "place of burial": ("Person", "Location", "Links a person to the place where they were buried."),
    "country of origin": ("Entity", "Country", "Links a work or product to the country it originates from."),
    "operator": ("Entity", "Organization", "Links a facility to the organization that operates it."),
    "located in or next to body of water": ("Location", "BodyOfWater", "Links a place to the body of water in or next to which it lies."),
    "work location": ("Person", "Location", "Links a person to the place where they worked."),
    "producer": ("CreativeWork", "Person", "Links a work to the person who produced it."),
    "language of work or name": ("CreativeWork", "Language", "Links a work or name to its language."),
    "filming location": ("MediaWork", "Location", "Links a film to a location where it was filmed."),
    "location": ("Entity", "Location", "Links an item to the place where it is physically located."),
    "conflict": ("Person", "Conflict", "Links a person (or military unit) to an armed conflict they were involved in."),
    "composer": ("CreativeWork", "Person", "Links a musical work to the person who composed it."),
    "residence": ("Person", "Location", "Links a person to the place where they reside."),
    "after a work by": ("CreativeWork", "Person", "Links a work to the person (author/artist) whose original work it is based on or inspired by."),
    "basin country": ("BodyOfWater", "Country", "Links a body of water to a country in its drainage basin."),
    "lake outflow": ("BodyOfWater", "BodyOfWater", "Links a lake to the watercourse through which it drains."),
    "inflows": ("BodyOfWater", "BodyOfWater", "Links a body of water to a watercourse that flows into it."),
    "executive producer": ("MediaWork", "Person", "Links a work to one of its executive producers."),
    "connecting line": ("TransportLine", "TransportLine", "Links a transit line or station to a transport line that connects to it."),
    "continent": ("Location", "Continent", "Links a place to the continent it is part of."),
    "headquarters location": ("Organization", "Location", "Links an organization to the place of its headquarters."),
    "student of": ("Person", "Person", "Links a person to their teacher or mentor."),
    "creator": ("CreativeWork", "Person", "Links a work to the person who created it."),
    "award received": ("Person", "Award", "Links a recipient to an award they received."),
    "military branch": ("Person", "MilitaryBranch", "Links a person (or unit) to the military branch they belong to."),
}


def load_csv(path):
    """Return ordered list of (name, general_definition) from a 2-col CSV."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            name = row[0].strip()
            gdef = row[1].strip() if len(row) > 1 else ""
            rows.append((name, gdef))
    return rows


def gold_relation_set(path):
    rels = set()
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                triples = ast.literal_eval(line)
            except Exception:
                continue
            for t in triples:
                if len(t) == 3:
                    rels.add(t[1])
    return rels


def build(dataset, mapping):
    csv_path = os.path.join(HERE, f"{dataset}_schema.csv")
    out_path = os.path.join(HERE, f"{dataset}_schema.json")
    gold_path = os.path.join(REPO, "soda", "evaluate", "references", f"{dataset}.txt")

    csv_rows = load_csv(csv_path)
    csv_names = [n for n, _ in csv_rows]

    # ---- VERIFY name parity: every CSV name has a mapping, and vice-versa ----
    csv_set, map_set = set(csv_names), set(mapping.keys())
    missing_in_map = csv_set - map_set
    extra_in_map = map_set - csv_set
    assert not missing_in_map, f"[{dataset}] names in CSV but not mapped: {sorted(missing_in_map)}"
    assert not extra_in_map, f"[{dataset}] mapped names not in CSV: {sorted(extra_in_map)}"

    # ---- VERIFY every referenced entity type exists in the taxonomy ----
    used_types = set()
    for h, t, _ in mapping.values():
        used_types.update([h, t])
    undefined = used_types - set(ENTITY_TYPES.keys())
    assert not undefined, f"[{dataset}] entity types used but not defined: {sorted(undefined)}"

    # ---- build relation_types (preserve CSV order) ----
    relation_types = {}
    for name, gdef in csv_rows:
        head, tail, detail = mapping[name]
        relation_types[name] = {
            "definition": detail,
            "general_definition": gdef,
            "head_type": head,
            "tail_type": tail,
            "attributes": {},
        }

    # ---- build entity_types (only the ones this dataset uses, + Entity root) ----
    keep = used_types | {"Entity"}
    entity_types = {}
    for et in ENTITY_TYPES:                       # stable, taxonomy-order
        if et in keep:
            parent, edef = ENTITY_TYPES[et]
            entity_types[et] = {"parent": parent, "definition": edef, "attributes": {}}

    schema = {"entity_types": entity_types, "relation_types": relation_types}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    # ---- VERIFY against gold reference (if present) ----
    gold = gold_relation_set(gold_path)
    gold_msg = "no gold file"
    if gold is not None:
        miss = gold - csv_set
        cov = 100 * len(gold & csv_set) / max(len(gold), 1)
        gold_msg = f"gold={len(gold)} coverage={cov:.1f}% missing={sorted(miss)}"

    print(f"[{dataset}] OK  relations={len(relation_types)}  entity_types={len(entity_types)}  "
          f"(CSV={len(csv_names)})  | {gold_msg}  -> {out_path}")


if __name__ == "__main__":
    build("rebel", REBEL)
    build("wiki-nre", WIKINRE)
    print("Done.")
