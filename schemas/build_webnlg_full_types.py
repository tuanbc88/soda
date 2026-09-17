"""
LLM-inferred entity typing for webnlg_full's NEW relations (the scale set, 372 rel vs the 159 sample).

Stage v1/v2 (inference) for the scale set: Claude Opus 4.8 infers head_type/tail_type for the 222 relations
in webnlg_full that are NOT in webnlg_schema.json, from the relation name + example gold (head, tail) entities
(schemas/webnlg_full_new_relations.txt). Literal tails (measurements / dates / codes / quoted free text) are
typed NumericValue / Date / TextValue from the gold VALUES; entity tails are typed from the example entities.
Inference is INDEPENDENT of DBpedia domain/range (the grounding KB) so the downstream consistency rate at
scale is a meaningful validation, not circular. Output: schemas/webnlg_full_schema.json (150 reused + 222 new).
Then `build_gold_entity_types.py --stage ground --dataset webnlg_full` checks them; mismatches -> author review.
(DECISIONS 2026-06-25.)
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))

# 12 new low-level types needed by the scale set (parents reuse the 8-class upper layer; most exist in rebel).
NEW_TYPES = {
    "Material":       {"parent": "Artifact", "definition": "A substance or material (e.g. granite, asphalt, bronze).", "attributes": {}},
    "Taxon":          {"parent": "Thing",    "definition": "A biological taxon (family, genus, species).", "attributes": {}},
    "CelestialBody":  {"parent": "Thing",    "definition": "An astronomical object such as an asteroid, planet, or star.", "attributes": {}},
    "MilitaryUnit":   {"parent": "Agent",    "definition": "An organized military force, branch, or unit.", "attributes": {}},
    "MilitaryRank":   {"parent": "Category", "definition": "A military rank.", "attributes": {}},
    "MedicalCondition": {"parent": "Category", "definition": "A disease or medical condition.", "attributes": {}},
    "Religion":       {"parent": "Category", "definition": "A religion or denomination.", "attributes": {}},
    "Watercourse":    {"parent": "Place",    "definition": "A river or watercourse.", "attributes": {}},
    "Sport":          {"parent": "Category", "definition": "A sport.", "attributes": {}},
    "SportsPosition": {"parent": "Category", "definition": "A playing position or role within a sports team.", "attributes": {}},
    "Title":          {"parent": "Category", "definition": "An office, title, or honorific.", "attributes": {}},
    "Field":          {"parent": "Category", "definition": "An academic field or discipline.", "attributes": {}},
    # added 2026-06-25 from author mismatch adjudication of the scale set
    "LaunchPad":      {"parent": "Place",    "definition": "A rocket launch pad or launch complex.", "attributes": {}},
    "Hospital":       {"parent": "Place",    "definition": "A hospital.", "attributes": {}},
    "SpaceMission":   {"parent": "Event",    "definition": "A space mission or spaceflight.", "attributes": {}},
}

# relation -> (head_type, tail_type). Literal tails from gold values: NumericValue / Date / TextValue.
TYPING = {
    "1stRunwayLengthFeet": ("Airport", "NumericValue"), "1stRunwayLengthMetre": ("Airport", "NumericValue"),
    "1stRunwayNumber": ("Airport", "NumericValue"), "1stRunwaySurfaceType": ("Airport", "Material"),
    "2ndRunwaySurfaceType": ("Airport", "Material"), "3rdRunwayLengthFeet": ("Airport", "NumericValue"),
    "3rdRunwaySurfaceType": ("Airport", "Material"), "4thRunwayLengthFeet": ("Airport", "NumericValue"),
    "4thRunwaySurfaceType": ("Airport", "Material"), "5thRunwayNumber": ("Airport", "NumericValue"),
    "5thRunwaySurfaceType": ("Airport", "Material"), "LCCN_number": ("CreativeWork", "TextValue"),
    "NationalRegisterOfHistoricPlacesReferenceNumber": ("Building", "TextValue"),
    "abbreviation": ("CreativeWork", "TextValue"), "academicDiscipline": ("CreativeWork", "Field"),
    "activeYearsEndDate": ("Person", "Date"), "activeYearsEndYear": ("Person", "Date"),
    "activeYearsStartDate": ("Person", "Date"),
    "addedToTheNationalRegisterOfHistoricPlaces": ("Building", "Date"),
    "administrativeArrondissement": ("Settlement", "AdministrativeTerritory"),
    "aircraftFighter": ("MilitaryUnit", "Vehicle"), "aircraftHelicopter": ("MilitaryUnit", "Vehicle"),
    "alternativeName": ("Entity", "TextValue"), "architecturalStyle": ("Building", "TextValue"),
    "architecture": ("Building", "TextValue"), "areaOfLand": ("Settlement", "NumericValue"),
    "areaOfWater": ("Settlement", "NumericValue"), "areaUrban": ("Settlement", "NumericValue"),
    "associatedRocket": ("LaunchPad", "Vehicle"), "attackAircraft": ("MilitaryUnit", "Vehicle"),
    "averageSpeed": ("CelestialBody", "NumericValue"), "awards": ("Person", "NumericValue"),
    "background": ("Person", "TextValue"), "backupPilot": ("Vehicle", "Person"),
    "bandMember": ("Organization", "Person"), "battle": ("MilitaryUnit", "Event"),
    "bedCount": ("Hospital", "NumericValue"), "birthName": ("Person", "TextValue"),
    "birthYear": ("Person", "Date"), "buildingType": ("Building", "TextValue"),
    "capital": ("Country", "Settlement"), "carbohydrate": ("Food", "NumericValue"),
    "chairman": ("Organization", "Person"), "chairmanTitle": ("Organization", "TextValue"),
    "chairperson": ("Organization", "Person"), "chancellor": ("EducationalInstitution", "Person"),
    "chief": ("Organization", "Person"), "christeningDate": ("Vehicle", "Date"),
    "class": ("Vehicle", "TextValue"), "codenCode": ("CreativeWork", "TextValue"),
    "college": ("Person", "EducationalInstitution"), "colour": ("Organization", "TextValue"),
    "commander": ("Event", "Person"), "comparable": ("Vehicle", "Vehicle"),
    "competeIn": ("EducationalInstitution", "SportsLeague"), "cost": ("Building", "NumericValue"),
    "countryOrigin": ("Vehicle", "Country"), "countySeat": ("AdministrativeTerritory", "Settlement"),
    "creatorOfDish": ("Food", "Person"), "crewMembers": ("Vehicle", "Person"),
    "dean": ("EducationalInstitution", "Person"), "deathCause": ("Person", "MedicalCondition"),
    "deathYear": ("Person", "Date"), "dedicatedTo": ("Building", "Person"),
    "density": ("CelestialBody", "NumericValue"), "derivative": ("Genre", "Genre"),
    "designCompany": ("Vehicle", "Organization"), "developer": ("Building", "Organization"),
    "diameter": ("Vehicle", "NumericValue"), "distributingCompany": ("Organization", "Organization"),
    "distributor": ("MediaWork", "Organization"), "district": ("Building", "Location"),
    "division": ("Taxon", "Taxon"), "doctoralAdvisor": ("Person", "Person"),
    "doctoralStudent": ("Person", "Person"), "draftRound": ("Person", "NumericValue"),
    "draftTeam": ("Person", "SportsTeam"), "draftYear": ("Person", "Date"),
    "editor": ("CreativeWork", "Person"), "eissnNumber": ("CreativeWork", "TextValue"),
    "elevationAboveTheSeaLevelInFeet": ("Airport", "NumericValue"),
    "elevationAboveTheSeaLevelInMetres": ("Airport", "NumericValue"),
    "employer": ("Person", "Organization"), "escapeVelocity": ("CelestialBody", "NumericValue"),
    "failedLaunches": ("Vehicle", "NumericValue"), "family": ("Taxon", "Taxon"),
    "fat": ("Food", "NumericValue"), "fate": ("Organization", "Organization"),
    "finalFlight": ("Vehicle", "Date"), "firstPublicationYear": ("CreativeWork", "Date"),
    "floorArea": ("Building", "NumericValue"), "floorCount": ("Building", "NumericValue"),
    "formerName": ("CelestialBody", "TextValue"), "fossil": ("Taxon", "Taxon"),
    "frequency": ("CreativeWork", "TextValue"), "function": ("Vehicle", "TextValue"),
    "garrison": ("MilitaryUnit", "Settlement"), "gemstone": ("AdministrativeTerritory", "Material"),
    "generalManager": ("SportsTeam", "Person"), "genus": ("Taxon", "Taxon"),
    "governingBody": ("Settlement", "Organization"), "hasDeputy": ("Person", "Person"),
    "hasToItsNortheast": ("Settlement", "Location"), "hasToItsNorthwest": ("AdministrativeTerritory", "Location"),
    "headquarter": ("Organization", "Settlement"), "higher": ("Award", "Award"),
    "hubAirport": ("Organization", "Airport"), "iataLocationIdentifier": ("Airport", "TextValue"),
    "impactFactor": ("CreativeWork", "NumericValue"), "inOfficeWhileGovernor": ("Person", "Person"),
    "inOfficeWhileMonarch": ("Person", "Person"), "inOfficeWhilePrimeMinister": ("Person", "Person"),
    "inOfficeWhileVicePresident": ("Person", "Person"), "inaugurationDate": ("Building", "Date"),
    "influencedBy": ("Person", "Person"), "isPartOfMilitaryConflict": ("Event", "Event"),
    "isbnNumber": ("CreativeWork", "TextValue"), "issnNumber": ("CreativeWork", "TextValue"),
    "jurisdiction": ("Organization", "AdministrativeTerritory"), "largestCity": ("Country", "Settlement"),
    "latinName": ("EducationalInstitution", "TextValue"), "launchSite": ("SpaceMission", "Building"),
    "layout": ("Vehicle", "TextValue"), "leaderParty": ("Settlement", "Organization"),
    "legislature": ("Country", "Organization"), "libraryofCongressClassification": ("CreativeWork", "TextValue"),
    "locationIdentifier": ("Airport", "TextValue"), "maidenFlight": ("Vehicle", "Date"),
    "maidenVoyage": ("Vehicle", "Date"), "mascot": ("EducationalInstitution", "Entity"),
    "mass": ("CelestialBody", "NumericValue"), "material": ("Building", "Material"),
    "maximumTemperature": ("CelestialBody", "NumericValue"), "mayor": ("Settlement", "Person"),
    "meanTemperature": ("CelestialBody", "NumericValue"), "mediaType": ("CreativeWork", "TextValue"),
    "militaryBranch": ("Person", "MilitaryUnit"), "militaryRank": ("Person", "MilitaryRank"),
    "minimumTemperature": ("CelestialBody", "NumericValue"), "modelStartYear": ("Vehicle", "Date"),
    "modelYears": ("Vehicle", "Date"), "mostChampions": ("SportsLeague", "SportsTeam"),
    "motto": ("Entity", "TextValue"), "musicFusionGenre": ("Genre", "Genre"),
    "musicSubgenre": ("Genre", "Genre"), "nativeName": ("Building", "TextValue"),
    "nearestCity": ("Building", "Settlement"), "neighboringMunicipality": ("Settlement", "Settlement"),
    "notableWork": ("Person", "CreativeWork"), "numberOfLocations": ("Organization", "NumericValue"),
    "numberOfRooms": ("Building", "NumericValue"), "numberOfVotesAttained": ("Person", "NumericValue"),
    "oclcNumber": ("CreativeWork", "TextValue"), "officialLanguage": ("Country", "Language"),
    "officialSchoolColour": ("EducationalInstitution", "TextValue"), "operatingIncome": ("Organization", "NumericValue"),
    "operator": ("Building", "Organization"), "order": ("Taxon", "Taxon"),
    "outlookRanking": ("EducationalInstitution", "NumericValue"), "owningOrganisation": ("Building", "Organization"),
    "part": ("Settlement", "AdministrativeTerritory"), "partialFailures": ("Vehicle", "NumericValue"),
    "partsType": ("Settlement", "CreativeWork"), "patronSaint": ("Country", "Person"),
    "percentageOfAreaWater": ("Country", "NumericValue"), "place": ("Event", "Location"),
    "playerNumber": ("Person", "NumericValue"), "position": ("Person", "SportsPosition"),
    "predecessor": ("Person", "Person"), "profession": ("Person", "Occupation"),
    "protein": ("Food", "NumericValue"), "recordLabel": ("Person", "Organization"),
    "rector": ("EducationalInstitution", "Person"), "related": ("Food", "Food"),
    "relatedMeanOfTransportation": ("Vehicle", "Vehicle"), "religion": ("Country", "Religion"),
    "representative": ("AdministrativeTerritory", "Person"), "residence": ("Person", "Location"),
    "ribbonAward": ("Person", "TextValue"), "river": ("Country", "Watercourse"),
    "rocketStages": ("Vehicle", "NumericValue"), "saint": ("Settlement", "Person"),
    "served": ("Food", "TextValue"), "serviceStartYear": ("Person", "Date"),
    "servingSize": ("Food", "NumericValue"), "servingTemperature": ("Food", "TextValue"),
    "shipBeam": ("Vehicle", "NumericValue"), "shipClass": ("Vehicle", "TextValue"),
    "shipDisplacement": ("Vehicle", "NumericValue"), "shipDraft": ("Vehicle", "NumericValue"),
    "shipInService": ("Vehicle", "Date"), "shipLaidDown": ("Vehicle", "Date"),
    "shipLaunch": ("Vehicle", "Date"), "shipOrdered": ("Vehicle", "Date"),
    "shipPower": ("Vehicle", "TextValue"), "significantBuilding": ("Person", "Building"),
    "significantProject": ("Person", "Building"), "similarDish": ("Food", "TextValue"),
    "site": ("Building", "Location"), "spokenIn": ("Language", "Country"),
    "sportGoverningBody": ("Sport", "Organization"), "stadium": ("SportsTeam", "Building"),
    "stateOfOrigin": ("Person", "Country"), "stylisticOrigin": ("Genre", "Genre"),
    "surfaceArea": ("CelestialBody", "NumericValue"), "temperature": ("CelestialBody", "NumericValue"),
    "timeInSpace": ("Person", "NumericValue"), "title": ("Person", "Title"),
    "topSpeed": ("Vehicle", "NumericValue"), "totalLaunches": ("Vehicle", "NumericValue"),
    "totalProduction": ("Vehicle", "NumericValue"), "trainerAircraft": ("MilitaryUnit", "Vehicle"),
    "transmission": ("Vehicle", "TextValue"), "transportAircraft": ("MilitaryUnit", "Vehicle"),
    "unit": ("Person", "MilitaryUnit"), "universityTeam": ("SportsTeam", "EducationalInstitution"),
    "voice": ("MediaWork", "Person"), "website": ("Building", "TextValue"),
    "wheelbase": ("Vehicle", "NumericValue"), "width": ("Vehicle", "NumericValue"),
    "year": ("Food", "Date"), "yearOfConstruction": ("Building", "Date"),
}


def main():
    base = json.load(open(os.path.join(HERE, "webnlg_schema.json"), encoding="utf-8"))
    et = dict(base["entity_types"])
    for t, spec in NEW_TYPES.items():
        et.setdefault(t, spec)
    rt = dict(base["relation_types"])          # 150 reused (already typed)
    added = 0
    for rel, (h, t) in TYPING.items():
        if rel in rt:
            continue
        rt[rel] = {"head_type": h, "tail_type": t,
                   "general_definition": f"Relation '{rel}' linking a {h} to a {t}.",
                   "role": "main"}
        added += 1
    out = {"entity_types": et, "relation_types": rt}
    p = os.path.join(HERE, "webnlg_full_schema.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"webnlg_full_schema.json: {len(rt)} relations ({added} newly typed + {len(rt)-added} reused), "
          f"{len(et)} entity_types (+{len(NEW_TYPES)} new) -> {p}")


if __name__ == "__main__":
    main()
