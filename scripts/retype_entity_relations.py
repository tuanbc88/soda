"""
One-shot script: retype the 113 relations that carry generic Entity head/tail
in webnlg_schema.json with precise types from the schema taxonomy.
"""
import json
import shutil

RETYPE = {
    # --- astronomical objects ---
    "absoluteMagnitude":     ("Entity",                  "NumericValue"),
    "apoapsis":              ("Entity",                  "NumericValue"),
    "cosparId":              ("Entity",                  "TextValue"),
    "discovered":            ("Entity",                  "Date"),
    "discoverer":            ("Entity",                  "Person"),
    "epoch":                 ("Entity",                  "Date"),
    "orbitalPeriod":         ("Entity",                  "NumericValue"),
    "periapsis":             ("Entity",                  "NumericValue"),
    "rotationPeriod":        ("Entity",                  "NumericValue"),
    # --- person ---
    "activeYearsStartYear":  ("Person",                  "Date"),
    "affiliation":           ("Person",                  "Organization"),
    "associatedBand/associatedMusicalArtist": ("Person", "Organization"),
    "child":                 ("Person",                  "Person"),
    "dateOfRetirement":      ("Person",                  "Date"),
    "debutTeam":             ("Person",                  "SportsTeam"),
    "draftPick":             ("SportsTeam",              "Person"),
    "firstAppearanceInFilm": ("Person",                  "CreativeWork"),
    "formerTeam":            ("Person",                  "SportsTeam"),
    "inOfficeWhilePresident":("Person",                  "Person"),
    "instrument":            ("Person",                  "Product"),
    "office":                ("Person",                  "Occupation"),
    "party":                 ("Person",                  "Organization"),
    "professionalField":     ("Person",                  "Industry"),
    "selectedByNasa":        ("Person",                  "Organization"),
    "servedAsChiefOfTheAstronautOfficeIn": ("Person",   "Date"),
    "training":              ("Person",                  "Location"),
    "youthclub":             ("Person",                  "SportsTeam"),
    "currentclub":           ("Person",                  "SportsTeam"),
    "currentteam":           ("Person",                  "SportsTeam"),
    # --- place / territory ---
    "areaCode":              ("Settlement",              "TextValue"),
    "areaTotal":             ("Location",                "NumericValue"),
    "ceremonialCounty":      ("Settlement",              "AdministrativeTerritory"),
    "demonym":               ("Location",                "TextValue"),
    "elevationAboveTheSeaLevel": ("Location",            "NumericValue"),
    "ethnicGroup":           ("Location",                "Organization"),
    "hasToItsNorth":         ("Location",                "Location"),
    "hasToItsSoutheast":     ("Location",                "Location"),
    "hasToItsSouthwest":     ("Location",                "Location"),
    "hasToItsWest":          ("Location",                "Location"),
    "isPartOf":              ("Location",                "Location"),
    "populationDensity":     ("Settlement",              "NumericValue"),
    "populationMetro":       ("Settlement",              "NumericValue"),
    "state":                 ("Settlement",              "AdministrativeTerritory"),
    "utcOffset":             ("Location",                "TimeZone"),
    "senators":              ("AdministrativeTerritory", "Person"),
    # --- building / infrastructure ---
    "buildDate":             ("Building",                "Date"),
    "builder":               ("Building",                "Organization"),
    "buildingStartDate":     ("Building",                "Date"),
    "cityServed":            ("Building",                "Settlement"),
    "completionDate":        ("Building",                "Date"),
    "currentTenants":        ("Building",                "Organization"),
    "icaoLocationIdentifier":("Building",                "TextValue"),
    "operatingOrganisation": ("Building",                "Organization"),
    "runwayLength":          ("Building",                "NumericValue"),
    "runwayName":            ("Building",                "TextValue"),
    "runwaySurfaceType":     ("Building",                "TextValue"),
    "tenant":                ("Building",                "Organization"),
    # --- organization ---
    "academicStaffSize":     ("EducationalInstitution",  "NumericValue"),
    "assembly":              ("Organization",             "Organization"),
    "anthem":                ("Organization",             "CreativeWork"),
    "campus":                ("EducationalInstitution",   "Location"),
    "champions":             ("SportsLeague",             "SportsTeam"),
    "club":                  ("Person",                   "SportsTeam"),
    "coach":                 ("SportsTeam",               "Person"),
    "foundationPlace":       ("Organization",             "Location"),
    "foundingYear":          ("Organization",             "Date"),
    "keyPerson":             ("Organization",             "Person"),
    "manager":               ("Organization",             "Person"),
    "netIncome":             ("Organization",             "NumericValue"),
    "numberOfDoctoralStudents":     ("EducationalInstitution", "NumericValue"),
    "numberOfEmployees":            ("Organization",           "NumericValue"),
    "numberOfMembers":              ("Organization",           "NumericValue"),
    "numberOfPostgraduateStudents": ("EducationalInstitution", "NumericValue"),
    "numberOfStudents":             ("EducationalInstitution", "NumericValue"),
    "numberOfUndergraduateStudents":("EducationalInstitution", "NumericValue"),
    "regionServed":          ("Organization",             "Location"),
    "revenue":               ("Organization",             "NumericValue"),
    "service":               ("Organization",             "TextValue"),
    "sportsOffered":         ("Organization",             "Industry"),
    "staff":                 ("Organization",             "NumericValue"),
    "subsidiary":            ("Organization",             "Organization"),
    "wasGivenTheTechnicalCampusStatusBy": ("Organization", "Organization"),
    # --- creative / media works ---
    "followedBy":            ("CreativeWork",             "CreativeWork"),
    "lastAired":             ("MediaWork",                "Date"),
    "literaryGenre":         ("CreativeWork",             "Genre"),
    "numberOfPages":         ("CreativeWork",             "NumericValue"),
    "precededBy":            ("CreativeWork",             "CreativeWork"),
    "series":                ("MediaWork",                "MediaWork"),
    "season":                ("MediaWork",                "MediaWork"),
    "course":                ("EducationalInstitution",   "CreativeWork"),
    # --- product / vehicle ---
    "bodyStyle":             ("Product",                  "TextValue"),
    "cylinderCount":         ("Product",                  "NumericValue"),
    "engine":                ("Product",                  "Product"),
    "powerType":             ("Product",                  "TextValue"),
    "productionEndYear":     ("Product",                  "Date"),
    "productionStartYear":   ("Product",                  "Date"),
    # --- food (no Food type in schema, both sides stay generic) ---
    "dishVariation":         ("Entity",                   "Entity"),
    "ingredient":            ("Entity",                   "Entity"),
    "mainIngredient":        ("Entity",                   "Entity"),
    "bird":                  ("Country",                  "Entity"),
    # --- generic numeric / text attributes ---
    "height":                ("Entity",                   "NumericValue"),
    "length":                ("Entity",                   "NumericValue"),
    "weight":                ("Entity",                   "NumericValue"),
    "fullName":              ("Entity",                   "TextValue"),
    "longName":              ("Entity",                   "TextValue"),
    "nickname":              ("Entity",                   "TextValue"),
    "status":                ("Entity",                   "TextValue"),
    "type":                  ("Entity",                   "TextValue"),
    "extinctionDate":        ("Entity",                   "Date"),
    "designer":              ("Entity",                   "Person"),
    "origin":                ("Entity",                   "Location"),
    "successor":             ("Entity",                   "Entity"),
    "leaderTitle":           ("Organization",              "Occupation"),
}


def main():
    schema_path = "schemas/webnlg_schema.json"
    backup_path = "schemas/webnlg_schema_pre_retype.json"

    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    rels = schema["relation_types"]
    changed = 0
    missing = []

    for rel_name, (new_head, new_tail) in RETYPE.items():
        if rel_name not in rels:
            missing.append(rel_name)
            continue
        old_head = rels[rel_name].get("head_type", "")
        old_tail = rels[rel_name].get("tail_type", "")
        if old_head == new_head and old_tail == new_tail:
            continue
        rels[rel_name]["head_type"] = new_head
        rels[rel_name]["tail_type"] = new_tail
        changed += 1

    still_generic = [
        (k, v.get("head_type"), v.get("tail_type"))
        for k, v in rels.items()
        if v.get("head_type") == "Entity" and v.get("tail_type") == "Entity"
    ]

    print(f"Changed:              {changed}")
    print(f"Missing from schema:  {missing}")
    print(f"Still Entity/Entity:  {len(still_generic)} (intentionally generic)")
    for r in still_generic:
        print(f"  {r[0]}")

    shutil.copy(schema_path, backup_path)
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    print(f"\nSaved. Backup at {backup_path}")


if __name__ == "__main__":
    main()
