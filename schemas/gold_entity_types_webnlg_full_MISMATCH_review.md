# webnlg_full (SCALE set, 381 rel) — entity-type MISMATCH review

> ✅ **APPLIED 2026-06-25.** All 13 decisions applied: 9 `fix` → schema head/tail changed (+ 3 new types:
> LaunchPad, Hospital, SpaceMission); 4 `accept` (district, editor, nearestCity, yearOfConstruction) → recorded
> in `gold_entity_types_overrides.json` (webnlg_full section). After re-grounding, **mismatch = 0**. Kept as the
> adjudication record.

> LLM-inferred (Claude Opus 4.8) head/tail types for the 222 NEW relations, grounded vs DBpedia
> domain/range. These slots disagree with the KB. Decide fix/accept (edit Decision). The 159 sample
> relations + their adjudicated accepts are inherited from webnlg.

| relation | slot | schema type | dbo prop | KB allowed | Decision | Analyze | 
|---|---|---|---|---|---|---|
| associatedRocket | head | Building | dbo:associatedRocket | LaunchPad | fix schema type | LaunchPad không phải Building nói chung. Property này chỉ áp dụng cho bệ phóng. |
| bedCount | head | Building | dbo:bedCount | Hospital | fix schema type | Chỉ bệnh viện mới có bedCount trong DBpedia. Building quá rộng.|
| college | tail | SportsTeam | dbo:college | EducationalInstitution | fix schema type | Tail rõ ràng là trường đại học/cao đẳng. |
| creatorOfDish | tail | Food | dbo:creatorOfDish | Person | fix schema type | Người tạo món ăn phải là Person |
| district | head | Building | dbo:district | Place | accept | Building là subtype của Place; quan hệ district thường áp dụng cho building/venue trong WebNLG. |
| editor | tail | Person | dbo:editor | Agent | accept | Person là subtype phổ biến nhất của Agent. Giống foundedBy/creator ở bộ trước. |
| fossil | head | AdministrativeTerritory | dbo:fossil | Species | fix schema type | Property fossil có domain Species (loài có hóa thạch được tìm thấy). AdministrativeTerritory sai hoàn toàn.|
| genus | head | Food | dbo:genus | Species | fix schema type | Genus là phân loại sinh học, domain phải là Species/taxon. |
| launchSite | head | Vehicle | dbo:launchSite | SpaceMission | fix schema type | Domain là SpaceMission, không phải Vehicle. |
| launchSite | tail | Location | dbo:launchSite | Building | fix schema type | Launch site thường là launch pad / launch complex. Building hẹp hơn Location. |
| nearestCity | head | Building | dbo:nearestCity | Place | accept | Building là subtype của Place; property này rất thường dùng cho airport/building/landmark. |
| order | head | Food | dbo:order | Species | fix schema type | Order là phân loại sinh học. Food sai domain. |
| yearOfConstruction | head | Building | dbo:yearOfConstruction | Place | accept | Đây là DBpedia domain rất rộng (Place). Building là specialization hợp lý và phổ biến hơn. |

# ghi chú
Nhìn pattern thì các relation sinh học (genus, order, fossil) và không gian vũ trụ (associatedRocket, launchSite) là các mismatch thật; còn các relation kiểu editor, yearOfConstruction, district, nearestCity chủ yếu là trường hợp schema dùng subtype hợp lý hơn domain/range rất rộng của DBpedia