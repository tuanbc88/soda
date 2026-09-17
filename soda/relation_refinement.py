import re


class RelationRefiner:

    def __init__(self, relation_hints=None):
        self.relation_hints = relation_hints or {}

    # ------------------------------------------------

    def refine_triples(self, triples, attributes_raw=None):

        entity_relations = []
        attributes = []

        attributes_raw = attributes_raw or []

        # -------------------------
        # refine triples
        # -------------------------

        for triple in triples:

            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                continue

            head, relation, tail = triple

            if not head or not relation or not tail:
                continue

            relation = self.normalize_relation(relation)

            if not self.validate_relation(head, relation, tail):
                continue

            entity_relations.append(
                (head.strip(), relation, tail.strip())
            )

        # -------------------------
        # refine attributes
        # -------------------------

        for attr in attributes_raw:

            if not isinstance(attr, dict):
                continue

            target = attr.get("target")
            attribute = attr.get("attribute")
            value = attr.get("value")

            if not target or not attribute or value is None:
                continue

            attribute = self.normalize_relation(attribute)

            attributes.append({
                "target": target.strip(),
                "attribute": attribute,
                "value": str(value).strip()
            })

        return entity_relations, attributes

    # ------------------------------------------------
    # normalize relation
    # ------------------------------------------------

    def normalize_relation(self, relation):

        r = relation.strip()
    
        if not r:
            return ""
    
        # nếu đã camelCase → giữ nguyên
        if re.match(r"^[a-z]+[A-Za-z0-9]*$", r):
            return r
    
        # snake_case → camelCase
        r = r.replace("_", " ")
    
        tokens = r.split()
    
        if not tokens:
            return ""
    
        tokens = [t.lower() for t in tokens]
    
        return tokens[0] + "".join(t.capitalize() for t in tokens[1:])

    # ------------------------------------------------
    # validate relation
    # ------------------------------------------------

    def validate_relation(self, head, relation, tail):

        if not relation:
            return False

        if relation.lower() == head.lower():
            return False

        if relation.lower() == str(tail).lower():
            return False

        if relation.isnumeric():
            return False

        if len(relation.split()) > 4:
            return False

        bad_relations = {"is", "are", "was", "be", "have", "has"}

        if relation.lower() in bad_relations:
            return False

        return True