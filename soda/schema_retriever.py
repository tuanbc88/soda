import logging
from typing import Dict, List

import torch
from sentence_transformers import util

logger = logging.getLogger(__name__)


class SchemaRetriever:
    """
    Handle schema retrieval using embedding similarity.
    """

    def __init__(
        self,
        target_schema_dict: dict,
        embedding_model,
        embedding_tokenizer,
        finetuned_e5mistral=False,
    ):

        self.schema = target_schema_dict
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        self.finetuned_e5mistral = finetuned_e5mistral

        self.relation_schema = self.schema.get("relation_types", {})
        self.entity_schema = self.schema.get("entities", {})

        logger.info(
            f"SchemaRetriever init: {len(self.relation_schema)} relations, {len(self.entity_schema)} entities"
        )

        self._build_index()

    # ------------------------------------------------

    def _relation_to_text(self, rel, info):

        parts = [rel]

        if "definition" in info:
            parts.append(info["definition"])

        if "aliases" in info:
            parts.extend(info["aliases"])

        if "examples" in info:
            for ex in info["examples"]:
                parts.append(" ".join(ex))

        return " ".join(parts)

    # ------------------------------------------------

    def _entity_to_text(self, ent, info):

        parts = [ent]

        if "definition" in info:
            parts.append(info["definition"])

        if "aliases" in info:
            parts.extend(info["aliases"])

        if "examples" in info:
            parts.extend(info["examples"])

        return " ".join(parts)

    # ------------------------------------------------

    def _encode(self, texts):

        if self.finetuned_e5mistral:

            inputs = self.embedding_tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self.embedding_model(**inputs)

            embeddings = outputs.last_hidden_state.mean(dim=1)

            return embeddings

        else:

            return self.embedding_model.encode(
                texts,
                convert_to_tensor=True
            )

    # ------------------------------------------------

    def _build_index(self):

        # relation index
        self.relation_list = list(self.relation_schema.keys())

        relation_texts = [
            self._relation_to_text(rel, self.relation_schema[rel])
            for rel in self.relation_list
        ]

        self.relation_embeddings = self._encode(relation_texts)

        # entity index
        self.entity_list = list(self.entity_schema.keys())

        entity_texts = [
            self._entity_to_text(ent, self.entity_schema[ent])
            for ent in self.entity_list
        ]

        self.entity_embeddings = self._encode(entity_texts)

        logger.info(
            f"Schema embedding index built: {len(self.relation_list)} relations, {len(self.entity_list)} entities"
        )

    # ------------------------------------------------

    def retrieve_relevant_relations(self, query_text: str, top_k: int = 5) -> List[str]:

        query_emb = self._encode([query_text])[0]

        scores = util.cos_sim(query_emb, self.relation_embeddings)[0]

        top = scores.topk(top_k)

        return [self.relation_list[i] for i in top.indices]

    # ------------------------------------------------

    def retrieve_relevant_entities(self, query_text: str, top_k: int = 5) -> List[str]:

        query_emb = self._encode([query_text])[0]

        scores = util.cos_sim(query_emb, self.entity_embeddings)[0]

        top = scores.topk(top_k)

        return [self.entity_list[i] for i in top.indices]

    # ------------------------------------------------

    def retrieve_schema(self, query_text: str, rel_top_k=5, ent_top_k=5):

        relations = self.retrieve_relevant_relations(query_text, rel_top_k)
        entities = self.retrieve_relevant_entities(query_text, ent_top_k)

        return {
            "relation_types": relations,
            "entities": entities
        }