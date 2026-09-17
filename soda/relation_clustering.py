import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        logger.info("[RelationCluster] Loading sentence transformer model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def collect_relation_phrases(triples: List[List[str]]) -> List[str]:
    relations = []
    for t in triples:
        if len(t) >= 2:
            relations.append(t[1])
    return list(set(relations))


def cluster_relations(relations: List[str], threshold: float = 0.9) -> Dict[str, List[str]]:
    """
    Simple greedy clustering using cosine similarity
    """

    if len(relations) == 0:
        return {}

    model = get_embedding_model()

    embeddings = model.encode(relations)

    clusters = []
    used = set()

    for i, r in enumerate(relations):

        if i in used:
            continue

        cluster = [r]
        used.add(i)

        for j in range(i + 1, len(relations)):

            if j in used:
                continue

            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[j].reshape(1, -1)
            )[0][0]

            if sim > threshold:
                cluster.append(relations[j])
                used.add(j)

        clusters.append(cluster)

    cluster_dict = {}

    for idx, cluster in enumerate(clusters):
        cluster_dict[f"cluster_{idx}"] = cluster

    logger.info(
        f"[RelationCluster] {len(relations)} relations → {len(cluster_dict)} clusters"
    )

    return cluster_dict


def build_relation_cluster_map(clusters: Dict[str, List[str]]) -> Dict[str, str]:

    relation_map = {}

    for cluster_id, rels in clusters.items():

        canonical_relation = rels[0]

        for r in rels:
            relation_map[r] = canonical_relation

    return relation_map


def normalize_triples(triples: List[List[str]], relation_map: Dict[str, str]):

    normalized = []

    for h, r, t in triples:

        new_r = relation_map.get(r, r)

        normalized.append([h, new_r, t])

    return normalized