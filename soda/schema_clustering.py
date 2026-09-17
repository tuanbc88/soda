"""
Generic schema clustering for cluster-augmented retrieval.

This module is INTENTIONALLY generic: it works on a precomputed embedding dict
{name: vector} regardless of whether `name` is a relation type or an entity type,
and regardless of which case's embedding space (name / name+def / detail /
split_vector_*) produced the vectors. That makes it reusable by both
SchemaCanonicalizer (relations) and EntityCanonicalizer (entity types).

It does NOT load any model and does NOT re-encode text — it consumes the
already-built, already-normalized embeddings each canonicalizer maintains in
`self.schema_embedding_dict`. Clustering therefore always happens in the same
space as the case it belongs to (this is exactly the per-case clustering the
experiment design wants).

Greedy single-pass clustering by cosine similarity (vectors assumed L2-normalized,
so cosine == dot product). Cheap to rebuild for the schema sizes used here
(tens to a few hundred types).
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


class SchemaClusterIndex:
    """
    Greedy cosine-similarity clusters over a {name: embedding} dict.

    Usage:
        idx = SchemaClusterIndex(threshold=0.85)
        idx.maybe_rebuild(embedding_dict)        # builds (or rebuilds if size changed)
        clusters = idx.query_clusters(q_vec, top_m=2)   # [(cluster_idx, sim), ...]
        members  = idx.cluster_members(cluster_idx)     # [name, name, ...]
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = float(threshold)
        self.names = []                 # list[str]
        self.clusters = []              # list[list[str]]
        self.name_to_cluster = {}       # name -> cluster idx
        self.centroids = []             # list[np.ndarray] (normalized)
        self._built_n = -1              # number of items at last build

    # -----------------------------------------------------------------
    def build(self, embedding_dict):
        names = list(embedding_dict.keys())
        self.names = names

        if not names:
            self.clusters = []
            self.name_to_cluster = {}
            self.centroids = []
            self._built_n = 0
            return

        embs = np.array(
            [_normalize(embedding_dict[n]) for n in names]
        )

        used = set()
        clusters_idx = []

        for i in range(len(names)):
            if i in used:
                continue

            members = [i]
            used.add(i)

            sims = embs @ embs[i]   # cosine (normalized)

            for j in range(len(names)):
                if j == i or j in used:
                    continue
                if sims[j] > self.threshold:
                    members.append(j)
                    used.add(j)

            clusters_idx.append(members)

        self.clusters = [[names[k] for k in m] for m in clusters_idx]
        self.centroids = [_normalize(embs[m].mean(axis=0)) for m in clusters_idx]

        self.name_to_cluster = {}
        for ci, m in enumerate(clusters_idx):
            for k in m:
                self.name_to_cluster[names[k]] = ci

        self._built_n = len(names)

        logger.info(
            f"[SchemaCluster] {len(names)} items -> "
            f"{len(self.clusters)} clusters (thr={self.threshold})"
        )

    # -----------------------------------------------------------------
    def maybe_rebuild(self, embedding_dict):
        """Rebuild only if the schema grew/shrank since the last build."""
        if self._built_n != len(embedding_dict):
            self.build(embedding_dict)

    # -----------------------------------------------------------------
    def query_clusters(self, q_vec, top_m: int = 2):
        if not self.centroids:
            return []

        q = _normalize(q_vec)
        cent = np.array(self.centroids)
        sims = cent @ q

        order = np.argsort(-sims)[:top_m]
        return [(int(ci), float(sims[ci])) for ci in order]

    # -----------------------------------------------------------------
    def cluster_members(self, cluster_idx):
        return self.clusters[cluster_idx]


def select_cluster_extras(
    names,
    scores,
    q_vec,
    cluster_index: SchemaClusterIndex,
    exclude,
    top_m: int = 2,
    extra_k: int = 3,
):
    """
    Pick extra candidate names from the query's nearest clusters that are not
    already in `exclude` (the item-level top-k). Extras are ranked by their
    item-level score so the merged candidate list stays score-ordered.

    names  : list[str]  aligned with `scores`
    scores : np.ndarray item-level similarity scores (after any type penalty)
    q_vec  : query embedding (normalized)
    exclude: set[str]   names already chosen by item-level retrieval
    """
    if extra_k <= 0:
        return []

    best = cluster_index.query_clusters(q_vec, top_m=top_m)
    if not best:
        return []

    score_by_name = {names[i]: float(scores[i]) for i in range(len(names))}

    pool = []
    for ci, _sim in best:
        for nm in cluster_index.cluster_members(ci):
            if nm in exclude:
                continue
            if nm in score_by_name:
                pool.append(nm)

    # dedup (preserve), then rank by item-level score
    pool = list(dict.fromkeys(pool))
    pool.sort(key=lambda nm: -score_by_name[nm])

    return pool[:extra_k]
