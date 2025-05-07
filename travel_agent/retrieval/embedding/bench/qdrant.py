# client.query_points(
#             collection_name=collection_name,
#             query=models.SparseVector(**sparse_vectors.as_object()),
#             limit=max(ks),
#             using="bm25",
#         )
# {
# "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
# },
from collections import defaultdict
from typing import Callable

import numpy as np

from travel_agent.retrieval.embedding.bench.utils import (
    average_precision_at_k,
)


def qdrant_benchmark(
    queries: list[str],
    query_payload_key: str,
    embed_query: Callable,
    qdrant_search: Callable,
    ks: list[int] = [10],
) -> dict[int, float]:
    ap_scores_by_k = defaultdict(list)

    for query in queries:
        embedding = embed_query(query)
        search_result = qdrant_search(embedding)
        top_types = [point.payload[query_payload_key] for point in search_result.points]

        for k in ks:
            top_k_types = top_types[:k]
            relevant_list = [1 if t == query else 0 for t in top_k_types]
            ap_at_k = average_precision_at_k(relevant_list, k)
            ap_scores_by_k[k].append(ap_at_k)

    return {k: np.mean(ap_scores_by_k[k]) for k in ks}
