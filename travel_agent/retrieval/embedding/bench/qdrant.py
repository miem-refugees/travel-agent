from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from travel_agent.retrieval.embedding.bench.tf_idf import benchmark_tfidf_similarity
from travel_agent.retrieval.embedding.bench.utils import (
    average_precision_at_k,
)
from travel_agent.retrieval.embedding.generation.st import (
    MODELS_PROMPTS,
    generate_embeddings,
    preprocess_text,
)
from travel_agent.utils import seed_everything

from fastembed import SparseTextEmbedding, SparseEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from typing import Callable

# client.query_points(
#             collection_name=collection_name,
#             query=models.SparseVector(**sparse_vectors.as_object()),
#             limit=max(ks),
#             using="bm25",
#         )


def qdrant_benchmark(
    queries: list[str],
    query_payload_key: str,
    embed_query: Callable,
    qdrant_search: Callable,
    ks: list[int] = [10],
) -> dict[int, float]:
    results = {}
    for query in queries:
        logger.debug(f"Processing query: {query}")
        embedding = embed_query(query)
        search_result = qdrant_search(embedding)
        top_types = []
        for point in search_result.points:
            top_types.append(point.payload[query_payload_key])
        ap_scores = []
        for k in ks:
            top_k_types = top_types[:k]
            relevant_list = [1 if t == query else 0 for t in top_k_types]
            ap_at_k = average_precision_at_k(relevant_list, k)
            ap_scores.append(ap_at_k)

            map_k = np.mean(ap_scores)
            results[k] = map_k
    return results
