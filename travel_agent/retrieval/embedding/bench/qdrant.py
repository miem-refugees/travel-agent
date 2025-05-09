import gc

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

import numpy as np
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from travel_agent.retrieval.embedding.utils import average_precision_at_k
from travel_agent.retrieval.embedding.generation.dense import embed_dense
from pathlib import Path

import numpy as np
import pandas as pd
from fastembed import SparseEmbedding, SparseTextEmbedding
from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

from travel_agent.retrieval.embedding.utils import average_precision_at_k

# from travel_agent.retrieval.embedding.generation.dense import preprocess_text
from travel_agent.utils import seed_everything

from typing import Callable

from collections import defaultdict
from travel_agent.retrieval.embedding.utils import clean_up_model


from travel_agent.retrieval.embedding.generation.sparse import (
    query_embed_bm25,
    BM25_MODEL_NAME,
)


import numpy as np
import torch
from fastembed import LateInteractionTextEmbedding
from loguru import logger
from tqdm import tqdm
from travel_agent.retrieval.embedding.generation.late_interaction import (
    COLBERT_MODEL_NAME,
    query_embed_colbert,
)


def qdrant_evaluate_queries(
    queries: list[str],
    search_results_fn: Callable,
    query_payload_key: str,
    ks: list[int],
) -> dict[int, float]:
    ap_scores_by_k = defaultdict(list)

    for query in queries:
        search_result = search_results_fn(query)
        top_types = [point.payload[query_payload_key] for point in search_result.points]
        for k in ks:
            top_k_types = top_types[:k]
            relevant_list = [1 if t == query else 0 for t in top_k_types]
            ap_at_k = average_precision_at_k(relevant_list, k)
            ap_scores_by_k[k].append(ap_at_k)

    return {k: np.mean(ap_scores_by_k[k]) for k in ks}


def qdrant_single_dense_benchmark(
    client: QdrantClient,
    collection_name: str,
    model_name: str,
    device: str,
    prompt: str,
    queries: list[str],
    query_payload_key: str,
    ks: list[int] = [10],
) -> dict[str, float]:
    model = SentenceTransformer(model_name, device=device)

    def get_search_results(query):
        embedding = embed_dense(model, sentences=query, prompt=prompt)
        search_result = client.query_points(
            collection_name, query=embedding, using=model_name, limit=max(ks)
        )
        return search_result

    results = qdrant_evaluate_queries(
        queries, get_search_results, query_payload_key, ks
    )
    clean_up_model(model, device)
    return results


def qdrant_bm25_benchmark(
    client: QdrantClient,
    collection_name: str,
    queries: list[str],
    query_payload_key: str,
    ks: list[int] = [10],
) -> dict[int, float]:
    def get_search_results(query):
        embedding = query_embed_bm25(query)
        search_result = client.query_points(
            collection_name,
            query=models.SparseVector(**embedding.as_object()),
            limit=max(ks),
            using=BM25_MODEL_NAME,
        )
        return search_result

    return qdrant_evaluate_queries(queries, get_search_results, query_payload_key, ks)


def qdrant_colbert_benchmark(
    client: QdrantClient,
    collection_name: str,
    queries: list[str],
    query_payload_key: str,
    ks: list[int] = [10],
) -> dict[int, float]:
    model = LateInteractionTextEmbedding(COLBERT_MODEL_NAME)

    def get_search_results(query):
        embedding = query_embed_colbert(model, query)
        search_result = client.query_points(
            collection_name, query=embedding, using=COLBERT_MODEL_NAME, limit=max(ks)
        )
        return search_result

    return qdrant_evaluate_queries(queries, get_search_results, query_payload_key, ks)


def qdrant_triple_model_reranking_benchmark(
    client: QdrantClient,
    collection_name: str,
    model_name: str,
    device: str,
    prompt: str,
    queries: list[str],
    query_payload_key: str,
    ks: list[int] = [10],
) -> dict[int, float]:
    dense_model = SentenceTransformer(model_name, device=device)
    colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL_NAME)

    def get_search_results(query):
        late_embedding = query_embed_colbert(colbert_model, query)
        dense_embedding = embed_dense(dense_model, sentences=query, prompt=prompt)
        sparse_embedding = query_embed_bm25(query)

        prefetch = [
            models.Prefetch(
                query=dense_embedding,
                using=model_name,
                limit=max(ks) * 2,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_embedding.as_object()),
                using=BM25_MODEL_NAME,
                limit=max(ks) * 2,
            ),
        ]

        search_result = client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=late_embedding,
            using=COLBERT_MODEL_NAME,
            limit=max(ks),
        )
        return search_result

    results = qdrant_evaluate_queries(
        queries, get_search_results, query_payload_key, ks
    )

    clean_up_model(dense_model, device)

    return results


def qdrant_bm25_1000_then_dense_benchmark(
    client: QdrantClient,
    collection_name: str,
    model_name: str,
    device: str,
    prompt: str,
    queries: list[str],
    query_payload_key: str,
    ks: list[int] = [10],
) -> dict[int, float]:
    model = SentenceTransformer(model_name, device=device)

    def get_search_results(query):
        dense_embedding = embed_dense(model, sentences=query, prompt=prompt)
        sparse_embedding = query_embed_bm25(query)

        search_result = client.query_points(
            collection_name=collection_name,
            prefetch=models.Prefetch(
                query=models.SparseVector(**sparse_embedding.as_object()),
                using=BM25_MODEL_NAME,
                limit=1000,
            ),
            query=dense_embedding,
            using=model_name,
            limit=max(ks),
        )
        return search_result

    results = qdrant_evaluate_queries(
        queries, get_search_results, query_payload_key, ks
    )
    clean_up_model(model, device)
    return results
