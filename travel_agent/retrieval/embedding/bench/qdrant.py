from collections import defaultdict
from typing import Callable

import numpy as np
from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from travel_agent.retrieval.embedding.generation.dense import MODELS_PROMPTS, embed_dense
from travel_agent.retrieval.embedding.generation.late_interaction import COLBERT_MODEL_NAME, query_embed_colbert
from travel_agent.retrieval.embedding.generation.sparse import BM25_MODEL_NAME, query_embed_bm25
from travel_agent.retrieval.embedding.utils import average_precision_at_k, clean_up_model


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
) -> dict[int, float]:
    model = SentenceTransformer(model_name, device=device)

    def get_search_results(query):
        embedding = embed_dense(model, sentences=query, prompt=prompt)
        search_result = client.query_points(collection_name, query=embedding, using=model_name, limit=max(ks))
        return search_result

    results = qdrant_evaluate_queries(queries, get_search_results, query_payload_key, ks)
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
        search_result = client.query_points(collection_name, query=embedding, using=COLBERT_MODEL_NAME, limit=max(ks))
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

    results = qdrant_evaluate_queries(queries, get_search_results, query_payload_key, ks)

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

    results = qdrant_evaluate_queries(queries, get_search_results, query_payload_key, ks)
    clean_up_model(model, device)
    return results


def qdrant_hybrid_search_top_models_benchmark(
    client: QdrantClient,
    collection_name: str,
    queries: list[str],
    query_payload_key: str,
    ks: list[int] = [10],
) -> dict[int, float]:
    device = "cpu"
    model_1_name = "sergeyzh/BERTA"
    model_2_name = "intfloat/multilingual-e5-large"
    model_3_name = "ai-forever/ru-en-RoSBERTa"
    model_4_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    model_1 = SentenceTransformer(model_1_name, device=device)
    model_2 = SentenceTransformer(model_2_name, device=device)
    model_3 = SentenceTransformer(model_3_name, device=device)
    model_4 = SentenceTransformer(model_4_name, device=device)

    def get_search_results(query):
        embedding_1 = embed_dense(model_1, sentences=query, prompt=MODELS_PROMPTS[model_1_name].get("query"))
        embedding_2 = embed_dense(model_2, sentences=query, prompt=MODELS_PROMPTS[model_2_name].get("query"))
        embedding_3 = embed_dense(model_3, sentences=query, prompt=MODELS_PROMPTS[model_3_name].get("query"))
        embedding_4 = embed_dense(model_4, sentences=query, prompt=MODELS_PROMPTS[model_4_name].get("query"))

        search_result = client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=embedding_1,
                    using=model_1_name,
                    limit=20,
                ),
                models.Prefetch(
                    query=embedding_2,
                    using=model_2_name,
                    limit=20,
                ),
                models.Prefetch(
                    query=embedding_3,
                    using=model_3_name,
                    limit=20,
                ),
                models.Prefetch(
                    query=embedding_4,
                    using=model_4_name,
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
        )
        return search_result

    results = qdrant_evaluate_queries(queries, get_search_results, query_payload_key, ks)
    clean_up_model(model_1, device)
    clean_up_model(model_2, device)
    clean_up_model(model_3, device)
    clean_up_model(model_4, device)
    return results
