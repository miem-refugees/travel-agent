from qdrant_client import models
from sentence_transformers import SentenceTransformer

from travel_agent.qdrant.client import create_client
from travel_agent.retrieval.embedding.generation.dense import MODELS_PROMPTS, embed_dense
from travel_agent.retrieval.embedding.generation.sparse import BM25_MODEL_NAME, query_embed_bm25


def get_search_results(
    query: str,
    dense_model_1: SentenceTransformer,
    dense_model_2: SentenceTransformer,
    prompt_1: str,
    prompt_2: str,
    limit: int,
    timeout: int = 500,
):
    sparse_embedding = query_embed_bm25(query)
    embedding_1 = embed_dense(dense_model_1, sentences=query, prompt=prompt_1)
    embedding_2 = embed_dense(dense_model_2, sentences=query, prompt=prompt_2)

    search_result = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=embedding_1,
                using=model_1_name,
                limit=limit * 2,
            ),
            models.Prefetch(
                query=embedding_2,
                using=model_2_name,
                limit=limit * 2,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_embedding.as_object()),
                using=BM25_MODEL_NAME,
                limit=1000,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        timeout=timeout,
    )
    return search_result


if __name__ == "__main__":
    client = create_client()

    device = "cpu"
    model_1_name = "sergeyzh/BERTA"
    model_2_name = "intfloat/multilingual-e5-small"
    collection_name = "yandex_full_dataset"

    dense_model_1 = SentenceTransformer(model_1_name, device=device)
    dense_model_2 = SentenceTransformer(model_2_name, device=device)

    prompt_1 = MODELS_PROMPTS[model_1_name].get("query")
    prompt_2 = MODELS_PROMPTS[model_2_name].get("query")

    limit = 20
    query = "Японские ресторан"
    results = get_search_results(query, dense_model_1, dense_model_2, prompt_1, prompt_2, limit)

    print(results)
    print(len(results.points))
