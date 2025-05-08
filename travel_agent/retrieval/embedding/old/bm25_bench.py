from pathlib import Path

import numpy as np
import pandas as pd
from fastembed import SparseEmbedding, SparseTextEmbedding
from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

from travel_agent.retrieval.embedding.bench.utils import average_precision_at_k
from travel_agent.retrieval.embedding.generation.dense import preprocess_text
from travel_agent.utils import seed_everything


def embed_using_bm25(docs: list[str]) -> list[SparseEmbedding]:
    bm25_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm25", language="russian")
    bm25_embeddings = list(bm25_embedding_model.embed(doc for doc in docs))
    return bm25_embeddings


def upload_bm25_to_qdrant(
    client: QdrantClient,
    bm25_embeddings: list[SparseEmbedding],
    df: pd.DataFrame,
    collection_name: str,
    doc_col: str,
    query_col: str,
):
    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )

    docs = df[doc_col].to_list()
    queries = df[query_col].to_list()

    points = []
    for idx, (bm25_embedding, doc, query) in enumerate(zip(bm25_embeddings, docs, queries)):
        point = PointStruct(
            id=idx,
            vector={
                "bm25": models.SparseVector(**bm25_embedding.as_object()),
            },
            payload={doc_col: doc, query_col: query},
        )
        points.append(point)

    operation_info = client.upsert(collection_name=collection_name, points=points)
    logger.info(f"Operation info {operation_info}")


def benchmark_bm25(
    client: QdrantClient,
    collection_name: str,
    df: pd.DataFrame,
    doc_col: str,
    query_col: str,
    ks: list[int] = [10],
) -> dict[int, float]:
    if doc_col not in df.columns or query_col not in df.columns:
        logger.error(f"DataFrame must contain '{doc_col}' and '{query_col}' columns")
        raise ValueError(f"DataFrame must contain '{doc_col}' and '{query_col}' columns")

    bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25", language="russian")

    results = {}

    for query in df[query_col].unique():
        logger.debug(f"Processing query: {query}")
        sparse_vectors = next(bm25_model.query_embed(query))
        search_result = client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(**sparse_vectors.as_object()),
            limit=max(ks),
            using="bm25",
        )
        top_types = []
        ids = [point.id for point in search_result.points]
        for point in search_result.points:
            top_types.append(point.payload[query_col])
        ap_scores = []
        for k in ks:
            top_k_types = top_types[:k]
            relevant_list = [1 if t == query else 0 for t in top_k_types]
            ap_at_k = average_precision_at_k(relevant_list, k)
            ap_scores.append(ap_at_k)

            map_k = np.mean(ap_scores)
            results[k] = map_k

    return results


if __name__ == "__main__":
    seed = 42
    seed_everything(seed)

    doc_col = "text"
    query_col = "question"
    k = [1, 3, 5, 10, 20]

    embedding_bench_path = Path("data") / "embedding_bench"
    embedding_bench_dataset_path = embedding_bench_path / "normal_rubrics_15886_exploded.parquet"
    dataset_name = embedding_bench_dataset_path.stem

    logger.info("Loading dataset")
    df = pd.read_parquet(embedding_bench_dataset_path)
    df[doc_col] = df[doc_col].apply(preprocess_text)
    df[query_col] = df[query_col].apply(preprocess_text)

    client = QdrantClient(path="qdrant_db")
    collection_name = "bm25_bench"

    bm25_embeddings = embed_using_bm25(df[doc_col].to_list())
    upload_bm25_to_qdrant(client, bm25_embeddings, df, collection_name, doc_col, query_col)

    results = benchmark_bm25(client, collection_name, df, doc_col, query_col, k)
