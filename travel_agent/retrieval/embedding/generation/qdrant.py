from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastembed import SparseEmbedding
from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from tqdm import trange

from travel_agent.retrieval.embedding.generation.dense import (
    MODELS_PROMPTS,
    generate_dense_models_embeddings,
    get_models_params_embedding_dim,
)
from travel_agent.retrieval.embedding.generation.late_interaction import (
    COLBERT_MODEL_NAME,
    generate_colbert_embeddings,
    get_colbert_embedding_dim,
)
from travel_agent.retrieval.embedding.generation.sparse import (
    BM25_MODEL_NAME,
    generate_bm25_embeddings,
)
from travel_agent.retrieval.embedding.utils import preprocess_text
from travel_agent.utils import seed_everything


def create_collection(
    client: QdrantClient,
    collection_name: str,
    vectors_config: Optional[dict[str, models.VectorParams]] = None,
    sparse_vectors_config: Optional[dict[str, models.SparseVectorParams]] = None,
) -> None:
    if client.collection_exists(collection_name=collection_name):
        logger.info(f"Collection {collection_name} exists, deleting...")
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )
    logger.info(f"Created collection {collection_name}")


def get_vectors_config(
    dense_models: dict[str, int] = {}, late_interaction_models: dict[str, int] = {}
) -> dict[str, models.VectorParams]:
    vectors_confg = {}
    for model_name, embedding_dim in dense_models.items():
        vectors_confg[model_name] = models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)
    for model_name, embedding_dim in late_interaction_models.items():
        vectors_confg[model_name] = models.VectorParams(
            size=embedding_dim,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
        )
    logger.info(f"Vectors config: {vectors_confg}")
    return vectors_confg


def get_sparse_vectors_config(
    bm25: bool = True,
) -> dict[str, models.SparseVectorParams] | None:
    if bm25:
        return {BM25_MODEL_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)}


def create_payload_index(client: QdrantClient, collection_name: str, field_name_schema: dict[str, str]) -> None:
    for field_name, field_schema in field_name_schema.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )


def upload_embeddings(
    client: QdrantClient,
    collection_name: str,
    payload_df: pd.DataFrame,
    dense_embeddings: dict[str, list[np.ndarray]] = {},
    late_interaction_embeddings: dict[str, list[np.ndarray]] = {},
    sparse_embeddings: dict[str, list[SparseEmbedding]] = {},
):
    num_points = len(payload_df)

    for idx in trange(num_points):
        vector = {}
        for model_name, embeddings in dense_embeddings.items():
            vector[model_name] = embeddings[idx]
        for model_name, embeddings in sparse_embeddings.items():
            vector[model_name] = models.SparseVector(**embeddings[idx].as_object())
        for model_name, embeddings in late_interaction_embeddings.items():
            vector[model_name] = embeddings[idx]

        payload = {col: payload_df.iloc[idx][col] for col in payload_df.columns}

        point = PointStruct(id=idx, vector=vector, payload=payload)

        client.upsert(collection_name=collection_name, points=[point])
    logger.info("Inserted embeddings into Qdrant")


def embed_and_upload_df_with_payload(
    client: QdrantClient,
    collection_name: str,
    payload_df: pd.DataFrame,
    field_name_schema: dict[str, str],
    doc_col: str,
    dense_models_prompts: dict[str, dict[str, Optional[str]]],
    late_interaction_model: bool,
    bm25: bool,
    device: str,
) -> None:
    model_params_embedding_dim = get_models_params_embedding_dim(dense_models_prompts)

    dense_models = {}
    for model_name in dense_models_prompts:
        dense_models[model_name] = model_params_embedding_dim[model_name]["embedding_dim"]

    if late_interaction_model:
        late_interaction_models = {COLBERT_MODEL_NAME: get_colbert_embedding_dim()["embedding_dim"]}
    else:
        late_interaction_models = {}

    vectors_config = get_vectors_config(dense_models, late_interaction_models)
    sparse_vectors_config = get_sparse_vectors_config(bm25=bm25)
    create_collection(client, collection_name, vectors_config, sparse_vectors_config)
    create_payload_index(client, collection_name, field_name_schema)

    docs = payload_df[doc_col].to_list()

    if dense_models_prompts:
        dense_embeddings = generate_dense_models_embeddings(docs, dense_models_prompts, device)
    else:
        dense_embeddings = {}

    if late_interaction_model:
        late_interaction_embeddings = {COLBERT_MODEL_NAME: generate_colbert_embeddings(docs)}
    else:
        late_interaction_embeddings = {}

    if bm25:
        sparse_embeddings = {BM25_MODEL_NAME: generate_bm25_embeddings(docs)}
    else:
        sparse_embeddings = {}

    upload_embeddings(
        client,
        collection_name,
        payload_df,
        dense_embeddings,
        late_interaction_embeddings,
        sparse_embeddings,
    )


if __name__ == "__main__":
    seed = 42
    seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    doc_col = "text"
    dataset_path = Path("data") / "embedding_bench" / "normal_rubrics_15886_exploded.parquet"
    dataset_name = dataset_path.stem

    client = QdrantClient(url="http://localhost:6333")

    df = pd.read_parquet(dataset_path)

    df[doc_col] = df[doc_col].apply(preprocess_text)

    if "address" in df.columns:
        df["address"] = df["address"].fillna("")
    if "question" in df.columns:
        df["question"] = df["question"].fillna("")
    if "name" in df.columns:
        df["name"] = df["name"].fillna("")
    if "rating" in df.columns:
        df["rating"] = df["rating"].fillna("0").astype(float)
    if "rubrics" in df.columns:
        df["rubrics"] = df["rubrics"].apply(lambda x: [rubric for rubric in str(x).split(";") if pd.notna(x)])
    if "text" in df.columns:
        df["text"] = df["text"].fillna("")

    field_name_schema = {
        "name": "text",
        "address": "text",
        "rubrics": "keyword",
        "rating": "float",
        "question": "text",
        "text": "text",
    }

    embed_and_upload_df_with_payload(
        client,
        dataset_name,
        df,
        field_name_schema,
        doc_col,
        MODELS_PROMPTS,
        True,
        True,
        device,
    )
