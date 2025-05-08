# client.query_points(
#             collection_name=collection_name,
#             query=models.SparseVector(**sparse_vectors.as_object()),
#             limit=max(ks),
#             using="bm25",
#         )
# {
# "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
# },
from typing import Any, Optional

from loguru import logger
from qdrant_client import QdrantClient, models
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from travel_agent.utils import seed_everything
from travel_agent.retrieval.embedding.generation.st import (
    MODELS_PROMPTS,
    get_models_params_embedding_dim,
)
from tqdm import tqdm, trange
from pathlib import Path

import numpy as np
import pandas as pd
from fastembed import SparseEmbedding, SparseTextEmbedding
from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

from travel_agent.retrieval.embedding.bench.utils import (
    average_precision_at_k,
)
from travel_agent.retrieval.embedding.generation.st import (
    preprocess_text,
)
from travel_agent.utils import seed_everything


def create_collection(
    client: QdrantClient,
    collection_name: str,
    vectors_config: Optional[dict[str, models.VectorParams]],
    sparse_vectors_config: Optional[dict[str, models.SparseVectorParams]],
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
    dense_models: dict[str, int], multivector_models: dict[str, int]
) -> dict[str, models.VectorParams]:
    vectors_confg = {}
    for model_name, embedding_dim in dense_models.items():
        vectors_confg[model_name] = models.VectorParams(
            size=embedding_dim, distance=models.Distance.COSINE
        )
    for model_name, embedding_dim in multivector_models.items():
        vectors_confg[model_name] = models.VectorParams(
            size=embedding_dim,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        )

    logger.info(f"Vectors config: {vectors_confg}")

    return vectors_confg


def get_sparse_vectors_config(
    bm25: bool = True,
) -> dict[str, models.SparseVectorParams] | None:
    if bm25:
        return {"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)}


def upload_embeddings(
    client: QdrantClient,
    collection_name: str,
    dense_embeddings: dict[str, list[np.ndarray]],
    sparse_embeddings: dict[str, list[SparseEmbedding]],
    payload_df: pd.DataFrame,
):

    num_points = len(payload_df)
    
    # print(num_points, len(dense_embeddings))
    
    # if dense_embeddings:
    #     assert num_points == len(dense_embeddings)
    # if sparse_embeddings:
    #     assert num_points == len(sparse_embeddings)


    for idx in trange(num_points):
        vector = {}

        for model_name, embeddings in dense_embeddings.items():
            vector[model_name] = embeddings[idx]
        for model_name, embeddings in sparse_embeddings.items():
            vector[model_name] = models.SparseVector(**embeddings[idx].as_object())

        payload = {col: payload_df.iloc[idx][col] for col in payload_df.columns}

        point = PointStruct(id=idx, vector=vector, payload=payload)

        client.upsert(collection_name=collection_name, points=[point])
    logger.info("Inserted embeddings into Qdrant")
from travel_agent.retrieval.embedding.generation.bm25 import generate_bm25_embeddings

def embed_and_upload_df_with_payload(client: QdrantClient, collection_name: str, df: pd.DataFrame, text_col: str, dense_models_prompts: dict[str, dict[str, Optional[str]]], multivector_models_prompts: dict, bm25=True, device:str) -> None:
    model_params_embedding_dim = get_models_params_embedding_dim(models_prompts)

    dense_models = {}
    for model_name in models_prompts:
        dense_models[model_name] = model_params_embedding_dim[model_name][
            "embedding_dim"
        ]

    multivector_models = {}

    vectors_config = get_vectors_config(dense_models, multivector_models)
    sparse_vectors_config = get_sparse_vectors_config(bm25=bm25)
    
    create_collection(client, collection_name, vectors_config, sparse_vectors_config)
    
    
    if bm25:
        sparse_embeddings = {"bm25": generate_bm25_embeddings(df[text_col].to_list())}
        
    if dense_models_prompts:
        
    
    
    upload_embeddings(client, collection_name, text_column_dict, {}, payload_df)


if __name__ == "__main__":
    
    seed = 42
    seed_everything(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    doc_col = "text"
    dataset_path = Path("data") / "prepared" / "sankt-peterburg.csv"
    dataset_name = dataset_path.stem
    embeddings_path = (
        Path("data") / "embedding" / f"st_embeddings_{dataset_name}.parquet"
    )
        client = QdrantClient(url="http://localhost:6333")
    
    
        test_df = pd.read_parquet("data/embedding/embeddings_sankt-peterburg.parquet")

    text_columns = [col for col in test_df.columns if col.startswith("text_")]
    basic_columns = [col for col in test_df.columns if not col.startswith("text_")]

    text_column_dict = {
        col.strip("text_"): test_df[col].tolist() for col in text_columns
    }

    payload_df = test_df[basic_columns]
    embed_and_upload_df_with_payload()