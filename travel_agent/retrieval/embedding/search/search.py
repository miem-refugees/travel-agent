from typing import Callable
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

from travel_agent.retrieval.embedding.generation.sparse import (
    generate_bm25_embeddings,
    BM25_MODEL_NAME,
)
from travel_agent.retrieval.embedding.generation.dense import (
    MODELS_PROMPTS,
    generate_dense_models_embeddings,
    get_models_params_embedding_dim,
)
from travel_agent.utils import seed_everything

embeddings = embed_query(query)
search_result = qdrant_search(client, collection_name, embeddings, max(ks))



def single_dense_query(client:QdrantClient, collection_name: str, embeddings: dict[str, np.ndarray | list[float]] | np.ndarray | list[float]) -> tuple[Callable, Callable]:
    return client.query_points(
    collection_name=collection_name
    query=vector
)