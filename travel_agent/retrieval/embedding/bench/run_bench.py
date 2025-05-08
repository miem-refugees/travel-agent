from travel_agent.retrieval.embedding.generation.dense import MODELS_PROMPTS
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
from travel_agent.retrieval.embedding.bench.qdrant import qdrant_benchmark
from travel_agent.retrieval.embedding.generation.qdrant import (
    embed_and_upload_df_with_payload,
)
import torch

if __name__ == "__main__":
    seed = 42
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_bench_path = Path("data") / "embedding_bench"
    embedding_bench_dataset_path = (
        embedding_bench_path / "normal_rubrics_15886_exploded.parquet"
    )
    dataset_name = embedding_bench_dataset_path.stem
    
    doc_col = "text"
    query_col = "question"
    k = [1, 3, 5, 10, 20]

    logger.info(f"Loading dataset {dataset_name}")
    df = pd.read_parquet(embedding_bench_dataset_path).sample(2000)

    client = QdrantClient(url="http://localhost:6333")

    embed_and_upload_df_with_payload(
        client, dataset_name, df, doc_col, MODELS_PROMPTS, True, True, device
    )
    
    
