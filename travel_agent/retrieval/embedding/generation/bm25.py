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

bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25", language="russian")


def query_embed_bm25(query: str) -> SparseEmbedding:
    return next(bm25_model.query_embed(query))


def generate_bm25_embeddings(docs: list[str]) -> list[SparseEmbedding]:
    bm25_embeddings = list(bm25_model.embed(doc for doc in docs))
    return bm25_embeddings
