import numpy as np
import torch
from fastembed import LateInteractionTextEmbedding
from loguru import logger
from tqdm import tqdm

COLBERT_MODEL_NAME = "jinaai/jina-colbert-v2"
BATCH_SIZE = 1


def get_colbert_embedding_dim() -> dict[str, int]:
    colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL_NAME)
    assert COLBERT_MODEL_NAME == "jinaai/jina-colbert-v2"
    return {
        "embedding_dim": query_embed_colbert(colbert_model, "Test query").shape[1],
        "num_params": 559_000_000,
    }


def query_embed_colbert(colbert_model: LateInteractionTextEmbedding, query: str) -> np.ndarray:
    return next(colbert_model.query_embed(query))


def generate_colbert_embeddings(docs: list[str]) -> list[np.ndarray]:
    CUDA = True if torch.cuda.is_available() else False
    colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL_NAME, cuda=CUDA)
    logger.info(f"Generating embeddings using {COLBERT_MODEL_NAME}")
    return [
        embedding
        for embedding in tqdm(
            colbert_model.embed(docs, batch_size=BATCH_SIZE),
            total=len(docs),
            desc="Batches",
        )
    ]
