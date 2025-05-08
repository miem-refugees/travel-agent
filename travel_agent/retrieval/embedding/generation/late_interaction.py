from fastembed import LateInteractionTextEmbedding
import numpy as np
from tqdm import tqdm
import numpy as np
from loguru import logger
import torch

COLBERT_MODEL_NAME = "jinaai/jina-colbert-v2"
BATCH_SIZE = 4
CUDA = True if torch.cuda.is_available() else False
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL_NAME, cuda=CUDA)


def get_colbert_embedding_dim():
    return query_embed_colbert("Test query").shape[1]


def query_embed_colbert(query: str) -> np.ndarray:
    return next(colbert_model.query_embed(query))


def generate_colbert_embeddings(docs: list[str]) -> list[np.ndarray]:
    logger.info(f"Generating embeddings using {COLBERT_MODEL_NAME}")
    return [
        embedding
        for embedding in tqdm(
            colbert_model.embed(docs, batch_size=BATCH_SIZE),
            total=len(docs),
            desc="Batches",
        )
    ]
