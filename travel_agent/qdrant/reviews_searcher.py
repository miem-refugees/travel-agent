from typing import List, Optional

import torch
from loguru import logger
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from travel_agent.qdrant import client
from travel_agent.retrieval.embedding.generation.dense import MODELS_PROMPTS, embed_dense
from travel_agent.retrieval.embedding.generation.sparse import BM25_MODEL_NAME, query_embed_bm25


class QdrantReviewsSearcher:
    def __init__(
        self,
        collection_name: str = "yandex_full_dataset",
        model_1_name: str = "sergeyzh/BERTA",
        model_2_name: str = "intfloat/multilingual-e5-small",
        retrieve_limit: int = 5,
        timeout: int = 500,
    ):
        self.collection_name = collection_name
        self.model_1_name = model_1_name
        self.model_2_name = model_2_name
        self.retrieve_limit = retrieve_limit
        self.client = client
        self.timeout = timeout

        device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        logger.debug("torch will use device: {}", device)

        logger.info("loading model {}", model_1_name)
        self.dense_model_1 = SentenceTransformer(model_1_name, device=device)

        logger.info("loading model {}", model_2_name)
        self.dense_model_2 = SentenceTransformer(model_2_name, device=device)

        self.prompt_1 = MODELS_PROMPTS[model_1_name].get("query")
        self.prompt_2 = MODELS_PROMPTS[model_2_name].get("query")

        # sanity check
        if not self.client.collection_exists(self.collection_name):
            raise Exception(f"Collection f{self.collection_name} does not exist in qdrant")

        collection_info = self.client.get_collection(self.collection_name)
        if collection_info.vectors_count == 0:
            raise Exception(f"Collection f{self.collection_name} is empty")

        logger.success("QdrantReviewsSearcher is ready")

    def query(
        self,
        query: str,
        min_rating: Optional[int] = None,
        address: Optional[str] = None,
        region: Optional[str] = None,
        rubrics: Optional[List[str]] = None,
    ) -> models.QueryResponse:
        filters = []

        if min_rating:
            filters.append(models.FieldCondition(key="rating", range=models.Range(gte=min_rating)))
        if address:
            filters.append(models.FieldCondition(key="address", match=models.MatchText(text=address)))
        if region:
            filters.append(models.FieldCondition(key="region", match=models.MatchText(text=region)))
        if rubrics:
            filters.append(models.FieldCondition(key="rubrics", match=models.MatchAny(any=rubrics)))

        sparse_embedding = query_embed_bm25(query)
        embedding_1 = embed_dense(self.dense_model_1, sentences=query, prompt=self.prompt_1)
        embedding_2 = embed_dense(self.dense_model_2, sentences=query, prompt=self.prompt_2)

        result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=embedding_1,
                    using=self.model_1_name,
                    limit=self.retrieve_limit * 2,
                ),
                models.Prefetch(
                    query=embedding_2,
                    using=self.model_2_name,
                    limit=self.retrieve_limit * 2,
                ),
                models.Prefetch(
                    query=models.SparseVector(**sparse_embedding.as_object()),
                    using=BM25_MODEL_NAME,
                    limit=self.retrieve_limit * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.DBSF),
            query_filter=models.Filter(must=filters),
            limit=self.retrieve_limit,
            timeout=self.timeout,
        )

        return result.points
