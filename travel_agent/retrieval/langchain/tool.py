from typing import List, Optional

import torch
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from travel_agent.retrieval.common.rubrics import ALL_RUBRICS
from travel_agent.retrieval.embedding.generation.st import MODELS_PROMPTS


class GetExistingAvailableRubricsSchema(BaseModel):
    """Schema for getting available travel review rubrics."""

    pass


class GetExistingAvailableRubricsTool(BaseTool):
    name: str = "get_existing_travel_review_rubrics"
    description: str = 'Получение возможных значений рубрик. Использовать если нужно вызвать утилиту "travel_review_query" с аргументом "rubrics"'
    args_schema: type[BaseModel] = GetExistingAvailableRubricsSchema

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[str]:
        """Get all available rubrics."""
        return ALL_RUBRICS[:50]


class TravelReviewQuerySchema(BaseModel):
    """Schema for querying travel reviews."""

    query: str = Field(
        description="Запрос для поиска отзыва. Должен быть семантически близок к искомым отзывам, например: посоветуй хорошую кофейню в Москве."
    )
    min_rating: Optional[int] = Field(
        None,
        description="Опциональное поле, фильтр минимального рейтинга (от 1 до 5)",
    )
    address: Optional[str] = Field(
        None,
        description='Опциональное поле, ключевое слово из адреса (город или улица), например: "Москва", "улица Энгельса"',
    )
    rubrics: Optional[str] = Field(
        None,
        description='Опциональное поле, использовать его только если не получилось получить подходящих результатов по параметру "query". '
        + "Аргумент можно получить из утилиты get_existing_travel_review_rubrics",
    )


class TravelReviewQueryTool(BaseTool):
    name: str = "travel_review_query"
    description: str = (
        "Использует семантический поиск для извлечения отзывов на различные заведения. "
        + 'Важно: не упоминайте весь текст отзыва в ответе. Разрешено использовать только смысл содержания, например: "Посетители отмечают, что в заведении чисто и комфортно"'
    )
    args_schema: type[BaseModel] = TravelReviewQuerySchema
    return_direct: bool = True

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        # Set these attributes as instance attributes (not model fields)
        instance._collection_name = None
        instance._client = None
        instance._embedder = None
        instance._embed_prompt = None
        instance._retrieve_limit = None
        return instance

    def __init__(
        self,
        embed_model_name: str,
        qdrant_client: QdrantClient,
        collection_name: str,
        retrieve_limit: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._collection_name = collection_name
        self._client = qdrant_client
        self._retrieve_limit = retrieve_limit

        device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        logger.info("Using device: {}", device)

        if embed_model_name not in MODELS_PROMPTS.keys():
            raise Exception(f"Model {embed_model_name} is not supported in MODELS_PROMPTS")

        self._embedder = SentenceTransformer(embed_model_name, device=device)
        self._embed_prompt = MODELS_PROMPTS[embed_model_name]["query"]

        # sanity checks
        if not self._client.collection_exists(self._collection_name):
            raise Exception(f"Collection {self._collection_name} does not exist in qdrant")

        collection_info = self._client.get_collection(self._collection_name)
        if collection_info.vectors_count == 0:
            raise Exception(f"Collection {self._collection_name} is empty")

    def _run(
        self,
        query: str,
        min_rating: Optional[int] = None,
        address: Optional[str] = None,
        rubrics: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[str]:
        """Search for travel reviews based on query and filters."""
        query_embedding = self._embedder.encode(
            query,
            prompt=self._embed_prompt,
        )

        filters = []
        if min_rating:
            filters.append(models.FieldCondition(key="rating", range=models.Range(gte=min_rating)))
        if address:
            filters.append(models.FieldCondition(key="address", match=models.MatchText(value=address)))
        if rubrics:
            filters.append(models.FieldCondition(key="rubrics", match=models.MatchValue(value=rubrics)))

        points = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=self._retrieve_limit,
            query_filter=models.Filter(must=filters),
        )

        if not points:
            return "По вашему запросу ничего не найдено."

        return [
            (
                f"=== Отзыв на {point.payload['name']} ===\n"
                f"Адрес: {point.payload['address']}\n"
                f"Рейтинг: {point.payload['rating']}\n"
                f"Категории: {point.payload['rubrics']}\n"
                f"Текст: {point.payload['text']}\n\n"
            )
            for point in points
        ]
