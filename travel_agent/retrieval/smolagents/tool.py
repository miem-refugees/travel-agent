from typing import List, Optional

import torch
from loguru import logger
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from smolagents import Tool
from travel_agent.retrieval.common.rubrics import ALL_RUBRICS

DEFAULT_LIMIT = 20


class GetExistingAvailableRubricsTool(Tool):
    name = "get_available_rubrics"
    description = f'Получение возможных значений рубрик. Использовать если нужно вызвать утилиту "travel_review_query" с аргументом "rubrics". Default limit = {DEFAULT_LIMIT}'
    inputs = {
        "offset": {
            "type": "integer",
            "description": "Отступ. Передать, если недостаточно полученных данных.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, offset: Optional[int] = 0):
        return ", ".join(ALL_RUBRICS[offset:DEFAULT_LIMIT])


class TravelReviewQueryTool(Tool):
    name = "travel_review_query"
    description = (
        "Использует семантический поиск для извлечения отзывов на различные заведения. "
        "Запрещено использовать результат напрямую - пример ответа: пользователи говорят, что в заведении X хорошая атмосфера, низкие цены, много хороших отзывов."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Запрос для поиска отзыва. Должен быть семантически близок к искомым отзывам, например: посоветуй хорошую кофейню в Москве.",
        },
        "min_rating": {
            "type": "integer",
            "description": "Опциональное поле, фильтр минимального рейтинга (от 1 до 5)",
            "nullable": True,
        },
        "address": {
            "type": "string",
            "description": 'Опциональное поле, ключевое слово из адреса (город или улица), например: "Москва" или "улица Энгельса"',
            "nullable": True,
        },
        "rubrics": {
            "type": "array",
            "description": 'Опциональное поле, использовать его только если не получилось получить подходящих результатов по параметру "query". '
            + f"Аргумент можно получить из утилиты {GetExistingAvailableRubricsTool.name}",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        embed_model_name: str,
        qdrant_client: QdrantClient,
        collection_name: str,
        retrieve_limit: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.collection_name = collection_name
        self.client = qdrant_client

        device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        logger.info("Using device: {}", device)

        self.embedder = SentenceTransformer(embed_model_name, device=device)
        self.embed_prompt = "query: "
        self.retrieve_limit = retrieve_limit

        # sanity checks
        if not self.client.collection_exists(self.collection_name):
            raise Exception(f"Collection f{self.collection_name} does not exist in qdrant")

        collection_info = self.client.get_collection(self.collection_name)
        if collection_info.vectors_count == 0:
            raise Exception(f"Collection f{self.collection_name} is empty")

    def forward(
        self,
        query: str,
        min_rating: Optional[int] = None,
        address: Optional[str] = None,
        rubrics: Optional[List[str]] = None,
    ):
        query_embedding = self.embedder.encode(
            query,
            prompt=self.embed_prompt,
        )

        filters = []

        if min_rating:
            filters.append(models.FieldCondition(key="rating", range=models.Range(gte=min_rating)))
        if address:
            filters.append(models.FieldCondition(key="address", match=models.MatchText(text=address)))
        if rubrics:
            filters.append(models.FieldCondition(key="rubrics", match=models.MatchAny(any=rubrics)))

        points = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=self.retrieve_limit,
            query_filter=models.Filter(must=filters),
        )

        if not points:
            return "По вашему запросу ничего не найдено."

        results = "Найденные отзывы о местах:\n\n"
        for i, point in enumerate(points, 1):
            results += f"=== Отзыв на {point.payload['name']} ===\n"
            results += f"Адрес: {point.payload['address']}\n"
            results += f"Рейтинг: {point.payload['rating']}\n"
            results += f"Категории: {point.payload['rubrics']}\n"
            results += f"Текст: {point.payload['text']}\n\n"

        return results
