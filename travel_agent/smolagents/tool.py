import os
import time
from typing import List, Optional
from urllib.parse import quote

from loguru import logger
from smolagents import Tool

from travel_agent.qdrant.reviews_searcher import QdrantReviewsSearcher
from travel_agent.retrieval.common.rubrics import ALL_RUBRICS

DEFAULT_LIMIT = 30


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
        "Прикладывай ссылки на карту из ответа утилиты как есть."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Запрос для поиска отзыва. Должен быть семантически близок к искомым отзывам, например: посоветуй хорошую кофейню в Москве.",
        },
        "min_rating": {
            "type": "integer",
            "description": "Опционально. Фильтр минимального рейтинга (от 1 до 5)",
            "nullable": True,
        },
        "address": {
            "type": "string",
            "description": 'Опционально. Название улицы или локации, например: "Улица Энгельса", "Болотная набережная, 15".',
            "nullable": True,
        },
        "region": {
            "type": "string",
            "description": 'Опционально. Название города, если известен город или региона России, например: "Рязань", "Тверская область", "Краснодарский край".',
            "nullable": True,
        },
        "rubrics": {
            "type": "array",
            "description": f'Опционально. Аргумент можно получить из утилиты {GetExistingAvailableRubricsTool.name}. Использовать его только если не получилось получить подходящих результатов по параметру "query"',
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        retrieve_limit: int = 10,
        timeout: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.searcher = QdrantReviewsSearcher(
            retrieve_limit=retrieve_limit,
            timeout=timeout,
            snapshot_url=os.getenv("QDRANT_SHAPSHOT"),
        )

    def forward(
        self,
        query: str,
        min_rating: Optional[int] = None,
        address: Optional[str] = None,
        region: Optional[str] = None,
        rubrics: Optional[List[str]] = None,
    ):
        start = time.time()
        points = self.searcher.query(
            query=query,
            min_rating=min_rating,
            address=address,
            region=region,
            rubrics=rubrics,
        )

        logger.debug("retrieved {} points in {} sec", len(points), (time.time() - start))

        if not points:
            return "По запросу ничего не найдено."

        results = "Найденные отзывы:\n\n"
        for i, point in enumerate(points, 1):
            name = point.payload.get("name_ru")
            link = (
                f"https://yandex.ru/maps/213/moscow/search/{quote(name)}"
                if point.payload.get("region") == "Москва"
                else f"https://yandex.ru/maps/?text={quote(name)}"
            )

            results += f"== Отзыв на [{name}]({link}) ==\n"
            results += f"Адрес: {point.payload.get('address')}\n"
            results += f"Регион: {point.payload.get('region')}\n"
            results += f"Рейтинг: {point.payload.get('rating')}\n"
            results += f"Категории: {point.payload.get('rubrics')}\n"
            results += f"Текст: {point.payload.get('text')}\n\n"

        return results
