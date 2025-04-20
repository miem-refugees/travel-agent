import torch
from loguru import logger
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from smolagents import Tool

from travel_agent.retrieval.embedding.embedding_generation import MODELS_PROMPTS


class TravelReviewQueryTool(Tool):
    name = "travel_review_query"
    description = "Использует семантический поиск для извлечения отзывов о местах из коллекции Qdrant."
    inputs = {
        "query": {
            "type": "string",
            "description": "Запрос для поиска. Должен быть семантически близок к искомым отзывам.",
        }
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

        if embed_model_name not in MODELS_PROMPTS.keys():
            raise Exception(f"Model f{embed_model_name} is not supported in MODELS_PROMPTS")

        self.embedder = SentenceTransformer(embed_model_name, device=device)
        self.embed_prompt = MODELS_PROMPTS[embed_model_name]["query"]
        self.retrieve_limit = retrieve_limit

        # sanity checks
        if not self.client.collection_exists(self.collection_name):
            raise Exception(f"Collection f{self.collection_name} does not exist in qdrant")

        collection_info = self.client.get_collection(self.collection_name)
        if collection_info.vectors_count == 0:
            raise Exception(f"Collection f{self.collection_name} is empty")

    def forward(self, query: str) -> str:
        query_embedding = self.embedder.encode(
            query,
            prompt=self.embed_prompt,
        )
        points = self.client.search(
            collection_name=self.collection_name, query_vector=query_embedding, limit=self.retrieve_limit
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
