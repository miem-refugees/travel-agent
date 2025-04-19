from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from smolagents import Tool


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
        embed_model: SentenceTransformer,
        qdrant_client: QdrantClient,
        collection_name: str,
        retrieve_limit: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.collection_name = collection_name
        self.client = qdrant_client
        self.embedder = embed_model
        self.retrieve_limit = retrieve_limit

        # sanity checks
        if not self.client.collection_exists(self.collection_name):
            raise Exception(f"Collection f{self.collection_name} does not exist in qdrant")

        collection_info = self.client.get_collection(self.collection_name)
        if collection_info.vectors_count == 0:
            raise Exception(f"Collection f{self.collection_name} is empty")

    def forward(self, query: str) -> str:
        query_embedding = self.embedder.encode(query, normalize_embeddings=True)
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
            results += f"Отзыв: {point.payload['text']}\n\n"

        return results
