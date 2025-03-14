from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from smolagents import Tool


class ReviewQueryTool(Tool):
    name = "reviews_query"
    description = "Поиск мест по отзывам на Яндекс картах"
    inputs = {
        "query": {
            "type": "string",
            "description": "Запрос, похожий на отзыв пользователя",
        }
    }
    output_type = "string"

    def __init__(
        self, qdrand_client: QdrantClient, model_name: str, review_max_size=50, **kwargs
    ):
        super().__init__(**kwargs)
        self.collection_name = "reviews"
        self.client = qdrand_client
        self.review_max_size = review_max_size

        if not self.client.collection_exists(self.collection_name):
            self.client.recover_snapshot(
                collection_name=self.collection_name,
                location="https://snapshots.qdrant.io/imdb-1000-jina.snapshot",
            )
        self.embedder = TextEmbedding(model_name=model_name)

    def _cut_text(self, text: str):
        if self.review_max_size is not None and len(text) > self.review_max_size:
            text = text[: self.review_max_size] + "..."

        return text

    def forward(self, query: str) -> str:
        points = self.client.query_points(
            self.collection_name, query=next(self.embedder.query_embed(query)), limit=5
        ).points
        docs = "Отзывы пользователей:\n" + "".join(
            [
                f"== Отзыв на {point.payload['Name']} ==\n"
                + f"Тип: {point.payload['Rubrics']}\n"
                + f"Адрес: {point.payload['Address']}\n"
                + f"Рейтинг: {point.payload['Rating']}\n"
                + f"Текст отзыва: {self._cut_text(point.payload['Text'])}\n"
                for i, point in enumerate(points)
            ]
        )

        return docs
