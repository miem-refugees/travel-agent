import logging

from tqdm import tqdm

from qdrant_client.models import PointStruct

from travel_agent.qdrant.qdrant import BATCH_SIZE, QdrantUploader


class ReviewsUploader(QdrantUploader):
    def __init__(self, embedding_model: str):
        super().__init__("reviews", embedding_model, "geo-reviews-dataset-2023.csv")

    def _populate_collection(self):
        logging.info("Encoding reviews for collection '%s'", self.collection_name)

        total_points = len(self.data)

        for start in tqdm(range(0, total_points, BATCH_SIZE)):
            end = min(start + BATCH_SIZE, total_points)
            batch_texts = self.data["text"][start:end].tolist()

            batch_embeddings = self.embedding_model.encode(batch_texts).tolist()

            points = []
            for i, embedding in enumerate(batch_embeddings):
                index = start + i
                points.append(
                    PointStruct(
                        id=index,
                        vector=embedding,
                        payload={
                            "Name": self.data.loc[i, "name_ru"],
                            "Address": self.data.loc[i, "address"],
                            "Rating": self.data.loc[i, "rating"],
                            "Rubrics": self.data.loc[i, "rubrics"],
                            "Text": self.data.loc[i, "text"],
                        },
                    )
                )

            self.client.upsert(collection_name=self.collection_name, points=points)

        logging.info(
            "Populated collection '%s' with %d points",
            self.collection_name,
            len(self.data),
        )


if __name__ == "__main__":
    print(
        "Dataset is too large to run locally. Use colab: https://colab.research.google.com/drive/1zF8VMa_EIGTeHef5WOg6aBom5UmhMmYW?usp=sharing"
    )

    # logging.basicConfig(
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #     level=logging.INFO,
    # )

    # ReviewsUploader("intfloat/multilingual-e5-large-instruct")
