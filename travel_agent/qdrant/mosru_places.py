import json
import logging
import os

import pandas as pd

from dotenv import load_dotenv
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

BATCH_SIZE = 100
MOSRU_COLLECTION = "mosru_places"


class MosruPlaces:
    def __init__(self, embedding_model: str):
        load_dotenv()

        self.collection_name = MOSRU_COLLECTION

        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.data: pd.DataFrame = self._load_data()

        self.client = QdrantClient(url=os.getenv("QDRANT_URL"), timeout=120)
        self.client.info()

        self.embedding_model = SentenceTransformer(embedding_model)

        # Ensure collection exists and is populated
        self._ensure_collection_exists()
        self._populate_collection()

        # Check Qdrant status after loading
        self._check_qdrant_status()

    def _load_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.data_dir / "merged_mosru.csv")
        return data

    def _ensure_collection_exists(self):
        """Create the Qdrant collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            logging.info(
                "creating collection '%s' as it does not exist", self.collection_name
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE,
                ),
            )
            logging.info("created collection '%s'", self.collection_name)
        else:
            logging.info("collection '%s' already exists", self.collection_name)

    def _populate_collection(self):
        logging.info("Encoding objects of '%s' ", self.collection_name)

        embeddings = self.embedding_model.encode(self.data["Name"].tolist()).tolist()
        logging.info(
            "Populating '%s' with %d embeddings...",
            self.collection_name,
            len(embeddings),
        )

        points = []
        for i in range(len(self.data)):
            points.append(
                PointStruct(
                    id=i,
                    vector=embeddings[i],
                    payload={
                        "Name": self.data.loc[i, "Name"],
                        "geoData": json.loads(
                            self.data.loc[i, "geoData"].replace("'", '"')
                        ),
                        "District": self.data.loc[i, "District"],
                        "Address": self.data.loc[i, "Address"],
                        "type": self.data.loc[i, "type"],
                    },
                )
            )

            if len(points) % BATCH_SIZE == 0 or i == len(self.data) - 1:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                points = []

        logging.info(
            "Populated collection '%s' with %d points",
            self.collection_name,
            len(self.data),
        )

    def _check_qdrant_status(self):
        collection_info = self.client.get_collection(self.collection_name)
        logging.info("Total points in collection: %d", collection_info.points_count)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    MosruPlaces("intfloat/multilingual-e5-large-instruct")
