import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

BATCH_SIZE = 100


class QdrantUploader:
    def __init__(self, collection_name: str, embedding_model: str, data_file: str):
        load_dotenv()

        self.collection_name = collection_name
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.data_file = data_file
        self.data: pd.DataFrame = self._load_data()

        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=120,  # Overrides global timeout for this search. Unit is seconds.
        )
        self.client.info()

        self.embedding_model = SentenceTransformer(embedding_model)

        # Ensure collection exists and is populated
        self._ensure_collection_exists()
        self._populate_collection()

        # Check Qdrant status after loading
        self._check_qdrant_status()

    def _load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / self.data_file)

    def _ensure_collection_exists(self):
        if not self.client.collection_exists(self.collection_name):
            logging.info("Creating collection '%s'", self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE,
                ),
            )
            logging.info("Created collection '%s'", self.collection_name)
        else:
            logging.info("Collection '%s' already exists", self.collection_name)

    def _populate_collection(self):
        raise NotImplementedError("Subclasses must implement _populate_collection")

    def _check_qdrant_status(self):
        collection_info = self.client.get_collection(self.collection_name)
        logging.info(
            "Total points in collection '%s': %d",
            self.collection_name,
            collection_info.points_count,
        )
