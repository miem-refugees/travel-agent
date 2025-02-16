import logging

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from travel_agent.qdrant.mosru_places import MOSRU_COLLECTION


class RagSearch:
    def __init__(self, qdrant: QdrantClient, embedding_model: SentenceTransformer):
        self.qdrant = qdrant
        self.embedding_model = embedding_model

    def search_places(self, query, top_k=3):
        """Search for places in Qdrant based on a query."""

        query_embedding = self.embedding_model.encode(query).tolist()

        hits = self.qdrant.query_points(
            collection_name=MOSRU_COLLECTION,
            query=query_embedding,
            limit=top_k,
        ).points

        logging.debug("qdrant search results by %s: %v", query, hits)

        results = [
            {
                "Name": hit.payload["Name"],
                "Address": hit.payload["Address"],
                "District": hit.payload["District"],
                "Type": hit.payload["type"],
            }
            for hit in hits
        ]

        return results
