from qdrant_client import QdrantClient


class Qdrant:
    def __init__(self, url, *args, **kwargs):
        self.client = QdrantClient(url=url, *args, **kwargs)

        self.places_collection = "places"

    def __getattr__(self, name):
        return getattr(self.client, name)

    def add_places_collection(self, documents: list):
        self.client.add(collection=self.places_collection, documents=documents)
