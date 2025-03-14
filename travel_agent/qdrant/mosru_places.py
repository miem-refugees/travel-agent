import json
import logging

from qdrant_client.models import PointStruct

from travel_agent.qdrant.qdrant import BATCH_SIZE, QdrantUploader


class MosruPlacesUploader(QdrantUploader):
    def __init__(self, embedding_model: str):
        super().__init__("mosru_places", embedding_model, "merged_mosru.csv")

    def _extract_first_geo_location(self, geo_data_str):
        try:
            geo_data = json.loads(geo_data_str.replace("'", '"'))
            if "coordinates" in geo_data and isinstance(geo_data["coordinates"], list):
                first_location = geo_data["coordinates"][0]
                return {"lat": first_location[1], "lon": first_location[0]}
        except Exception as e:
            logging.error("Error parsing geoData: %s", e)
        return None

    def _populate_collection(self):
        logging.info("Encoding objects for collection '%s'", self.collection_name)
        embeddings = self.embedding_model.encode(self.data["Name"].tolist()).tolist()

        points = []
        for i in range(len(self.data)):
            payload = {
                "name": self.data.loc[i, "Name"],
                "district": self.data.loc[i, "District"],
                "address": self.data.loc[i, "Address"],
                "type": self.data.loc[i, "type"],
            }

            location = self._extract_first_geo_location(self.data.loc[i, "geoData"])
            if location is not None:
                payload["location"] = location

            points.append(
                PointStruct(
                    id=i,
                    vector=embeddings[i],
                    payload=payload,
                )
            )
            if len(points) % BATCH_SIZE == 0 or i == len(self.data) - 1:
                self.client.upsert(collection_name=self.collection_name, points=points)
                points = []

        logging.info(
            "Populated collection '%s' with %d points",
            self.collection_name,
            len(self.data),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    MosruPlacesUploader("intfloat/multilingual-e5-large-instruct")
