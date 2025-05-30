import os

from loguru import logger
from qdrant_client import QdrantClient


def create_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY", None)

    try:
        logger.debug("Connecting to Qdrant at {}", url)

        client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=os.getenv("QDRANT_TIMEOUT", 3600),
        )

        # ping server
        info = client.info()

        logger.success("connected: {}", info)
        return client

    except Exception as e:
        logger.error("Failed to connect to Qdrant: {}", e)
        raise
