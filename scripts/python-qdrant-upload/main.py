import argparse
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from tqdm import tqdm

from travel_agent.qdrant import client as qdrant_client


def load_dataframe(file_path):
    try:
        logger.info("Loading dataframe from {}", file_path)
        df = pd.read_parquet(file_path)
        logger.success("Successfully loaded dataframe with {} rows and {} columns", len(df), len(df.columns))
        return df
    except Exception as e:
        logger.error("Failed to load dataframe: {}", e)
        raise


def create_collection(client, collection_name, vector_size):
    try:
        client.get_collection(collection_name)
        logger.info("Collection '{}' already exists", collection_name)

        collection_info = client.get_collection(collection_name)
        existing_size = collection_info.config.params.vectors.size
        if existing_size != vector_size:
            logger.warning("Existing collection has vector size {}, but input has size {}", existing_size, vector_size)

        return False

    except UnexpectedResponse:
        logger.info("Creating collection '{}' with vector size {}", collection_name, vector_size)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        logger.success("Created new collection '{}'", collection_name)
        return True
    except Exception as e:
        logger.error("Error checking/creating collection '{}': {}", collection_name, e)
        raise


def upload_data_to_collection(client, df, collection_name, embedding_column, batch_size=100):
    logger.info("Processing data for collection '{}' using embedding column '{}'", collection_name, embedding_column)

    sample_vector = None
    for _, row in df.iterrows():
        if isinstance(row[embedding_column], (list, np.ndarray)) and (
            (isinstance(row[embedding_column], list) and len(row[embedding_column]) > 0)
            or (isinstance(row[embedding_column], np.ndarray) and row[embedding_column].size > 0)
        ):
            sample_vector = row[embedding_column]
            break

    if sample_vector is None:
        logger.error("No valid embeddings found in column '{}'", embedding_column)
        return False

    vector_size = len(sample_vector)
    logger.info("Vector size determined: {}", vector_size)

    create_collection(client, collection_name, vector_size)

    points = []
    processed = 0
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Uploading to {collection_name}"):
        if not isinstance(row[embedding_column], (list, np.ndarray)) or (
            (isinstance(row[embedding_column], list) and len(row[embedding_column]) == 0)
            or (isinstance(row[embedding_column], np.ndarray) and row[embedding_column].size == 0)
        ):
            skipped += 1
            continue

        vector = (
            row[embedding_column].tolist() if isinstance(row[embedding_column], np.ndarray) else row[embedding_column]
        )

        try:
            point = models.PointStruct(
                id=idx,
                vector=vector,
                payload={
                    "address": str(row["address"]) if pd.notna(row["address"]) else "",
                    "name": str(row["name_ru"]) if pd.notna(row["name_ru"]) else "",
                    "rating": float(row["rating"]) if pd.notna(row["rating"]) else 0.0,
                    "rubrics": str(row["rubrics"]) if pd.notna(row["rubrics"]) else "",
                    "text": str(row["text"]) if pd.notna(row["text"]) else "",
                },
            )
            points.append(point)
            processed += 1

            if len(points) >= batch_size:
                try:
                    client.upsert(collection_name=collection_name, points=points)
                    points = []
                except Exception as e:
                    logger.error("Error uploading batch to collection '{}': {}", collection_name, e)
        except Exception as e:
            logger.error("Error creating point for index {}: {}", idx, e)
            skipped += 1

    if points:
        try:
            client.upsert(collection_name=collection_name, points=points)
        except Exception as e:
            logger.error("Error uploading final batch to collection '{}': {}", collection_name, e)

    logger.success("Uploaded {} points to '{}', skipped {} invalid embeddings", processed, collection_name, skipped)
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload embeddings from parquet file to Qdrant")
    parser.add_argument("--dataset", required=True, help="Path to parquet file with embeddings")
    args = parser.parse_args()

    load_dotenv()

    logger.info("Will use dataset: {}", args.dataset)

    try:
        df = load_dataframe(args.dataset)

        embedding_columns = [col for col in df.columns if col.startswith("text_")]
        logger.info("Found {} embedding models to process", len(embedding_columns))

        for embed_col in embedding_columns:
            model_name = embed_col.replace("text_", "").replace("/", "_").replace("-", "_")
            collection_name = f"{os.getenv('COLLECTION_PREFIX', 'moskva')}_{model_name}"

            upload_data_to_collection(
                client=qdrant_client,
                df=df,
                collection_name=collection_name,
                embedding_column=embed_col,
                batch_size=int(os.getenv("BATCH_SIZE", 100)),
            )

            logger.info("Completed processing for '{}'", collection_name)

        logger.success("All collections have been processed successfully")

    except Exception as e:
        logger.error("An error occurred during processing: {}", e)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
