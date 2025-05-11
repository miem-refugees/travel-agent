from pathlib import Path

import pandas as pd
import torch
from loguru import logger

from travel_agent.qdrant.client import create_client
from travel_agent.retrieval.embedding.generation.dense import MODELS_PROMPTS_FINAL
from travel_agent.retrieval.embedding.generation.qdrant import embed_and_upload_df_with_payload
from travel_agent.retrieval.embedding.utils import preprocess_text
from travel_agent.utils import seed_everything

if __name__ == "__main__":
    seed = 42
    seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")

    doc_col = "text"
    dataset_path = Path("data") / "prepared" / "prepared.csv"
    dataset_name = dataset_path.stem

    client = create_client()
    # client = QdrantClient("http://localhost:6333")
    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset size {len(df)}")

    df[doc_col] = df[doc_col].apply(preprocess_text)

    df["address"] = df["address"].fillna("")
    df["name_ru"] = df["name_ru"].fillna("")
    df["rating"] = df["rating"].fillna("0").astype(float)
    df["rubrics"] = df["rubrics"].apply(lambda x: [rubric for rubric in str(x).split(";") if pd.notna(x)])
    df["text"] = df["text"].fillna("")
    df["region"] = df["region"].fillna("")

    field_name_schema = {
        "address": "text",
        "name_ru": "text",
        "rating": "float",
        "rubrics": "keyword",
        "text": "text",
        "region": "keyword",
    }

    embed_and_upload_df_with_payload(
        client=client,
        collection_name="yandex_full_dataset",
        payload_df=df,
        field_name_schema=field_name_schema,
        doc_col=doc_col,
        dense_models_prompts=MODELS_PROMPTS_FINAL,
        late_interaction_model=False,
        bm25=True,
        device=device,
    )
