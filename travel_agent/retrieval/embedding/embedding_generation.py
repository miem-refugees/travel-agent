import gc
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from travel_agent.retrieval.embedding.utils import seed_everything

MODELS_PROMPTS = {
    "cointegrated/rubert-tiny2": {"query": None, "passage": None},
    "DeepPavlov/rubert-base-cased-sentence": {"query": None, "passage": None},
    "ai-forever/sbert_large_nlu_ru": {"query": None, "passage": None},
    "ai-forever/sbert_large_mt_nlu_ru": {"query": None, "passage": None},
    "sentence-transformers/distiluse-base-multilingual-cased-v1": {
        "query": None,
        "passage": None,
    },
    "sentence-transformers/distiluse-base-multilingual-cased-v2": {
        "query": None,
        "passage": None,
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "query": None,
        "passage": None,
    },
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "query": None,
        "passage": None,
    },
    "intfloat/multilingual-e5-large": {"query": "query: ", "passage": "passage: "},
    "intfloat/multilingual-e5-base": {"query": "query: ", "passage": "passage: "},
    "intfloat/multilingual-e5-small": {"query": "query: ", "passage": "passage: "},
    "ai-forever/ru-en-RoSBERTa": {
        "query": "search_query: ",
        "passage": "search_document: ",
    },
    # "ai-forever/FRIDA": {"query": "search_query: ", "passage": "search_document: "},
    "sergeyzh/BERTA": {"query": "search_query: ", "passage": "search_document: "},
}


def get_dynamic_batch_size(model: SentenceTransformer) -> int:
    num_params = sum(p.numel() for p in model._first_module().auto_model.parameters())

    if num_params > 500_000_000:
        return 8
    if num_params > 200_000_000:
        return 32
    else:
        return 64


def embed(model: SentenceTransformer, prompt: str) -> np.ndarray:
    return model.encode(
        df[doc_col].to_list(),
        batch_size=get_dynamic_batch_size(model),
        prompt=prompt,
        show_progress_bar=True,
    )


def preprocess_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("\\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("-", "")
    text = text.replace(";", " ")
    return text


def mean_pool_sentence_embeddings(texts: list[str], model: SentenceTransformer, prompt: Optional[str]) -> np.ndarray:
    all_embeddings = []
    for text in tqdm(texts):
        sentences = sent_tokenize(text, language="russian")
        sentence_embeddings = model.encode(sentences, convert_to_numpy=True, batch_size=32, prompt=prompt)
        mean_embedding = np.mean(sentence_embeddings, axis=0)
        all_embeddings.append(mean_embedding)
    return np.array(all_embeddings)


def generate_embeddings(
    df: pd.DataFrame,
    doc_col: str,
    embedding_col: str,
    model: SentenceTransformer,
    prompt: Optional[str],
) -> pd.DataFrame:
    if doc_col not in df.columns:
        logger.error(f"DataFrame must contain '{doc_col}' column")
        raise ValueError(f"DataFrame must contain '{doc_col}' column")

    if embedding_col in df.columns:
        logger.info(f"Embedding column {embedding_col} already exists, skipping")

    else:
        logger.info(f"Generating embeddings for {doc_col} and saving to {embedding_col} column")
        doc_embeddings = model.encode(
            df[doc_col].to_list(),
            batch_size=get_dynamic_batch_size(model),
            prompt=prompt,
            show_progress_bar=True,
        )
        # doc_embeddings = mean_pool_sentence_embeddings(df[doc_col].tolist(), model)
        df[embedding_col] = list(doc_embeddings)
    return df


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    seed = 42
    seed_everything(seed)

    doc_col = "text"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = Path("data") / "prepared" / "sankt-peterburg.csv"
    dataset_name = dataset_path.stem
    embeddings_path = Path("data") / "embedding" / f"embeddings_{dataset_name}.parquet"

    if embeddings_path.exists():
        logger.info(f"Loading embeddings from {str(embeddings_path)}")
        df = pd.read_parquet(embeddings_path)
    else:
        logger.info(f"Existing {str(embeddings_path)} not found, using raw")
        df = pd.read_csv(dataset_path)
        df[doc_col] = df[doc_col].apply(preprocess_text)

    for model_name in MODELS_PROMPTS:
        logger.info(f"Generating embeddings using {model_name}")
        model = SentenceTransformer(model_name, device=device)
        embedding_col = f"{doc_col}_{model_name}"
        df = generate_embeddings(
            df=df,
            doc_col=doc_col,
            embedding_col=embedding_col,
            model=model,
            prompt=MODELS_PROMPTS[model_name].get("passage"),
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(embeddings_path, index=False)
