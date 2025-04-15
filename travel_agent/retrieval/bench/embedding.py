from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from travel_agent.utils import seed_everything

MODELS = [
    "DeepPavlov/rubert-base-cased-sentence",
    "sentence-transformers/distiluse-base-multilingual-cased-v1",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]


def preprocess_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("\\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("-", "")
    text = text.replace(";", " ")
    return text


def mean_pool_sentence_embeddings(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    all_embeddings = []
    for text in tqdm(texts):
        sentences = sent_tokenize(text, language="russian")
        sentence_embeddings = model.encode(sentences, convert_to_numpy=True, batch_size=64)
        mean_embedding = np.mean(sentence_embeddings, axis=0)
        all_embeddings.append(mean_embedding)
    return np.array(all_embeddings)


def average_precision_at_k(relevant_list: list[int], k: int) -> float:
    score = 0.0
    num_hits = 0
    for i, rel in enumerate(relevant_list[:k]):
        if rel:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / min(k, sum(relevant_list)) if sum(relevant_list) > 0 else 0.0


def generate_embeddings(df: pd.DataFrame, doc_col: str, embedding_col: str, model: SentenceTransformer) -> pd.DataFrame:
    if doc_col not in df.columns:
        logger.error(f"DataFrame must contain '{doc_col}' column")
        raise ValueError(f"DataFrame must contain '{doc_col}' column")

    if embedding_col in df.columns:
        logger.info(f"Embedding column {embedding_col} already exists, skipping")

    else:
        logger.info(f"Generating embeddings for {doc_col} and saving to {embedding_col}")
        doc_embeddings = mean_pool_sentence_embeddings(df[doc_col].tolist(), model)
        df[embedding_col] = list(doc_embeddings)
    return df


def benchmark_similarity(
    df: pd.DataFrame,
    doc_col: str,
    query_col: str,
    embedding_col: str,
    model: SentenceTransformer,
    ks: list[int] = [10],
) -> dict[int, float]:
    if doc_col not in df.columns or query_col not in df.columns or embedding_col not in df.columns:
        logger.error(f"DataFrame must contain '{doc_col}', '{query_col}' and '{embedding_col}' columns")
        raise ValueError(f"DataFrame must contain '{doc_col}', '{query_col}' and '{embedding_col}' columns")

    doc_embeddings = np.array(df[embedding_col].tolist())
    results = {}

    for k in ks:
        ap_scores = []

        for query in df[query_col].unique():
            logger.debug(f"Processing query: {query}")
            query_embedding = model.encode([query], convert_to_numpy=True)

            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

            top_k_indices = np.argsort(similarities)[::-1][:k]
            top_k_types = df.iloc[top_k_indices][query_col].tolist()

            for i in range(min(3, k)):
                logger.debug(f"Top {i + 1} doc for query '{query}': {df.iloc[top_k_indices][doc_col].tolist()[i]}")

            relevant_list = [1 if t == query else 0 for t in top_k_types]
            ap_at_k = average_precision_at_k(relevant_list, k)
            ap_scores.append(ap_at_k)

        map_k = np.mean(ap_scores)
        results[k] = map_k

    return results


if __name__ == "__main__":
    import sys

    logger.remove()
    logger.add(sys.stdout, level="INFO")

    seed = 42
    seed_everything(seed)

    doc_col = "text"
    # query_col = "question"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = [1, 3, 5, 10, 20]

    dataset_name = "moskva"

    embeddings_path = Path("data") / "embedding" / f"{dataset_name}-with-embeddings.parquet"
    if embeddings_path.exists():
        logger.info(f"Loading embeddings from {str(embeddings_path)}")
        df = pd.read_parquet(embeddings_path)
    else:
        logger.info(f"Existing {str(embeddings_path)} not found, using raw")
        df_path = Path("data") / "prepared" / f"{dataset_name}.csv"
        df = pd.read_csv(df_path)
        df[doc_col] = df[doc_col].apply(preprocess_text)
        # df[query_col] = df[query_col].apply(preprocess_text)

    # results = defaultdict(dict)

    for model_name in MODELS:
        logger.info(f"Generating embeddings using {model_name}")
        model = SentenceTransformer(model_name, device=device)
        embedding_col = f"{doc_col}_{model_name}"
        df = generate_embeddings(df=df, doc_col=doc_col, embedding_col=embedding_col, model=model)

        logger.info(f"Benchmarking model: {model_name}")

        # map_k = benchmark_similarity(
        #     df=df,
        #     doc_col=doc_col,
        #     query_col=query_col,
        #     embedding_col=embedding_col,
        #     model=model,
        #     ks=k,
        # )

        # for k_val, score in map_k.items():
        #     results[model_name][f"map@{k_val}"] = score
        torch.cuda.empty_cache()

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(embeddings_path, index=False)

    # results_df = pd.DataFrame.from_dict(results, orient="index").reset_index()
    # results_df = results_df.rename(columns={"index": "model"})
    # results_path = Path("data") / "embedding" / "bench_results.csv"
    # results_path.parent.mkdir(parents=True, exist_ok=True)
    # results_df.to_csv(results_path, index=False)
