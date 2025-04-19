import gc
import sys
import time
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from loguru import logger
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from travel_agent.retrieval.embedding.embedding_generation import (
    generate_embeddings,
    preprocess_text,
)
from travel_agent.utils import seed_everything

nltk.download("stopwords")


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


def average_precision_at_k(relevant_list: list[int], k: int) -> float:
    score = 0.0
    num_hits = 0
    for i, rel in enumerate(relevant_list[:k]):
        if rel:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / min(k, sum(relevant_list)) if sum(relevant_list) > 0 else 0.0


def benchmark_similarity(
    df: pd.DataFrame,
    doc_col: str,
    query_col: str,
    embedding_col: str,
    model: SentenceTransformer,
    prompt: str,
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
            logger.debug(f"Processing query: {query} and prompt: {prompt}")
            query_embedding = model.encode([query], convert_to_numpy=True, prompt=prompt)

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


def benchmark_tfidf_similarity(
    df: pd.DataFrame,
    doc_col: str,
    query_col: str,
    ks: list[int] = [10],
):
    logger.info("Benchmarking model: TFIDF")
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words=stopwords.words("russian"))

    start_embed = time.time()
    doc_embeddings = tfidf_vectorizer.fit_transform(df[doc_col].to_list())
    end_embed = time.time()
    embedding_duration = end_embed - start_embed

    start_bench = time.time()
    map_results = {}

    for k in ks:
        ap_scores = []
        for query in df[query_col].unique():
            query_embedding = tfidf_vectorizer.transform([query])

            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

            top_k_indices = np.argsort(similarities)[::-1][:k]
            top_k_types = df.iloc[top_k_indices][query_col].tolist()

            for i in range(min(3, k)):
                logger.debug(f"Top {i + 1} doc for query '{query}': {df.iloc[top_k_indices][doc_col].tolist()[i]}")

            relevant_list = [1 if t == query else 0 for t in top_k_types]
            ap_at_k = average_precision_at_k(relevant_list, k)
            ap_scores.append(ap_at_k)

        map_k = np.mean(ap_scores)
        map_results[k] = map_k

    end_bench = time.time()
    benchmark_duration = end_bench - start_bench

    results = {}
    for k_val, score in map_results.items():
        results[f"map@{k_val}"] = score

    results["embedding_duration_sec"] = embedding_duration
    results["benchmark_duration_sec"] = benchmark_duration
    results["total_duration_sec"] = embedding_duration + benchmark_duration

    return results


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    seed = 42
    seed_everything(seed)

    doc_col = "text"
    query_col = "question"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = [1, 3, 5, 10, 20]

    embedding_bench_path = Path("data") / "embedding_bench"
    embedding_bench_dataset_path = embedding_bench_path / "normal_rubrics_15886_exploded.parquet"
    dataset_name = embedding_bench_dataset_path.stem
    embeddings_path = embedding_bench_path / f"embeddings_{dataset_name}.parquet"

    if embeddings_path.exists():
        logger.info(f"Loading embeddings from {str(embeddings_path)}")
        df = pd.read_parquet(embeddings_path)
    else:
        logger.info(f"Existing {str(embeddings_path)} not found, using raw")
        df = pd.read_parquet(embedding_bench_dataset_path)
        df[doc_col] = df[doc_col].apply(preprocess_text)
        df[query_col] = df[query_col].apply(preprocess_text)

    results = {}

    for model_name in MODELS_PROMPTS:
        logger.info(f"Generating embeddings using {model_name}")

        model = SentenceTransformer(model_name, device=device)
        embedding_col = f"{doc_col}_{model_name}"

        start_embed = time.time()
        df = generate_embeddings(
            df=df,
            doc_col=doc_col,
            embedding_col=embedding_col,
            model=model,
            prompt=MODELS_PROMPTS[model_name]["passage"],
        )
        end_embed = time.time()
        embedding_duration = end_embed - start_embed

        logger.info(f"Benchmarking model: {model_name}")

        start_benchmark = time.time()
        map_k = benchmark_similarity(
            df=df,
            doc_col=doc_col,
            query_col=query_col,
            embedding_col=embedding_col,
            model=model,
            prompt=MODELS_PROMPTS[model_name]["query"],
            ks=k,
        )
        end_benchmark = time.time()
        benchmark_duration = end_benchmark - start_benchmark

        if model_name not in results:
            results[model_name] = {}

        for k_val, score in map_k.items():
            results[model_name][f"map@{k_val}"] = score

        results[model_name]["embedding_duration_sec"] = embedding_duration
        results[model_name]["benchmark_duration_sec"] = benchmark_duration
        results[model_name]["total_duration_sec"] = embedding_duration + benchmark_duration

        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    results["tfidf"] = benchmark_tfidf_similarity(df, query_col, doc_col, ks=k)

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(embeddings_path, index=False)

    results_df = pd.DataFrame.from_dict(results, orient="index").reset_index()
    results_df = results_df.rename(columns={"index": "model"})
    results_path = embedding_bench_path / f"{dataset_name}_results.csv"
    results_df.to_csv(results_path, index=False)
