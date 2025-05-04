import time

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from travel_agent.retrieval.embedding.bench.utils import (
    apply_stemmer,
    average_precision_at_k,
    stop_words,
)


def benchmark_tfidf_similarity(
    df: pd.DataFrame,
    doc_col: str,
    query_col: str,
    ks: list[int] = [10],
):
    logger.info("Benchmarking model: TF-IDF")
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words=stop_words)

    start_embed = time.time()
    docs = df[doc_col].to_list()
    docs = [apply_stemmer(doc) for doc in docs]
    doc_embeddings = tfidf_vectorizer.fit_transform(docs)
    end_embed = time.time()
    embedding_duration = end_embed - start_embed

    start_bench = time.time()
    map_results = {}

    for k in ks:
        ap_scores = []
        for query in df[query_col].unique():
            query_embedding = tfidf_vectorizer.transform([apply_stemmer(query)])

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
