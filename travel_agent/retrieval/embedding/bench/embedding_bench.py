import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from travel_agent.retrieval.embedding.bench.tf_idf import benchmark_tfidf_similarity
from travel_agent.retrieval.embedding.bench.utils import average_precision_at_k
from travel_agent.retrieval.embedding.generation.dense import MODELS_PROMPTS, generate_embeddings, preprocess_text
from travel_agent.utils import seed_everything


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


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    seed = 42
    seed_everything(seed)

    doc_col = "text"
    query_col = "question"
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        if device.lower().startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    results["tfidf"] = benchmark_tfidf_similarity(df, query_col, doc_col, ks=k)

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(embeddings_path, index=False)

    results_df = pd.DataFrame.from_dict(results, orient="index").reset_index()
    results_df = results_df.rename(columns={"index": "model"})
    results_path = embedding_bench_path / f"{dataset_name}_results_123.csv"
    results_df.to_csv(results_path, index=False)
