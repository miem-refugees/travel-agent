import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from loguru import logger
from qdrant_client import QdrantClient

from travel_agent.retrieval.embedding.bench.qdrant import (
    qdrant_bm25_1000_then_colbert_benchmark,
    qdrant_bm25_1000_then_dense_benchmark,
    qdrant_bm25_benchmark,
    qdrant_colbert_benchmark,
    qdrant_hybrid_search_top_models_2_benchmark,
    qdrant_hybrid_search_top_models_2_benchmark_dbsf,
    qdrant_hybrid_search_top_models_2_rerank_benchmark,
    qdrant_hybrid_search_top_models_benchmark,
    qdrant_hybrid_search_top_models_benchmark_dbsf,
    qdrant_single_dense_benchmark,
    qdrant_triple_model_reranking_benchmark,
)
from travel_agent.retrieval.embedding.generation.dense import MODELS_PROMPTS, get_models_params_embedding_dim
from travel_agent.retrieval.embedding.generation.late_interaction import COLBERT_MODEL_NAME, get_colbert_embedding_dim
from travel_agent.retrieval.embedding.generation.sparse import BM25_MODEL_NAME
from travel_agent.utils import seed_everything


def format_num_params(num_params):
    if num_params >= 1e6:
        return f"{round(num_params / 1e6)}M"
    elif num_params >= 1e3:
        return f"{round(num_params / 1e3)}K"
    return str(num_params)


def run_and_record_benchmark(
    experiment_name: str,
    func: Callable,
    embedding_dim: str | int | float = "-",
    num_params: str | int | float = "-",
    **kwargs: Any | QdrantClient,
) -> dict[str, str | int | float]:
    start_time = time.time()
    result = func(**kwargs)
    duration = time.time() - start_time

    row: dict[str, str | int | float] = {"experiment": experiment_name}

    for k_val, score in result.items():
        row[f"map@{k_val}"] = float(score)

    row["benchmark_duration_sec"] = duration
    row["embedding_dim"] = embedding_dim
    row["num_params"] = format_num_params(num_params) if isinstance(num_params, (int, float)) else num_params
    return row


def benchmark_dense_models(
    results: list[dict[str, str | float | int]],
    client: QdrantClient,
    dataset_name: str,
    queries: list[str],
    query_col: str,
    device: str,
    ks: list[int],
) -> None:
    logger.info("Benchmarking dense...")
    model_params_embedding_dim = get_models_params_embedding_dim(MODELS_PROMPTS)
    for model_name, prompt_info in MODELS_PROMPTS.items():
        row = run_and_record_benchmark(
            experiment_name=model_name,
            func=qdrant_single_dense_benchmark,
            embedding_dim=model_params_embedding_dim[model_name]["embedding_dim"],
            num_params=model_params_embedding_dim[model_name]["num_params"],
            client=client,
            collection_name=dataset_name,
            model_name=model_name,
            device=device,
            prompt=prompt_info.get("query"),
            queries=queries,
            query_payload_key=query_col,
            ks=ks,
        )
        results.append(row)


def benchmark_bm25(
    results: list[dict[str, str | float | int]],
    client: QdrantClient,
    dataset_name: str,
    queries: list[str],
    query_col: str,
    ks: list[int],
) -> None:
    logger.info("Benchmarking bm25...")
    row = run_and_record_benchmark(
        experiment_name=BM25_MODEL_NAME,
        func=qdrant_bm25_benchmark,
        client=client,
        collection_name=dataset_name,
        queries=queries,
        query_payload_key=query_col,
        ks=ks,
    )
    results.append(row)


def benchmark_colbert(
    results: list[dict[str, str | float | int]],
    client: QdrantClient,
    dataset_name: str,
    queries: list[str],
    query_col: str,
    ks: list[int],
) -> None:
    logger.info("Benchmarking colbert...")
    colbert_params_embedding_dim = get_colbert_embedding_dim()
    row = run_and_record_benchmark(
        experiment_name=COLBERT_MODEL_NAME,
        func=qdrant_colbert_benchmark,
        client=client,
        collection_name=dataset_name,
        queries=queries,
        query_payload_key=query_col,
        ks=ks,
        embedding_dim=colbert_params_embedding_dim["embedding_dim"],
        num_params=colbert_params_embedding_dim["num_params"],
    )
    results.append(row)


def benchmark_reranking(
    results: list[dict[str, str | float | int]],
    client: QdrantClient,
    dataset_name: str,
    queries: list[str],
    query_col: str,
    device: str,
    ks: list[int],
) -> None:
    logger.info("Benchmarking reranking...")
    for model_name, prompt_info in MODELS_PROMPTS.items():
        row = run_and_record_benchmark(
            experiment_name=f"{model_name}+{BM25_MODEL_NAME}_reranking_{COLBERT_MODEL_NAME}",
            func=qdrant_triple_model_reranking_benchmark,
            client=client,
            collection_name=dataset_name,
            model_name=model_name,
            device=device,
            prompt=prompt_info.get("query"),
            queries=queries,
            query_payload_key=query_col,
            ks=ks,
        )
        results.append(row)


def benchmark_multi_stage(
    results: list[dict[str, str | float | int]],
    client: QdrantClient,
    dataset_name: str,
    queries: list[str],
    query_col: str,
    device: str,
    ks: list[int],
) -> None:
    logger.info("Benchmarking multistage...")
    for model_name, prompt_info in MODELS_PROMPTS.items():
        row = run_and_record_benchmark(
            experiment_name=f"multi_stage_1000_{BM25_MODEL_NAME}_top_k_{model_name}",
            func=qdrant_bm25_1000_then_dense_benchmark,
            client=client,
            collection_name=dataset_name,
            model_name=model_name,
            device=device,
            prompt=prompt_info.get("query"),
            queries=queries,
            query_payload_key=query_col,
            ks=ks,
        )
        results.append(row)


def benchmark_hybrid(
    results: list[dict[str, str | float | int]],
    client: QdrantClient,
    dataset_name: str,
    queries: list[str],
    query_col: str,
    ks: list[int],
) -> None:
    logger.info("Benchmarking hybrid...")
    row = run_and_record_benchmark(
        experiment_name="hybrid_search_top_models",
        func=qdrant_hybrid_search_top_models_benchmark,
        client=client,
        collection_name=dataset_name,
        queries=queries,
        query_payload_key=query_col,
        ks=ks,
    )
    results.append(row)


def benchmark_hybrid_2(
    results: list[dict[str, str | float | int]],
    client: QdrantClient,
    dataset_name: str,
    queries: list[str],
    query_col: str,
    ks: list[int],
) -> None:
    logger.info("Benchmarking hybrid_2...")
    row = run_and_record_benchmark(
        experiment_name="hybrid_search_top_models_2",
        func=qdrant_hybrid_search_top_models_2_benchmark,
        client=client,
        collection_name=dataset_name,
        queries=queries,
        query_payload_key=query_col,
        ks=ks,
    )
    results.append(row)


def benchmark_hybrid_3(
    results: list[dict[str, str | float | int]],
    client: QdrantClient,
    dataset_name: str,
    queries: list[str],
    query_col: str,
    ks: list[int],
) -> None:
    logger.info("Benchmarking hybrid_3...")
    row = run_and_record_benchmark(
        experiment_name="hybrid_search_top_models_3",
        func=qdrant_hybrid_search_top_models_2_rerank_benchmark,
        client=client,
        collection_name=dataset_name,
        queries=queries,
        query_payload_key=query_col,
        ks=ks,
    )
    results.append(row)


def benchmark_multi_stage_colbert(
    results: list[dict[str, str | float | int]],
    client: QdrantClient,
    dataset_name: str,
    queries: list[str],
    query_col: str,
    ks: list[int],
) -> None:
    logger.info("Benchmarking multistage with ColBERT...")
    colbert_params_embedding_dim = get_colbert_embedding_dim()
    row = run_and_record_benchmark(
        experiment_name=f"multi_stage_1000_{BM25_MODEL_NAME}_top_k_{COLBERT_MODEL_NAME}",
        func=qdrant_bm25_1000_then_colbert_benchmark,
        client=client,
        collection_name=dataset_name,
        queries=queries,
        query_payload_key=query_col,
        ks=ks,
        embedding_dim=colbert_params_embedding_dim["embedding_dim"],
        num_params=colbert_params_embedding_dim["num_params"],
    )
    results.append(row)


def benchmark_hybrid_dbsf(
    results: list[dict[str, str | float | int]],
    client: QdrantClient,
    dataset_name: str,
    queries: list[str],
    query_col: str,
    ks: list[int],
) -> None:
    logger.info("Benchmarking hybrid...")
    row = run_and_record_benchmark(
        experiment_name="hybrid_search_top_models_dbsf",
        func=qdrant_hybrid_search_top_models_benchmark_dbsf,
        client=client,
        collection_name=dataset_name,
        queries=queries,
        query_payload_key=query_col,
        ks=ks,
    )
    results.append(row)


def benchmark_hybrid_2_dbsf(
    results: list[dict[str, str | float | int]],
    client: QdrantClient,
    dataset_name: str,
    queries: list[str],
    query_col: str,
    ks: list[int],
) -> None:
    logger.info("Benchmarking hybrid_2...")
    row = run_and_record_benchmark(
        experiment_name="hybrid_search_top_models_2_dbsf",
        func=qdrant_hybrid_search_top_models_2_benchmark_dbsf,
        client=client,
        collection_name=dataset_name,
        queries=queries,
        query_payload_key=query_col,
        ks=ks,
    )
    results.append(row)


def main():
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_bench_path = Path("data") / "embedding_bench"
    embedding_bench_dataset_path = embedding_bench_path / "normal_rubrics_15886_exploded.parquet"
    dataset_name = embedding_bench_dataset_path.stem

    query_col = "question"
    k = [1, 3, 5, 10, 20]

    logger.info(f"Loading dataset {dataset_name}")
    df = pd.read_parquet(embedding_bench_dataset_path)
    queries = list(set(df[query_col].to_list()))
    logger.info(f"Number of unique queries: {len(queries)}")
    logger.info(f"Dataset size {len(df)}")

    client = QdrantClient(url="http://localhost:6333")

    results = []

    benchmark_dense_models(results, client, dataset_name, queries, query_col, device, k)
    benchmark_bm25(results, client, dataset_name, queries, query_col, k)
    benchmark_colbert(results, client, dataset_name, queries, query_col, k)
    benchmark_reranking(results, client, dataset_name, queries, query_col, device, k)
    benchmark_multi_stage(results, client, dataset_name, queries, query_col, device, k)
    benchmark_multi_stage_colbert(results, client, dataset_name, queries, query_col, k)
    benchmark_hybrid(results, client, dataset_name, queries, query_col, k)
    benchmark_hybrid_2(results, client, dataset_name, queries, query_col, k)
    benchmark_hybrid_3(results, client, dataset_name, queries, query_col, k)

    benchmark_hybrid_dbsf(results, client, dataset_name, queries, query_col, k)
    benchmark_hybrid_2_dbsf(results, client, dataset_name, queries, query_col, k)

    df_results = pd.DataFrame(results)
    save_path = Path("data") / "embedding_bench" / f"{dataset_name}_new_results+one_more.csv"
    df_results.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
