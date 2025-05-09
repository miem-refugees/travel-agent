from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from qdrant_client import QdrantClient

from travel_agent.retrieval.embedding.bench.qdrant import (
    qdrant_single_dense_benchmark,
    qdrant_bm25_benchmark,
    qdrant_colbert_benchmark,
    qdrant_triple_model_reranking_benchmark,
    qdrant_bm25_1000_then_dense_benchmark,
)
from travel_agent.retrieval.embedding.generation.dense import (
    MODELS_PROMPTS,
    get_models_params_embedding_dim,
)
from travel_agent.utils import seed_everything
import time
from travel_agent.retrieval.embedding.generation.sparse import BM25_MODEL_NAME
from travel_agent.retrieval.embedding.generation.late_interaction import (
    COLBERT_MODEL_NAME,
    get_colbert_embedding_dim,
)


def format_num_params(num_params):
    if num_params >= 1e6:
        return f"{round(num_params / 1e6)}M"
    elif num_params >= 1e3:
        return f"{round(num_params / 1e3)}K"
    return str(num_params)


def run_and_record_benchmark(
    name, func, *args, embedding_dim="-", num_params="-", **kwargs
):
    start_time = time.time()
    result = func(*args, **kwargs)
    duration = time.time() - start_time

    row = {
        "model": name,
        "benchmark_duration_sec": duration,
        "embedding_dim": embedding_dim,
        "num_params": (
            format_num_params(num_params)
            if isinstance(num_params, (int, float))
            else num_params
        ),
    }

    for k_val, score in result.items():
        row[f"map@{k_val}"] = float(score)

    return row


def benchmark_dense_models(
    results, client, dataset_name, queries, query_col, device, k, model_params
):
    for model_name, prompt_info in MODELS_PROMPTS.items():
        row = run_and_record_benchmark(
            model_name,
            qdrant_single_dense_benchmark,
            client,
            dataset_name,
            model_name,
            device,
            prompt_info["query"],
            queries,
            query_col,
            k,
            embedding_dim=model_params[model_name]["embedding_dim"],
            num_params=model_params[model_name]["num_params"],
        )
        results.append(row)


def benchmark_bm25(results, client, dataset_name, queries, query_col, k):
    row = run_and_record_benchmark(
        BM25_MODEL_NAME,
        qdrant_bm25_benchmark,
        client,
        dataset_name,
        queries,
        query_col,
        ks=k,
    )
    results.append(row)


def benchmark_colbert(
    results, client, dataset_name, queries, query_col, k, colbert_params
):
    row = run_and_record_benchmark(
        COLBERT_MODEL_NAME,
        qdrant_colbert_benchmark,
        client,
        dataset_name,
        queries,
        query_col,
        ks=k,
        embedding_dim=colbert_params["embedding_dim"],
        num_params=colbert_params["num_params"],
    )
    results.append(row)


def benchmark_reranking(results, client, dataset_name, queries, query_col, device, k):
    for model_name, prompt_info in MODELS_PROMPTS.items():
        row = run_and_record_benchmark(
            f"{model_name}+{BM25_MODEL_NAME}_reranking_{COLBERT_MODEL_NAME}",
            qdrant_triple_model_reranking_benchmark,
            client,
            dataset_name,
            model_name,
            device,
            prompt_info["query"],
            queries,
            query_col,
            k,
        )
        results.append(row)


def benchmark_multi_stage(results, client, dataset_name, queries, query_col, device, k):
    for model_name, prompt_info in MODELS_PROMPTS.items():
        row = run_and_record_benchmark(
            f"multi_stage_1000_{BM25_MODEL_NAME}_top_k_{model_name}",
            qdrant_bm25_1000_then_dense_benchmark,
            client,
            dataset_name,
            model_name,
            device,
            prompt_info["query"],
            queries,
            query_col,
            k,
        )
        results.append(row)


if __name__ == "__main__":
    seed = 42
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_bench_path = Path("data") / "embedding_bench"
    embedding_bench_dataset_path = (
        embedding_bench_path / "normal_rubrics_15886_exploded.parquet"
    )
    dataset_name = embedding_bench_dataset_path.stem

    doc_col = "text"
    query_col = "question"
    k = [1, 3, 5, 10, 20]

    logger.info(f"Loading dataset {dataset_name}")
    df = pd.read_parquet(embedding_bench_dataset_path).sample(300)

    client = QdrantClient(url="http://localhost:6333")

    queries = list(set(df[query_col].to_list()))

    model_params_embedding_dim = get_models_params_embedding_dim(MODELS_PROMPTS)

    results = []

    ## ALL DENSE MODELS
    for model_name in MODELS_PROMPTS:
        start_time = time.time()

        result = qdrant_single_dense_benchmark(
            client,
            dataset_name,
            model_name,
            device,
            MODELS_PROMPTS[model_name]["query"],
            queries,
            query_col,
            k,
        )

        duration = time.time() - start_time

        embedding_dim = model_params_embedding_dim[model_name]["embedding_dim"]
        num_params = model_params_embedding_dim[model_name]["num_params"]

        if num_params >= 1e6:
            num_params_str = f"{round(num_params / 1e6)}M"
        elif num_params >= 1e3:
            num_params_str = f"{round(num_params / 1e3)}K"
        else:
            num_params_str = str(num_params)

        result_row = {
            "model": model_name,
            "benchmark_duration_sec": duration,
            "embedding_dim": embedding_dim,
            "num_params": num_params_str,
        }

        for k_val, score in result.items():
            result_row[f"map@{k_val}"] = float(score)

        results.append(result_row)

    ### BM25
    start_time = time.time()
    bm25_result = qdrant_bm25_benchmark(client, dataset_name, queries, query_col, ks=k)
    duration = time.time() - start_time

    bm25_row = {
        "model": BM25_MODEL_NAME,
        "benchmark_duration_sec": duration,
        "embedding_dim": "-",
        "num_params": "-",
    }

    for k_val, score in bm25_result.items():
        bm25_row[f"map@{k_val}"] = float(score)

    results.append(bm25_row)

    ## COLBERT
    colbert_params_embedding_dim = get_colbert_embedding_dim()

    start_time = time.time()
    colbert_result = qdrant_colbert_benchmark(
        client, dataset_name, queries, query_col, ks=k
    )
    duration = time.time() - start_time

    embedding_dim = colbert_params_embedding_dim["embedding_dim"]
    num_params = colbert_params_embedding_dim["num_params"]

    if num_params >= 1e6:
        num_params_str = f"{round(num_params / 1e6)}M"
    elif num_params >= 1e3:
        num_params_str = f"{round(num_params / 1e3)}K"
    else:
        num_params_str = str(num_params)

    colbert_row = {
        "model": COLBERT_MODEL_NAME,
        "benchmark_duration_sec": duration,
        "embedding_dim": embedding_dim,
        "num_params": num_params,
    }

    for k_val, score in colbert_result.items():
        colbert_row[f"map@{k_val}"] = float(score)

    results.append(colbert_row)

    ## ALL DENSE MODELS RERANKING

    for model_name in MODELS_PROMPTS:
        start_time = time.time()

        result = qdrant_triple_model_reranking_benchmark(
            client,
            dataset_name,
            model_name,
            device,
            MODELS_PROMPTS[model_name]["query"],
            queries,
            query_col,
            k,
        )

        duration = time.time() - start_time

        result_row = {
            "model": f"{model_name}+{BM25_MODEL_NAME}_reranking_{COLBERT_MODEL_NAME}",
            "benchmark_duration_sec": duration,
            "embedding_dim": "-",
            "num_params": "-",
        }

        for k_val, score in result.items():
            result_row[f"map@{k_val}"] = float(score)

        results.append(result_row)

    ## 1000 BM25 and then dense

    for model_name in MODELS_PROMPTS:
        start_time = time.time()

        result = qdrant_bm25_1000_then_dense_benchmark(
            client,
            dataset_name,
            model_name,
            device,
            MODELS_PROMPTS[model_name]["query"],
            queries,
            query_col,
            k,
        )

        duration = time.time() - start_time

        result_row = {
            "model": f"multi_stage_1000_{BM25_MODEL_NAME}_top_k_{model_name}",
            "benchmark_duration_sec": duration,
            "embedding_dim": "-",
            "num_params": "-",
        }

        for k_val, score in result.items():
            result_row[f"map@{k_val}"] = float(score)

        results.append(result_row)

    df_results = pd.DataFrame(results)

    print(df_results)
