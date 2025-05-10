# ## ALL DENSE MODELS
# for model_name in MODELS_PROMPTS:
#     start_time = time.time()

#     result = qdrant_single_dense_benchmark(
#         client,
#         dataset_name,
#         model_name,
#         device,
#         MODELS_PROMPTS[model_name]["query"],
#         queries,
#         query_col,
#         k,
#     )

#     duration = time.time() - start_time

#     embedding_dim = model_params_embedding_dim[model_name]["embedding_dim"]
#     num_params = model_params_embedding_dim[model_name]["num_params"]

#     if num_params >= 1e6:
#         num_params_str = f"{round(num_params / 1e6)}M"
#     elif num_params >= 1e3:
#         num_params_str = f"{round(num_params / 1e3)}K"
#     else:
#         num_params_str = str(num_params)

#     result_row = {
#         "model": model_name,
#         "benchmark_duration_sec": duration,
#         "embedding_dim": embedding_dim,
#         "num_params": num_params_str,
#     }

#     for k_val, score in result.items():
#         result_row[f"map@{k_val}"] = float(score)

#     results.append(result_row)

#     ### BM25
#     start_time = time.time()
#     bm25_result = qdrant_bm25_benchmark(client, dataset_name, queries, query_col, ks=k)
#     duration = time.time() - start_time

#     bm25_row = {
#         "model": BM25_MODEL_NAME,
#         "benchmark_duration_sec": duration,
#         "embedding_dim": "-",
#         "num_params": "-",
#     }

#     for k_val, score in bm25_result.items():
#         bm25_row[f"map@{k_val}"] = float(score)

#     results.append(bm25_row)

#     ## COLBERT
#     colbert_params_embedding_dim = get_colbert_embedding_dim()

#     start_time = time.time()
#     colbert_result = qdrant_colbert_benchmark(client, dataset_name, queries, query_col, ks=k)
#     duration = time.time() - start_time

#     embedding_dim = colbert_params_embedding_dim["embedding_dim"]
#     num_params = colbert_params_embedding_dim["num_params"]

#     if num_params >= 1e6:
#         num_params_str = f"{round(num_params / 1e6)}M"
#     elif num_params >= 1e3:
#         num_params_str = f"{round(num_params / 1e3)}K"
#     else:
#         num_params_str = str(num_params)

#     colbert_row = {
#         "model": COLBERT_MODEL_NAME,
#         "benchmark_duration_sec": duration,
#         "embedding_dim": embedding_dim,
#         "num_params": num_params,
#     }

#     for k_val, score in colbert_result.items():
#         colbert_row[f"map@{k_val}"] = float(score)

#     results.append(colbert_row)

#     ## ALL DENSE MODELS RERANKING

#     for model_name in MODELS_PROMPTS:
#         start_time = time.time()

#         result = qdrant_triple_model_reranking_benchmark(
#             client,
#             dataset_name,
#             model_name,
#             device,
#             MODELS_PROMPTS[model_name]["query"],
#             queries,
#             query_col,
#             k,
#         )

#         duration = time.time() - start_time

#         result_row = {
#             "model": f"{model_name}+{BM25_MODEL_NAME}_reranking_{COLBERT_MODEL_NAME}",
#             "benchmark_duration_sec": duration,
#             "embedding_dim": "-",
#             "num_params": "-",
#         }

#         for k_val, score in result.items():
#             result_row[f"map@{k_val}"] = float(score)

#         results.append(result_row)

#     ## 1000 BM25 and then dense

#     for model_name in MODELS_PROMPTS:
#         start_time = time.time()

#         result = qdrant_bm25_1000_then_dense_benchmark(
#             client,
#             dataset_name,
#             model_name,
#             device,
#             MODELS_PROMPTS[model_name]["query"],
#             queries,
#             query_col,
#             k,
#         )

#         duration = time.time() - start_time

#         result_row = {
#             "model": f"multi_stage_1000_{BM25_MODEL_NAME}_top_k_{model_name}",
#             "benchmark_duration_sec": duration,
#             "embedding_dim": "-",
#             "num_params": "-",
#         }

#         for k_val, score in result.items():
#             result_row[f"map@{k_val}"] = float(score)

#         results.append(result_row)


# if __name__ == "__main__":
#     seed = 42
#     seed_everything(seed)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     embedding_bench_path = Path("data") / "embedding_bench"
#     embedding_bench_dataset_path = embedding_bench_path / "normal_rubrics_15886_exploded.parquet"
#     dataset_name = embedding_bench_dataset_path.stem

#     doc_col = "text"
#     query_col = "question"
#     k = [1, 3, 5, 10, 20]

#     logger.info(f"Loading dataset {dataset_name}")
#     df = pd.read_parquet(embedding_bench_dataset_path).sample(300)

#     client = QdrantClient(url="http://localhost:6333")

#     queries = list(set(df[query_col].to_list()))

#     results = []

#     df_results = pd.DataFrame(results)

#     print(df_results)
