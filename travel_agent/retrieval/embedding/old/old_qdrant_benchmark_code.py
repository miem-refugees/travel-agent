def qdrant_single_dense_benchmark(
    client: QdrantClient,
    collection_name: str,
    model_name: str,
    device: str,
    prompt: str,
    queries: list[str],
    query_payload_key: str,
    ks: list[int] = [10],
) -> dict[int, float]:
    ap_scores_by_k = defaultdict(list)

    model = SentenceTransformer(model_name, device=device)

    for query in queries:
        embedding = embed_dense(model, sentences=query, prompt=prompt)
        search_result = client.query_points(collection_name, query=embedding, using=model_name, limit=max(ks))
        top_types = [point.payload[query_payload_key] for point in search_result.points]

        for k in ks:
            top_k_types = top_types[:k]
            relevant_list = [1 if t == query else 0 for t in top_k_types]
            ap_at_k = average_precision_at_k(relevant_list, k)
            ap_scores_by_k[k].append(ap_at_k)

    del model
    gc.collect()
    if device.lower().startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return {k: np.mean(ap_scores_by_k[k]) for k in ks}


def qdrant_bm25_benchmark(
    client: QdrantClient,
    collection_name: str,
    queries: list[str],
    query_payload_key: str,
    ks: list[int] = [10],
) -> dict[int, float]:
    ap_scores_by_k = defaultdict(list)

    for query in queries:
        embedding = query_embed_bm25(query)
        search_result = client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(**embedding.as_object()),
            limit=max(ks),
            using=BM25_MODEL_NAME,
        )
        top_types = [point.payload[query_payload_key] for point in search_result.points]

        for k in ks:
            top_k_types = top_types[:k]
            relevant_list = [1 if t == query else 0 for t in top_k_types]
            ap_at_k = average_precision_at_k(relevant_list, k)
            ap_scores_by_k[k].append(ap_at_k)

    return {k: np.mean(ap_scores_by_k[k]) for k in ks}


def qdrant_colbert_benchmark(
    client: QdrantClient,
    collection_name: str,
    queries: list[str],
    query_payload_key: str,
    ks: list[int] = [10],
) -> dict[int, float]:
    ap_scores_by_k = defaultdict(list)

    colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL_NAME)

    for query in queries:
        embedding = query_embed_colbert(colbert_model, query)
        search_result = client.query_points(collection_name, query=embedding, using=COLBERT_MODEL_NAME, limit=max(ks))
        top_types = [point.payload[query_payload_key] for point in search_result.points]

        for k in ks:
            top_k_types = top_types[:k]
            if len(top_k_types) != k:
                logger.info(len(top_k_types))
            relevant_list = [1 if t == query else 0 for t in top_k_types]
            ap_at_k = average_precision_at_k(relevant_list, k)
            ap_scores_by_k[k].append(ap_at_k)

    return {k: np.mean(ap_scores_by_k[k]) for k in ks}


def qdrant_triple_model_reranking_benchmark(
    client: QdrantClient,
    collection_name: str,
    model_name: str,
    device: str,
    prompt: str,
    queries: list[str],
    query_payload_key: str,
    ks: list[int] = [10],
) -> dict[int, float]:
    colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL_NAME)
    model = SentenceTransformer(model_name, device=device)

    ap_scores_by_k = defaultdict(list)

    for query in queries:
        late_embedding = query_embed_colbert(colbert_model, query)
        dense_embedding = embed_dense(model, sentences=query, prompt=prompt)
        sparse_embedding = query_embed_bm25(query)

        prefetch = [
            models.Prefetch(
                query=dense_embedding,
                using=model_name,
                limit=max(ks) * 2,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_embedding.as_object()),
                using=BM25_MODEL_NAME,
                limit=max(ks) * 2,
            ),
        ]

        search_result = client.query_points(
            collection_name,
            prefetch=prefetch,
            query=late_embedding,
            using=COLBERT_MODEL_NAME,
            limit=max(ks),
        )
        top_types = [point.payload[query_payload_key] for point in search_result.points]

        for k in ks:
            top_k_types = top_types[:k]
            relevant_list = [1 if t == query else 0 for t in top_k_types]
            ap_at_k = average_precision_at_k(relevant_list, k)
            ap_scores_by_k[k].append(ap_at_k)

    del model
    gc.collect()
    if device.lower().startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return {k: np.mean(ap_scores_by_k[k]) for k in ks}


def qdrant_bm25_1000_then_dense_benchmark(
    client: QdrantClient,
    collection_name: str,
    model_name: str,
    device: str,
    prompt: str,
    queries: list[str],
    query_payload_key: str,
    ks: list[int] = [10],
) -> dict[int, float]:
    model = SentenceTransformer(model_name, device=device)

    ap_scores_by_k = defaultdict(list)

    for query in queries:
        dense_embedding = embed_dense(model, sentences=query, prompt=prompt)
        sparse_embedding = query_embed_bm25(query)

        search_result = client.query_points(
            collection_name=collection_name,
            prefetch=models.Prefetch(
                query=models.SparseVector(**sparse_embedding.as_object()),
                using=BM25_MODEL_NAME,
                limit=1000,
            ),
            query=dense_embedding,
            using=model_name,
            limit=max(ks),
        )
        top_types = [point.payload[query_payload_key] for point in search_result.points]

        for k in ks:
            top_k_types = top_types[:k]
            relevant_list = [1 if t == query else 0 for t in top_k_types]
            ap_at_k = average_precision_at_k(relevant_list, k)
            ap_scores_by_k[k].append(ap_at_k)

    del model
    gc.collect()
    if device.lower().startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return {k: np.mean(ap_scores_by_k[k]) for k in ks}
