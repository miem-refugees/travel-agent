from typing import Callable

# embeddings = embed_query(query)
# search_result = qdrant_search(client, collection_name, embeddings, max(ks))

# embeddings: dict[str, np.ndarray | list[float]] | np.ndarray | list[float]
# qdrant_search(client, collection_name, query, max(ks))


def make_single_vector_search(embed_fn: Callable, model: str) -> Callable:
    def search_fn(client, collection_name, query: str, k: int):
        embedding = embed_fn(query)
        return client.query_points(
            collection_name=collection_name,
            query=embedding,
            limit=k,
            using=model,
        )

    return search_fn
