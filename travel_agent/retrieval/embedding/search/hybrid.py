# from qdrant_client import models
# from sentence_transformers import SentenceTransformer

# from travel_agent.retrieval.embedding.generation.dense import (
#     MODELS_PROMPTS,
#     embed_dense,
# )
# from travel_agent.retrieval.embedding.generation.sparse import (
#     BM25_MODEL_NAME,
#     query_embed_bm25,
# )

# device = "cpu"
# model_1_name = "sergeyzh/BERTA"
# model_2_name = "intfloat/multilingual-e5-small"

# model_1 = SentenceTransformer(model_1_name, device=device)
# model_2 = SentenceTransformer(model_2_name, device=device)


# def get_search_results(query):
#     sparse_embedding = query_embed_bm25(query)
#     embedding_1 = embed_dense(model_1, sentences=query, prompt=MODELS_PROMPTS[model_1_name].get("query"))
#     embedding_2 = embed_dense(model_2, sentences=query, prompt=MODELS_PROMPTS[model_2_name].get("query"))

#     search_result = client.query_points(
#         collection_name=collection_name,
#         prefetch=[
#             models.Prefetch(
#                 query=embedding_1,
#                 using=model_1_name,
#                 limit=40,
#             ),
#             models.Prefetch(
#                 query=embedding_2,
#                 using=model_2_name,
#                 limit=40,
#             ),
#             models.Prefetch(
#                 query=models.SparseVector(**sparse_embedding.as_object()),
#                 using=BM25_MODEL_NAME,
#                 limit=1000,
#             ),
#         ],
#         query=models.FusionQuery(fusion=models.Fusion.RRF),
#         limit=20,
#     )
#     return search_result


# if __name__ == "__main__":
#     pass
