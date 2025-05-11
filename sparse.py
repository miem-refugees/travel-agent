from fastembed import SparseEmbedding, SparseTextEmbedding

BM25_MODEL_NAME = "Qdrant/bm25"
bm25_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME, language="russian")


def query_embed_bm25(query: str) -> SparseEmbedding:
    return next(bm25_model.query_embed(query))
