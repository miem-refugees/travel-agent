from fastembed import SparseEmbedding, SparseTextEmbedding

BM25_MODEL_NAME = "Qdrant/bm25"
bm25_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME, language="russian")


def query_embed_bm25(query: str) -> SparseEmbedding:
    return next(bm25_model.query_embed(query))


def generate_bm25_embeddings(docs: list[str]) -> list[SparseEmbedding]:
    bm25_embeddings = list(bm25_model.embed(doc for doc in docs))
    return bm25_embeddings
