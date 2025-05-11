import numpy as np
from sentence_transformers import SentenceTransformer

MODELS_PROMPTS = {
    "cointegrated/rubert-tiny2": {"query": None, "passage": None},
    "DeepPavlov/rubert-base-cased-sentence": {"query": None, "passage": None},
    "ai-forever/sbert_large_nlu_ru": {"query": None, "passage": None},
    "ai-forever/sbert_large_mt_nlu_ru": {"query": None, "passage": None},
    "sentence-transformers/distiluse-base-multilingual-cased-v1": {
        "query": None,
        "passage": None,
    },
    "sentence-transformers/distiluse-base-multilingual-cased-v2": {
        "query": None,
        "passage": None,
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "query": None,
        "passage": None,
    },
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "query": None,
        "passage": None,
    },
    "intfloat/multilingual-e5-large": {"query": "query: ", "passage": "passage: "},
    "intfloat/multilingual-e5-base": {"query": "query: ", "passage": "passage: "},
    "intfloat/multilingual-e5-small": {"query": "query: ", "passage": "passage: "},
    "ai-forever/ru-en-RoSBERTa": {
        "query": "search_query: ",
        "passage": "search_document: ",
    },
    # "ai-forever/FRIDA": {"query": "search_query: ", "passage": "search_document: "},
    "sergeyzh/BERTA": {"query": "search_query: ", "passage": "search_document: "},
}


def get_dynamic_batch_size(model: SentenceTransformer) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if num_params > 500_000_000:
        return 8
    if num_params > 200_000_000:
        return 32
    else:
        return 64


def embed_dense(
    model: SentenceTransformer,
    sentences: str | list[str],
    prompt: str | None,
    progress_bar: bool = False,
) -> np.ndarray:
    return model.encode(
        sentences,
        batch_size=get_dynamic_batch_size(model),
        prompt=prompt,
        show_progress_bar=progress_bar,
    )
