import gc
from typing import Optional

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

MODELS_PROMPTS = {
    "cointegrated/rubert-tiny2": {"query": None, "passage": None},
    # "DeepPavlov/rubert-base-cased-sentence": {"query": None, "passage": None},
    # "ai-forever/sbert_large_nlu_ru": {"query": None, "passage": None},
    # "ai-forever/sbert_large_mt_nlu_ru": {"query": None, "passage": None},
    # "sentence-transformers/distiluse-base-multilingual-cased-v1": {
    #     "query": None,
    #     "passage": None,
    # },
    # "sentence-transformers/distiluse-base-multilingual-cased-v2": {
    #     "query": None,
    #     "passage": None,
    # },
    # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
    #     "query": None,
    #     "passage": None,
    # },
    # "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
    #     "query": None,
    #     "passage": None,
    # },
    # "intfloat/multilingual-e5-large": {"query": "query: ", "passage": "passage: "},
    # "intfloat/multilingual-e5-base": {"query": "query: ", "passage": "passage: "},
    # "intfloat/multilingual-e5-small": {"query": "query: ", "passage": "passage: "},
    # "ai-forever/ru-en-RoSBERTa": {
    #     "query": "search_query: ",
    #     "passage": "search_document: ",
    # },
    # # "ai-forever/FRIDA": {"query": "search_query: ", "passage": "search_document: "},
    # "sergeyzh/BERTA": {"query": "search_query: ", "passage": "search_document: "},
}


def get_models_params_embedding_dim(
    models_prompts: dict[str, dict[str, Optional[str]]],
) -> dict[str, dict[str, str | int]]:
    models_params_embedding_dim = {}
    for model_name in models_prompts:
        model = SentenceTransformer(model_name, device="cpu")
        embedding_dim = model.encode("Test string").shape[0]
        num_params = sum(p.numel() for p in model.parameters())
        models_params_embedding_dim[model_name] = {}
        models_params_embedding_dim[model_name]["embedding_dim"] = embedding_dim
        models_params_embedding_dim[model_name]["num_params"] = num_params
    return models_params_embedding_dim


def get_dynamic_batch_size(model: SentenceTransformer) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if num_params > 500_000_000:
        return 8
    if num_params > 200_000_000:
        return 32
    else:
        return 64


def embed_dense(model: SentenceTransformer, sentences: str | list[str], prompt: str, progress_bar: bool = False) -> np.ndarray:
    return model.encode(
        sentences,
        batch_size=get_dynamic_batch_size(model),
        prompt=prompt,
        show_progress_bar=progress_bar,
    )


def generate_dense_models_embeddings(
    docs: list[str], models_prompts: dict[str, dict[str, Optional[str]]], device: str
) -> dict[str, np.ndarray]:
    dense_embeddings = {}

    for model_name in models_prompts:
        logger.info(f"Generating embeddings using {model_name}")
        model = SentenceTransformer(model_name, device=device)

        doc_embeddings = embed_dense(model, docs, models_prompts[model_name].get("passage"), True)

        dense_embeddings[model_name] = doc_embeddings

        del model
        gc.collect()
        if device.lower().startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return dense_embeddings
