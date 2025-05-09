import sys
from unicodedata import category

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize



def average_precision_at_k(relevant_list: list[int], k: int) -> float:
    score = 0.0
    num_hits = 0
    for i, rel in enumerate(relevant_list[:k]):
        if rel:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / min(k, sum(relevant_list)) if sum(relevant_list) > 0 else 0.0

def preprocess_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("\\n", " ")
    text = text.replace("\t", " ")
    # text = text.replace("-", "")
    # text = text.replace(";", " ")
    return text
import gc
import torch
def clean_up_model(model, device):
    del model
    gc.collect()
    if device.lower().startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()