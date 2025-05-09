import sys
from unicodedata import category

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")


punctuation_chars_numbers = set(
    [chr(i) for i in range(sys.maxunicode) if category(chr(i)).startswith("P") or category(chr(i)).startswith("Nd")]
)
stemmer = SnowballStemmer("russian")
stop_words = stopwords.words("russian")


def apply_stemmer(sentence: str) -> str:
    sentence = sentence.lower()
    sentence = "".join([char for char in sentence if char not in punctuation_chars_numbers])
    tokens = word_tokenize(sentence, language="russian")
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
    return " ".join(stemmed_words)


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
