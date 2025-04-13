from pathlib import Path

import nltk
import pandas as pd
import yaml
from loguru import logger
from nltk.corpus import stopwords
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer


class Params(BaseModel):
    dataset: str
    out_dir: str
    top_n_phrases: int
    top_n_phrases_ngram: set[int]


def extract_unique_rubrics(df) -> pd.DataFrame:
    df["rubrics_list"] = df["rubrics"].str.split(";")

    exploded = df.explode("rubrics_list")
    exploded["rubrics"] = exploded["rubrics_list"].str.strip()

    unique_rubrics_df = exploded["rubrics"].value_counts().reset_index()

    return unique_rubrics_df


def extract_top_phrases(df, top_n_phrases: int, ngrap_range: tuple):
    logger.info("Downloading nltk russian stopwords")
    nltk.download("stopwords")
    russian_stopwords = stopwords.words("russian")

    vectorizer = TfidfVectorizer(
        stop_words=russian_stopwords,
        ngram_range=ngrap_range,
        lowercase=True,
        min_df=3,
        max_df=0.8,  # Filter too popular phrases
    )

    X = vectorizer.fit_transform(df["text"].astype(str).tolist())
    ngram_freq = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0])

    sorted_ngrams = sorted(ngram_freq, key=lambda x: x[1], reverse=True)[:top_n_phrases]
    result_df = pd.DataFrame(sorted_ngrams, columns=["phrase", "count"])
    return result_df


def main():
    params = Params.model_validate(yaml.safe_load(open("params.yaml"))["analyze"])
    logger.debug("Loaded params: {}", params)

    df = pd.read_csv(params.dataset).dropna()
    logger.debug("Loaded dataset: {}", params.dataset)

    unique_rubrics_df = extract_unique_rubrics(df)
    logger.info("Unique rubrics: {}", len(unique_rubrics_df))

    top_phrases_df = extract_top_phrases(
        df, params.top_n_phrases, tuple(params.top_n_phrases_ngram)
    )
    print(f"Extracted top {params.top_n_phrases} phrases")

    out_dir = Path(params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_rubrics_df.to_csv(f"{params.out_dir}/unique_rubrics.csv", index=False)
    logger.info("Saved unique unique_rubrics.csv")

    top_phrases_df.to_csv(f"{params.out_dir}/top_phrases.csv", index=False)
    logger.info("Saved top_phrases.csv")


if __name__ == "__main__":
    main()
