import re
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
import yaml
from loguru import logger
from nltk.corpus import stopwords
from pydantic import BaseModel
from rake_nltk import Rake
from tqdm import tqdm


class Params(BaseModel):
    dataset: str
    out_dir: str
    top_n_phrases: int
    min_words: int


def extract_unique_rubrics(df) -> pd.DataFrame:
    df["rubrics_list"] = df["rubrics"].str.split(";")

    exploded = df.explode("rubrics_list")
    exploded["rubrics"] = exploded["rubrics_list"].str.strip()

    unique_rubrics_df = exploded["rubrics"].value_counts().reset_index()

    return unique_rubrics_df


def extract_top_phrases(df, top_n: int, min_words=3):
    logger.info("Downloading nltk russian stopwords")
    nltk.download("stopwords", quiet=True)
    russian_stopwords = stopwords.words("russian")

    rake = Rake(language="russian", stopwords=russian_stopwords)

    phrase_counter = Counter()

    for text in tqdm(
        df["text"].astype(str).tolist(),
        leave=False,
        desc="Extracting phrases with RAKE",
    ):
        rake.extract_keywords_from_text(text)
        raw_phrases = rake.get_ranked_phrases()

        cleaned_phrases = [
            re.sub(r"[^\w\s]", "", phrase).strip().lower()
            for phrase in raw_phrases
            if len(phrase.split()) >= min_words
        ]

        phrase_counter.update(cleaned_phrases)

    top_phrases = phrase_counter.most_common(top_n)
    result_df = pd.DataFrame(top_phrases, columns=["phrase", "count"])

    return result_df


def main():
    params = Params.model_validate(yaml.safe_load(open("params.yaml"))["analyze"])
    logger.debug("Loaded params: {}", params)

    df = pd.read_csv(params.dataset).dropna()
    logger.debug("Loaded dataset: {}", params.dataset)

    unique_rubrics_df = extract_unique_rubrics(df)
    logger.info("Unique rubrics: {}", len(unique_rubrics_df))

    top_phrases_df = extract_top_phrases(df, params.top_n_phrases, params.min_words)
    print(f"Extracted top {params.top_n_phrases} phrases")

    out_dir = Path(params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_rubrics_df.to_csv(f"{params.out_dir}/unique_rubrics.csv", index=False)
    logger.info("Saved unique unique_rubrics.csv")

    top_phrases_df.to_csv(f"{params.out_dir}/top_phrases.csv", index=False)
    logger.info("Saved top_phrases.csv")


if __name__ == "__main__":
    main()
