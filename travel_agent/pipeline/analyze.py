import re
from collections import Counter, defaultdict
from pathlib import Path

import nltk
import pandas as pd
import yaml
from loguru import logger
from nltk.corpus import stopwords
from pydantic import BaseModel


class Params(BaseModel):
    dataset: str
    out_dir: str
    top_n: int


custom_stopwords = {
    "очень",
    "спасибо",
    "это",
    "всё",
    "все",
    "просто",
    "быстро",
    "рекомендую",
    "огромное",
    "всем",
    "молодцы",
    "супер",
    "хочу",
    "ещё",
}


def clean_review(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|<.*?>", " ", text)  # Removing URLs
    text = (
        text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    )  # Handling escaped characters
    text = re.sub(
        r"[^а-яa-zё0-9\s]", " ", text
    )  # Remove non-Russian and non-alphanumeric characters
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text


def extract_unique_rubrics(df) -> pd.DataFrame:
    df["rubrics_list"] = df["rubrics"].str.split(";")

    exploded = df.explode("rubrics_list")
    exploded["rubrics"] = exploded["rubrics_list"].str.strip()

    unique_rubrics_df = exploded["rubrics"].value_counts().reset_index()

    return unique_rubrics_df


def extract_top_by_rubrics(df, top_n_words) -> pd.DataFrame:
    logger.info("Loading nltk stopwords...")
    nltk.download("stopwords", quiet=True)
    russian_stopwords = set(stopwords.words("russian"))
    for word in custom_stopwords:
        russian_stopwords.add(word)

    df["text"] = df["text"].apply(clean_review)

    exploded = df.explode("rubrics_list")
    exploded["rubrics"] = exploded["rubrics_list"].str.strip()

    df = exploded.groupby("rubrics")["text"].apply(" ".join).reset_index()

    def count_words(text: str):
        words = re.findall(r"\b[а-яё]+\b", text)
        words = [
            word for word in words if word not in russian_stopwords and len(word) > 1
        ]
        return Counter(words)

    rubric_word_counts = []

    for _, row in df.iterrows():
        rubric = row["rubrics"]
        text = row["text"]
        word_count = count_words(text)
        rubric_word_counts.append((rubric, word_count))

    rubric_top_words = defaultdict(list)
    for rubric, word_count in rubric_word_counts:
        top_n = word_count.most_common(top_n_words)
        top_words = [word for word, _ in top_n]  # List of most common words
        rubric_top_words[rubric].extend(top_words)  # Aggregate words for each rubric

    top_words_list = []
    for rubric, words in rubric_top_words.items():
        top_words_list.append(
            {
                "rubric": rubric,
                "words": ", ".join(words),  # Join words with a comma
            }
        )

    top_words_df = pd.DataFrame(top_words_list)
    return top_words_df


def main():
    params = Params.model_validate(yaml.safe_load(open("params.yaml"))["analyze"])
    logger.debug("Loaded params: {}", params)

    df = pd.read_csv(params.dataset).dropna()
    logger.debug("Loaded dataset: {}", params.dataset)

    unique_rubrics_df = extract_unique_rubrics(df)
    logger.info("Unique rubrics: {}", len(unique_rubrics_df))

    top_by_rubrics = extract_top_by_rubrics(df, params.top_n)
    logger.info("Top words by rubric done")

    out_dir = Path(params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_rubrics_df.to_csv(f"{params.out_dir}/unique_rubrics.csv", index=False)
    logger.info("Saved unique unique_rubrics.csv")

    top_by_rubrics.to_csv(f"{params.out_dir}/top_by_rubrics.csv", index=False)
    logger.info("Saved top_by_rubrics.csv")


if __name__ == "__main__":
    main()
