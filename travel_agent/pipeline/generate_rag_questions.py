import argparse
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm, trange

from .analyze import extract_unique_rubrics

question_col = "question"
expected_places_col = "expected_places"


class Params(BaseModel):
    out_dir: str
    n_questions: dict[str, int]
    keywords: list[str]


def generate_questions_df(df, params) -> pd.DataFrame:
    unique_rubrics = extract_unique_rubrics(df).rubrics.tolist()

    addresses = sorted(df["address"].dropna().unique().tolist())
    ratings = [i for i in range(1, 6)]
    keywords = params.keywords
    params.n_questions["keyword"] = len(keywords)

    logger.info("Unique rubrics: {}", len(unique_rubrics))
    logger.info("Unique addresses: {}", len(addresses))
    logger.info("Keywords: {}", len(keywords))

    questions_data = []

    for question_type, count in tqdm(params.n_questions.items(), leave=False):
        for i in trange(count, desc=f"Question type {question_type}", leave=False):
            if question_type == "rubric":
                if i >= len(unique_rubrics):
                    continue
                rubric = unique_rubrics[i]

                q = f"Какие {rubric} ты знаешь?"
                expected = df[df["rubrics"].str.contains(rubric, na=False)]["name_ru"].unique().tolist()

            elif question_type == "rating":
                if i >= len(ratings):
                    continue
                rating = ratings[i]

                q = f"Покажи заведения с рейтингом {rating} и больше."
                expected = df[df["rating"] >= rating]["name_ru"].unique().tolist()

            elif question_type == "address":
                if i >= len(addresses):
                    continue
                addr = addresses[i]

                q = f"Какие заведения есть по адресу {addr}?"
                expected = df[df["address"] == addr]["name_ru"].unique().tolist()

            elif question_type == "keyword":
                if i >= len(keywords):
                    continue
                kw = keywords[i].strip().lower()

                q = f"Покажи мне места, в которых упоминается {kw}"
                expected = df[df["text"].str.contains(kw, case=False, na=False)]["name_ru"].unique().tolist()

            elif question_type == "rating_and_rubric":
                if len(unique_rubrics) == 0:
                    continue
                rating = 1 + (i % 5)
                rubric = unique_rubrics[i % len(unique_rubrics)]

                q = f"Найди заведения {rubric} с рейтингом не ниже {rating}."
                subset = df[(df["rating"] >= rating) & (df["rubrics"].str.contains(rubric, na=False))]
                expected = subset["name_ru"].unique().tolist()

            else:
                logger.error("unknown question_type: {}", question_type)
                continue

            if len(expected) == 0:
                logger.warning("No expected places for question: {}", q)
                continue

            questions_data.append({question_col: q, expected_places_col: expected})

    return pd.DataFrame(questions_data)


def main():
    params = Params.model_validate(yaml.safe_load(open("params.yaml"))["generate_rag_questions"])
    logger.debug("Loaded params: {}", params)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="csv-dataset path")

    args = parser.parse_args()
    input_data_path = args.dataset

    df = pd.read_csv(input_data_path).dropna()
    logger.info("Read dataset from {}", input_data_path)

    questions_df = generate_questions_df(df, params)

    out_dir = Path(params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = f"{params.out_dir}/{Path(input_data_path).name}"
    questions_df.to_csv(output_path, index=False)
    logger.info("Saved {} questions to {}", questions_df.size, output_path)


if __name__ == "__main__":
    main()
