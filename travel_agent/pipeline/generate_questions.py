import random
from pydantic import BaseModel
import yaml
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm, trange


question_col = "question"
expected_places_col = "expected_places"

question_types = {"rubric", "rating", "address", "keyword", "rating_and_rubric"}


class Params(BaseModel):
    seed: int
    dataset: str
    out_dir: str
    out_filename: str
    n_questions: dict[str, int]
    keywords: list[str]


def generate_questions_df(df, params) -> pd.DataFrame:
    assert question_types == set(params.n_questions.keys())

    random.seed(params.seed)

    unique_rubrics = set()
    for row in df["rubrics"].dropna():
        unique_rubrics.update(r.strip().lower() for r in row.split(";"))
    unique_rubrics = sorted(
        unique_rubrics
    )  # обратно в list, чтобы избежать рандомного порядка

    addresses = sorted(df["address"].dropna().unique().tolist())
    ratings = [i for i in range(1, 6)]

    logger.info("Unique rubrics: {}", len(unique_rubrics))
    logger.info("Unique addresses: {}", len(addresses))

    questions_data = []

    for question_type, count in tqdm(params.n_questions.items(), leave=False):
        for _ in trange(count, desc=f"Question type {question_type}", leave=False):
            if question_type == "rubric":
                rubric = random.choice(unique_rubrics)
                unique_rubrics.remove(rubric)

                q = f"Какие {rubric} ты знаешь?"
                expected = (
                    df[df["rubrics"].str.contains(rubric, na=False)]["name_ru"]
                    .unique()
                    .tolist()
                )

            elif question_type == "rating":
                rating = random.choice(ratings)
                ratings.remove(rating)

                if len(ratings) == 0:
                    continue

                q = f"Покажи заведения с рейтингом {rating} и больше."
                expected = df[df["rating"] >= rating]["name_ru"].unique().tolist()

            elif question_type == "address":
                addr = random.choice(addresses)
                addresses.remove(addr)

                q = f"Какие заведения есть по адресу {addr}?"
                expected = df[df["address"] == addr]["name_ru"].unique().tolist()

            elif question_type == "keyword":
                kw = random.choice(params.keywords)
                params.keywords.remove(kw)

                q = f"Где в отзывах упоминается «{kw}»?"
                expected = (
                    df[df["text"].str.contains(kw, case=False, na=False)]["name_ru"]
                    .unique()
                    .tolist()
                )

            elif question_type == "rating_and_rubric":
                rating = random.randint(1, 5)
                rubric = random.choice(unique_rubrics).lower()

                q = f"Найди заведения {rubric} с рейтингом не ниже {rating}."
                subset = df[
                    (df["rating"] >= rating)
                    & (df["rubrics"].str.contains(rubric, na=False))
                ]
                expected = subset["name_ru"].unique().tolist()

            else:
                logger.error("unknown question_type: {}", question_type)
                continue

            questions_data.append({question_col: q, expected_places_col: expected})

    return pd.DataFrame(questions_data)


def main():
    params = Params.model_validate(
        yaml.safe_load(open("params.yaml"))["generate_questions"]
    )
    logger.debug("Loaded params: {}", params)

    df = pd.read_csv(params.dataset).dropna()
    logger.info("Read dataset from {}", params.dataset)

    questions_df = generate_questions_df(df, params)

    out_dir = Path(params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = f"{params.out_dir}/{params.out_filename}"

    questions_df.to_csv(output_path, index=False)
    logger.info("Saved questions to {}", output_path)

    logger.info(
        "Logging result: \n- {}", "\n- ".join(questions_df[question_col].to_list())
    )


if __name__ == "__main__":
    main()
