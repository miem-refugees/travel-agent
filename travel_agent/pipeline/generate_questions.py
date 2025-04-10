import random
import yaml
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm, trange


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

question_col = "question"
expected_places_col = "expected_places"

question_types = {"rubric", "rating", "address", "keyword", "rating_and_rubric"}


def generate_questions_df(df, params) -> pd.DataFrame:
    assert question_types == set(params["n_questions"].keys())

    random.seed(params["seed"])

    unique_rubrics = set()
    for row in df["rubrics"].dropna():
        unique_rubrics.update(r.strip().lower() for r in row.split(";"))
    unique_rubrics = sorted(
        unique_rubrics
    )  # обратно в list, чтобы избежать рандомного порядка

    addresses = df["address"].dropna().unique().tolist()
    ratings = [i for i in range(1, 6)]

    logger.info("Unique rubrics: %d", len(unique_rubrics))
    logger.info("Unique addresses: %d", len(addresses))

    questions_data = []

    for question_type, count in tqdm(params["n_questions"].items(), leave=False):
        for _ in trange(count, desc=f"Question type {question_type}", leave=False):
            if question_type == "rubric":
                rubric = random.choice(unique_rubrics)
                unique_rubrics.remove(rubric)

                q = f"Какие интересные места из категории {rubric} ты знаешь?"
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
                kw = random.choice(params["keywords"])
                params["keywords"].remove(kw)

                q = f"Где в отзывах упоминается «{kw}»?"
                expected = (
                    df[df["text"].str.contains(kw, case=False, na=False)]["name_ru"]
                    .unique()
                    .tolist()
                )

            elif question_type == "rating_and_rubric":
                rating = random.randint(1, 5)
                rubric = random.choice(list(unique_rubrics)).lower()

                q = f"Найди заведения {rubric} с рейтингом не ниже {rating}."
                subset = df[
                    (df["rating"] >= rating)
                    & (df["rubrics"].str.contains(rubric, na=False))
                ]
                expected = subset["name_ru"].unique().tolist()

            else:
                logger.error("unknown question_type: %s", question_type)

            questions_data.append({question_col: q, expected_places_col: expected})

    return pd.DataFrame(questions_data)


def main():
    params = yaml.safe_load(open("params.yaml"))["generate_questions"]
    logger.debug("Loaded params: %s", params)

    df = pd.read_csv(params["dataset"]).dropna()
    logger.info("Read dataset from %s", params["dataset"])

    questions_df = generate_questions_df(df, params)

    out_dir = Path(params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = f"{params['out_dir']}/{params['out_filename']}"

    questions_df.to_csv(output_path, index=False)
    logger.info("Saved questions to %s, logging result:", output_path)

    logger.info(
        "Logging result questions: \n%s",
        "\n".join(questions_df[question_col].to_list()),
    )


if __name__ == "__main__":
    main()
