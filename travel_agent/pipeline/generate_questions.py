import random
import yaml
import logging
import pandas as pd
from pathlib import Path
from tqdm import trange


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_questions_df(df, params) -> pd.DataFrame:
    random.seed(params["seed"])

    unique_rubrics = set()
    for row in df["rubrics"].dropna():
        unique_rubrics.update(r.strip() for r in row.split(";"))

    addresses = df["address"].dropna().unique()

    logger.info("Unique rubrics: %d", len(unique_rubrics))
    logger.info("Unique addresses: %d", len(addresses))

    questions_data = []

    for _ in trange(params["n_questions"], desc="Processing questions", leave=False):
        question_type = random.choice(
            ["rubric", "rating", "address", "keyword", "combined"]
        )

        if question_type == "rubric":
            rubric = random.choice(list(unique_rubrics))
            q = f"Какие места в Москве относятся к категории «{rubric}»?"
            expected = (
                df[df["rubrics"].str.contains(rubric, na=False)]["name_ru"]
                .unique()
                .tolist()
            )

        elif question_type == "rating":
            q = "Какие заведения в Москве имеют рейтинг 5?"
            expected = df[df["rating"] >= 5.0]["name_ru"].unique().tolist()

        elif question_type == "address":
            addr = random.choice(addresses)
            q = f"Что интересного можно найти по адресу: {addr}?"
            expected = df[df["address"] == addr]["name_ru"].unique().tolist()

        elif question_type == "keyword":
            kw = random.choice(params["keywords"])
            q = f"Где в Москве в отзывах упоминается «{kw}»?"
            expected = (
                df[df["text"].str.contains(kw, case=False, na=False)]["name_ru"]
                .unique()
                .tolist()
            )

        elif question_type == "combined":
            rubric = random.choice(list(unique_rubrics))
            q = f"Найди заведения в Москве с рейтингом 5 и рубрикой «{rubric}»."
            subset = df[
                (df["rating"] >= 5.0) & (df["rubrics"].str.contains(rubric, na=False))
            ]
            expected = subset["name_ru"].unique().tolist()

        questions_data.append({"question": q, "expected_places": expected})

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
    logger.info("Saved questions to %s", output_path)


if __name__ == "__main__":
    main()
