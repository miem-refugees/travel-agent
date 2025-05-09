from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from pydantic import BaseModel
from transliterate import slugify


class Params(BaseModel):
    dataset: str
    out_dir: str
    cities: list[str]
    random_state: int


def preprocess_df(df):
    processed_df = df.dropna(subset=["name_ru"])

    logger.info("Removed {} NaN rows", df.shape[0] - processed_df.shape[0])

    deduplicated = processed_df.drop_duplicates(subset=["address", "name_ru", "text"])

    logger.info("Deduplicated {} rows", processed_df.shape[0] - deduplicated.shape[0])

    return deduplicated


def save_city_prefix_dataset(df, city, out_dir):
    logger.info("Creating '{}'-prefix dataset", city)

    city_df = df[df["address"].str.startswith(city, na=False)]

    filename = f"{slugify(city)}.csv"
    city_df.to_csv(out_dir / filename, index=False)

    logger.info("Created {}", filename)


def save_norm_rubrics_distribution(df, size, out_dir, random_state):
    logger.info("Creating {}-sized rubrics distribution", size)

    df = df.dropna(subset=["rubrics"])
    df["rubrics_list"] = df["rubrics"].str.split(";")

    exploded = df.explode("rubrics_list")
    exploded["rubrics_list"] = exploded["rubrics_list"].str.strip()

    unique_rubrics = exploded["rubrics_list"].value_counts()
    n_rubrics = len(unique_rubrics)
    max_per_rubric = size // n_rubrics

    balanced = exploded.groupby("rubrics_list", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max_per_rubric), random_state=random_state),
        include_groups=False,
    )

    result_df = balanced.drop_duplicates(subset=["address", "name_ru", "text"])

    filename = f"normal_rubrics_{result_df.shape[0]}.csv"
    result_df.to_csv(out_dir / filename, index=False)

    logger.info("Created {} rows in {}", result_df.shape[0], filename)


def main():
    params = Params.model_validate(yaml.safe_load(open("params.yaml"))["prepare"])
    logger.debug("Loaded params: {}", params)

    df = pd.read_csv(params.dataset)
    df = preprocess_df(df)

    out_dir = Path(params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for city in params.cities:
        save_city_prefix_dataset(df, city, out_dir)

    save_norm_rubrics_distribution(df, 20000, out_dir, params.random_state)


if __name__ == "__main__":
    main()
